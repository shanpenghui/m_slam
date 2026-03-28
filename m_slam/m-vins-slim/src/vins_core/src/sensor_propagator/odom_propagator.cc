#include "sensor_propagator/odom_propagator.h"

#include "math_common/math.h"

namespace vins_core {

void OdomPropagator::Propagate(
        const common::OdomDatas& prop_data,
        common::State* state_ptr,
        Eigen::Matrix<double, 9, 9>* Phi_summed_ptr,
        Eigen::Matrix<double, 9, 9>* Qd_summed_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);

    if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
        Phi_summed_ptr->setIdentity();
        Qd_summed_ptr->setZero();
    }

    // Loop through all odom messages, and use them to move the state forward in time
    // This uses the zero'th order quat, and then constant acceleration discrete
    if(prop_data.size() > 1u) {
        for(size_t i = 0u; i<prop_data.size() - 1u; ++i) {
            if (prop_data[i + 1u].timestamp_ns == prop_data[i].timestamp_ns) {
                continue;
            }
            CHECK_GT(prop_data[i + 1u].timestamp_ns, prop_data[i].timestamp_ns);
            if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
                Eigen::Matrix<double, 9, 9>& Phi_summed = *Phi_summed_ptr;
                Eigen::Matrix<double, 9, 9>& Qd_summed = *Qd_summed_ptr;
                // Get the next state Jacobian and noise Jacobian for this odom reading
                Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
                Eigen::Matrix<double, 9, 9> Qdi = Eigen::Matrix<double, 9, 9>::Zero();
                PredictAndCompute(prop_data[i], prop_data[i + 1u], &state, &F, &Qdi);
                // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
                // NOTE: Phi_summed = Phi_i*Phi_summed
                // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
                Phi_summed = F * Phi_summed;
                Qd_summed = F * Qd_summed * F.transpose() + Qdi;
            } else {
                PredictAndCompute(prop_data[i], prop_data[i + 1u], &state);
            }
        }
    }
    if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
        CHECK(!Phi_summed_ptr->hasNaN());
        CHECK(!Qd_summed_ptr->hasNaN());
        VLOG(5) << "Phi: " << std::endl << *Phi_summed_ptr;
        VLOG(5) << "Q: " << std::endl << *Qd_summed_ptr;
    }
}

void OdomPropagator::PredictAndCompute(
        const common::OdomData& data_minus,
        const common::OdomData& data_plus,
        common::State* state_ptr,
        Eigen::Matrix<double, 9, 9>* F_ptr,
        Eigen::Matrix<double, 9, 9>* Qd_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);

    const aslam::Transformation T_OmtoG(data_minus.q, data_minus.p);
    const aslam::Transformation T_OptoG(data_plus.q, data_plus.p);
    const Eigen::Vector2d& bt = state.bt;
    const double br = state.br;

    const aslam::Transformation T_OptoOm = T_OmtoG.inverse() * T_OptoG;
    const Eigen::Matrix3d R_OptoOm = T_OptoOm.getRotationMatrix();
    const Eigen::Vector3d p_OpinOm = T_OptoOm.getPosition();

    // Compute bias.
    Eigen::Vector3d p_bias = Eigen::Vector3d::Zero();
    p_bias(0) = bt(0) * std::abs(p_OpinOm(0));
    p_bias(1) = bt(1) * std::abs(p_OpinOm(1));

    Eigen::AngleAxisd angle_axis;
    angle_axis.fromRotationMatrix(R_OptoOm);
    const double angle_norm = std::abs(angle_axis.angle());

    Eigen::Matrix3d R_bias = Eigen::Matrix3d::Identity();
    R_bias.block<2, 2>(0, 0) << std::cos(br * angle_norm), -std::sin(br * angle_norm),
                                std::sin(br * angle_norm), std::cos(br * angle_norm);
    const Eigen::Quaterniond q_bias(R_bias);

    // Propagate.
    aslam::Transformation& T_OtoG_propagated = state.T_OtoG;
    T_OtoG_propagated = T_OtoG_propagated * T_OptoOm;

    // Apply bias.
    Eigen::Quaterniond q_OtoG_propagated = T_OtoG_propagated.getEigenQuaternion();
    Eigen::Vector3d p_OtoG_propagated = T_OtoG_propagated.getPosition();

    q_OtoG_propagated = q_OtoG_propagated * q_bias;
    q_OtoG_propagated.normalize();
    q_OtoG_propagated = common::Positify(q_OtoG_propagated);
    p_OtoG_propagated += T_OmtoG.getRotationMatrix() * p_bias;

    int p_id = 0;  // position
    int r_id = 3; // quaternion
    int bt_id = 6;  // bt
    int br_id = 8;  // br

    if (F_ptr != nullptr && Qd_ptr != nullptr) {
        Eigen::Matrix<double, 9, 9>& F = *CHECK_NOTNULL(F_ptr);
        Eigen::Matrix<double, 9, 9>& Qd = *CHECK_NOTNULL(Qd_ptr);

        F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(p_id, r_id) = -T_OmtoG.getRotationMatrix() * common::skew_x(p_OpinOm);
        F.block<3, 2>(p_id, bt_id) = T_OmtoG.getRotationMatrix().block<3, 2>(0, 0);
        F.block<3, 1>(p_id, br_id) = Eigen::Vector3d::Zero();

        F.block<3, 3>(r_id, p_id) = Eigen::Matrix3d::Zero();
        F.block<3, 3>(r_id, r_id) = R_OptoOm.transpose();
        F.block<3, 2>(r_id, bt_id) = Eigen::Matrix<double, 3, 2>::Zero();
        F.block<3, 1>(r_id, br_id) = Eigen::Vector3d::UnitZ();

        Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Identity();
        G.block<3, 3>(p_id, p_id) = 0.5 * T_OmtoG.getRotationMatrix() +
                0.5 * T_OptoG.getRotationMatrix();
        const Eigen::Vector3d b_tmp = (0.5 * Eigen::Matrix3d::Identity() + 0.5 * R_OptoOm).inverse() *
                p_OpinOm;
        G.block<3, 3>(p_id, r_id) = -0.5 * T_OptoG.getRotationMatrix() * common::skew_x(b_tmp);

        Eigen::Vector3d scale_p(0.05, 0.01, 0.002);
        Eigen::Vector3d scale_q(0.002, 0.002, 0.05);

        Eigen::Vector3d p_sigma = scale_p.cwiseProduct(p_OpinOm.cwiseAbs());
        Eigen::Vector3d q_sigma = scale_q * angle_norm;

        const double kMinPSigma = 4e-4;
        p_sigma = (p_sigma.array() > kMinPSigma).select(p_sigma.array(), kMinPSigma);
        const double kMinQSigma = 1.0 * common::kDegToRad;
        q_sigma = (q_sigma.array() > kMinQSigma).select(q_sigma.array(), kMinQSigma);

        Eigen::Vector3d bias_sigma(1e-5, 1e-5, 1e-5);
        Eigen::Matrix<double, 9, 1> full_sigma;
        full_sigma.head<3>() = p_sigma;
        full_sigma.segment<3>(3) = q_sigma;
        full_sigma.segment<3>(6) = bias_sigma;

        Qd = G * full_sigma.cwiseProduct(full_sigma).asDiagonal() * G.transpose();
    }
}
}
