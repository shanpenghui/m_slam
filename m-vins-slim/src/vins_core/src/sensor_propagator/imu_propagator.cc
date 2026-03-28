#include "sensor_propagator/imu_propagator.h"

#include <unsupported/Eigen/MatrixFunctions>

#include "math_common/math.h"
#include "sensor_propagator/odom_propagator.h"

namespace vins_core {

ImuPropagator::ImuPropagator(const NoiseManager& noises)
    : noises_(noises), gravity_(Eigen::Vector3d(0.0, 0.0, 9.7940)) {

    Qc_.block<3,3>(0, 0) = noises_.sigma_acc_2 * Eigen::Matrix<double,3,3>::Identity();
    Qc_.block<3,3>(3, 3) = noises_.sigma_gyro_2 * Eigen::Matrix<double,3,3>::Identity();
    Qc_.block<3,3>(6, 6) = noises_.sigma_acc_2 * Eigen::Matrix<double,3,3>::Identity();
    Qc_.block<3,3>(9, 9) = noises_.sigma_gyro_2 * Eigen::Matrix<double,3,3>::Identity();
    Qc_.block<3,3>(12, 12) = noises_.sigma_bg_2 * Eigen::Matrix<double,3,3>::Identity();
    Qc_.block<3,3>(15, 15) = noises_.sigma_ba_2 * Eigen::Matrix<double,3,3>::Identity();
}

void ImuPropagator::PredictMeanDiscrete(const double dt,
                                        const Eigen::Vector3d& gyro_hat1, const Eigen::Vector3d& acc_hat1,
                                        const Eigen::Vector3d& gyro_hat2, const Eigen::Vector3d& acc_hat2,
                                        const common::State& state,
                                        Eigen::Quaterniond* new_q_ptr,
                                        Eigen::Vector3d* new_v_ptr,
                                        Eigen::Vector3d* new_p_ptr) {
    Eigen::Quaterniond& new_q = *CHECK_NOTNULL(new_q_ptr);
    Eigen::Vector3d& new_v = *CHECK_NOTNULL(new_v_ptr);
    Eigen::Vector3d& new_p = *CHECK_NOTNULL(new_p_ptr);

    // If we are averaging the IMU, then do so
    Eigen::Vector3d gyro_hat = 0.5 * (gyro_hat1 + gyro_hat2);
    Eigen::Vector3d acc_hat = 0.5 * (acc_hat1 + acc_hat2);

    // NOTE(chien): In our system, IMU measurements has been rotated to the odom frame.
    // So T_OtoG equal T_ItoG.
    const aslam::Transformation& T_ItoG = state.T_OtoG;
    const Eigen::Quaterniond& q_ItoG = T_ItoG.getEigenQuaternion();
    const Eigen::Matrix<double,3,3> R_ItoG = q_ItoG.toRotationMatrix();

    Eigen::Quaterniond q_plus(1, gyro_hat(0) * dt * 0.5, gyro_hat(1) * dt * 0.5, gyro_hat(2) * dt * 0.5);
    q_plus.normalize();
    q_plus = common::Positify(q_plus);
    new_q = q_ItoG * q_plus;
    new_q.normalize();
    new_q = common::Positify(new_q);

    const Eigen::Vector3d acc_g_hat = R_ItoG * acc_hat - gravity_;
    // Velocity: just the acceleration in the local frame, minus global gravity
    new_v = state.velocity + acc_g_hat * dt;

    // Position: just velocity times dt, with the acceleration integrated twice
    new_p = T_ItoG.getPosition() + state.velocity * dt + 0.5 * acc_g_hat * dt * dt;
}

void ImuPropagator::PredictAndCompute(const common::ImuData& data_minus,
                                      const common::ImuData& data_plus,
                                      common::State* state_ptr,
                                      Eigen::Matrix<double, 15, 15>* F_ptr,
                                      Eigen::Matrix<double, 15, 15>* Qd_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);
    // NOTE(chien): In our system, IMU measurements has been rotated to the odom frame.
    // So T_OtoG equal T_ItoG.
    const aslam::Transformation& T_ItoG = state.T_OtoG;

    // Time elapsed over interval
    double dt = common::NanoSecondsToSeconds(data_plus.timestamp_ns - data_minus.timestamp_ns);
    CHECK_GT(dt, 0.0);
    // Corrected imu measurements
    Eigen::Matrix<double,3,1> w_hat1 = data_minus.gyro - state.bg;
    Eigen::Matrix<double,3,1> a_hat1 = data_minus.acc - state.ba;
    Eigen::Matrix<double,3,1> w_hat2 = data_plus.gyro - state.bg;
    Eigen::Matrix<double,3,1> a_hat2 = data_plus.acc - state.ba;
    Eigen::Matrix<double,3,1> w_hat12 = 0.5 * (w_hat1 + w_hat2);

    // Compute the new state mean value
    Eigen::Quaterniond new_q;
    Eigen::Vector3d new_v, new_p;
    PredictMeanDiscrete(dt, w_hat1, a_hat1, w_hat2, a_hat2, state, &new_q, &new_v, &new_p);

    // Get the locations of each entry of the imu state
    int p_id = 0;  // position
    int th_id = 3; // quaternion
    int v_id = 6;  // velocity
    int bg_id = 9;  // bg
    int ba_id = 12;  // ba

    if (F_ptr != nullptr && Qd_ptr != nullptr) {
        Eigen::Matrix<double, 15, 15>& F = *CHECK_NOTNULL(F_ptr);
        Eigen::Matrix<double, 15, 15>& Qd = *CHECK_NOTNULL(Qd_ptr);
        // Now compute Jacobian of new state wrt old state and noise
        Eigen::Matrix<double,3,3> R_ItoG1 = T_ItoG.getRotationMatrix();
        Eigen::Matrix<double,3,3> R_ItoG2 = new_q.toRotationMatrix();

        F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(p_id, th_id) = -0.25 * R_ItoG1 * common::skew_x(a_hat1) * dt * dt -
                              0.25 * R_ItoG2 * common::skew_x(a_hat2) *
                              (Eigen::Matrix3d::Identity() - common::skew_x(w_hat12) * dt) * dt * dt;
        F.block<3, 3>(p_id, v_id) = Eigen::MatrixXd::Identity(3,3) * dt;
        F.block<3, 3>(p_id, bg_id) = 0.25 * R_ItoG2 * common::skew_x(a_hat2) * dt * dt * dt;
        F.block<3, 3>(p_id, ba_id) = -0.25 * (R_ItoG1 + R_ItoG2) * dt * dt;
        F.block<3, 3>(th_id, th_id) = Eigen::Matrix3d::Identity() - common::skew_x(w_hat12) * dt;
        F.block<3, 3>(th_id, bg_id) = -Eigen::MatrixXd::Identity(3,3) * dt;
        F.block<3, 3>(v_id, th_id) = -0.5 * R_ItoG1 * common::skew_x(a_hat1) * dt -
                              0.5 * R_ItoG2 * common::skew_x(a_hat2) *
                              (Eigen::Matrix3d::Identity() - common::skew_x(w_hat12) * dt) * dt;
        F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(v_id, bg_id) = 0.5 * R_ItoG2 * common::skew_x(a_hat2) * dt * dt;
        F.block<3, 3>(v_id, ba_id) = -0.5 * (R_ItoG1 + R_ItoG2) * dt;
        F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();


        Eigen::Matrix<double,15,18> G = Eigen::Matrix<double,15,18>::Zero();
        G.block<3, 3>(p_id, 0) =  0.25 * R_ItoG1 * dt;
        G.block<3, 3>(p_id, 3) =  0.25 * -R_ItoG2 * common::skew_x(a_hat2) * dt * dt * 0.5;
        G.block<3, 3>(p_id, 6) =  0.25 * R_ItoG2 * dt;
        G.block<3, 3>(p_id, 9) =  G.block<3, 3>(p_id, 3);
        G.block<3, 3>(th_id, 3) =  0.5 * Eigen::Matrix3d::Identity();
        G.block<3, 3>(th_id, 9) =  0.5 * Eigen::Matrix3d::Identity();
        G.block<3, 3>(v_id, 0) =  0.5 * R_ItoG1;
        G.block<3, 3>(v_id, 3) =  0.5 * -R_ItoG2 * common::skew_x(a_hat2) * dt * 0.5;
        G.block<3, 3>(v_id, 6) =  0.5 * R_ItoG2;
        G.block<3, 3>(v_id, 9) =  G.block<3, 3>(v_id, 3);
        G.block<3, 3>(9, 12) = Eigen::Matrix3d::Identity();
        G.block<3, 3>(12, 15) = Eigen::Matrix3d::Identity();

        // Compute the noise injected into the state over the interval
        Qd = G * Qc_ * G.transpose() * dt;
    }

    // Now replace imu estimate with propagated values
    state.T_OtoG.update(new_q, new_p);
    state.velocity = new_v;
}

void ImuPropagator::Propagate(const common::ImuDatas& prop_data,
                              common::State* state_ptr,
                              Eigen::Matrix<double, 15, 15>* Phi_summed_ptr,
                              Eigen::Matrix<double, 15, 15>* Qd_summed_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);
    if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
        Phi_summed_ptr->setIdentity();
        Qd_summed_ptr->setZero();
    }

    // Loop through all IMU messages, and use them to move the state forward in time
    // This uses the zero'th order quat, and then constant acceleration discrete
    if(prop_data.size() > 1u) {
        for(size_t i = 0u; i<prop_data.size() - 1u; ++i) {
            if (prop_data[i + 1u].timestamp_ns == prop_data[i].timestamp_ns) {
                continue;
            }
            CHECK_GT(prop_data[i + 1u].timestamp_ns, prop_data[i].timestamp_ns);
            if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
                Eigen::Matrix<double, 15, 15>& Phi_summed = *Phi_summed_ptr;
                Eigen::Matrix<double, 15, 15>& Qd_summed = *Qd_summed_ptr;
                // Get the next state Jacobian and noise Jacobian for this IMU reading
                Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
                Eigen::Matrix<double, 15, 15> Qdi = Eigen::Matrix<double, 15, 15>::Zero();
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
#ifdef DEBUG
    if (Phi_summed_ptr != nullptr && Qd_summed_ptr != nullptr) {
        CHECK(!Phi_summed_ptr->hasNaN());
        CHECK(!Qd_summed_ptr->hasNaN());
        VLOG(10) << "Phi: " << std::endl << *Phi_summed_ptr;
        VLOG(10) << "Q: " << std::endl << *Qd_summed_ptr;
    }
#endif
}

bool ImuPropagator::StaticInitialize(const common::ImuDatas& imu_data,
                                     bool wait_for_jerk,
                                     common::State* first_state_ptr) {
    common::State& first_state = *CHECK_NOTNULL(first_state_ptr);

    // Return if we don't have any measurements
    if(imu_data.empty()) {
        return false;
    }

    // Newest imu timestamp
    double newesttime = common::NanoSecondsToSeconds(imu_data.back().timestamp_ns);

    // First lets collect a window of IMU readings from the newest measurement to the oldest
    common::ImuDatas window_newest, window_secondnew;
    for(const common::ImuData& data : imu_data) {
        if(common::NanoSecondsToSeconds(data.timestamp_ns) > newesttime-1*init_window_length_ &&
          common::NanoSecondsToSeconds(data.timestamp_ns) <= newesttime-0*init_window_length_) {
            window_newest.push_back(data);
        }
        if(common::NanoSecondsToSeconds(data.timestamp_ns) > newesttime-2*init_window_length_ &&
          common::NanoSecondsToSeconds(data.timestamp_ns) <= newesttime-1*init_window_length_) {
            window_secondnew.push_back(data);
        }
    }

    // Return if both of these failed
    if(window_newest.empty() || window_secondnew.empty()) {
        return false;
    }

    // Calculate the sample variance for the newest one
    Eigen::Matrix<double,3,1> a_avg = Eigen::Matrix<double,3,1>::Zero();
    for(const common::ImuData& data : window_newest) {
        a_avg += data.acc;
    }
    a_avg /= (int)window_newest.size();
    double a_var = 0;
    for(const common::ImuData& data : window_newest) {
        a_var += (data.acc-a_avg).dot(data.acc-a_avg);
    }
    a_var = std::sqrt(a_var/((int)window_newest.size()-1));

    // If it is below the threshold and we want to wait till we detect a jerk
    if(a_var < imu_excite_threshold_ && wait_for_jerk) {
        LOG(WARNING) << "No IMU excitation, below threshold "
                       << a_var << " < " << imu_excite_threshold_;
        return false;
    }

    // Sum up our current accelerations and velocities
    Eigen::Vector3d linsum = Eigen::Vector3d::Zero();
    Eigen::Vector3d angsum = Eigen::Vector3d::Zero();
    for(size_t i=0; i<window_secondnew.size(); i++) {
        linsum += window_secondnew.at(i).acc;
        angsum += window_secondnew.at(i).gyro;
    }

    // Calculate the mean of the linear acceleration and angular velocity
    Eigen::Vector3d linavg = Eigen::Vector3d::Zero();
    Eigen::Vector3d angavg = Eigen::Vector3d::Zero();
    linavg = linsum/window_secondnew.size();
    angavg = angsum/window_secondnew.size();

    // Calculate variance of the
    double a_var2 = 0;
    for(const common::ImuData& data : window_secondnew) {
        a_var2 += (data.acc-linavg).dot(data.acc-linavg);
    }
    a_var2 = std::sqrt(a_var2/((int)window_secondnew.size()-1));

    // If it is above the threshold and we are not waiting for a jerk
    // Then we are not stationary (i.e. moving) so we should wait till we are
    if((a_var > imu_excite_threshold_ || a_var2 > imu_excite_threshold_) && !wait_for_jerk) {
        LOG(WARNING) << "Too much IMU excitation, above threshold "
                       << a_var << "," << a_var2 << " > " << imu_excite_threshold_;
        return false;
    }

    // Get z axis, which alines with -g (z_in_G=0,0,1)
    Eigen::Vector3d z_axis = linavg/linavg.norm();

    // Create an x_axis
    Eigen::Vector3d e_1(1,0,0);

    // Make x_axis perpendicular to z
    Eigen::Vector3d x_axis = e_1-z_axis*z_axis.transpose()*e_1;
    x_axis= x_axis/x_axis.norm();

    // Get z from the cross product of these two
    Eigen::Matrix<double,3,1> y_axis = common::skew_x(z_axis)*x_axis;

    // From these axes get rotation
    Eigen::Matrix<double,3,3> Ro;
    Ro.block(0,0,3,1) = x_axis;
    Ro.block(0,1,3,1) = y_axis;
    Ro.block(0,2,3,1) = z_axis;

    // Create our state variables
    Eigen::Quaterniond q_GtoI(Ro);
    Eigen::Matrix3d R_GtoI = q_GtoI.toRotationMatrix();

    Eigen::Matrix<double,3,1> bg = angavg;
    Eigen::Matrix<double,3,1> ba = linavg - R_GtoI * gravity_;
    // NOTE(chien): In our system, IMU measurements has been rotated to the odom frame.
    // So T_OtoG equal T_ItoG.
    const Eigen::Quaterniond q_OtoG = Eigen::Quaterniond(R_GtoI.transpose());
    first_state.T_OtoG.update(q_OtoG, Eigen::Vector3d::Zero());
    first_state.velocity = Eigen::Vector3d::Zero();
    first_state.ba = ba;
    first_state.bg = bg;
    LOG(INFO) << "Init IMU state velocity: " << first_state.velocity.transpose();
    LOG(INFO) << "Init IMU state bg: " << first_state.bg.transpose();
    LOG(INFO) << "Init IMU state ba: " << first_state.ba.transpose();

    // Done!!!
    return true;
}

bool ImuPropagator::StaticInitialize(common::KeyFrame* keyframe_ptr,
                                     bool wait_for_jerk) {
    common::KeyFrame& keyframe = *CHECK_NOTNULL(keyframe_ptr);
    common::State& first_state = keyframe.state;
    const common::ImuDatas& imu_data = keyframe.sensor_meas.imu_datas;

    return StaticInitialize(imu_data, wait_for_jerk, &first_state);
}

bool ImuPropagator::DynamicInitialize(const std::vector<common::OdomDatas>& odom_datas,
                                      const std::vector<common::ImuDatas>& imu_datas,
                                      common::State* first_state_ptr) {
    common::State& first_state = *CHECK_NOTNULL(first_state_ptr);

    CHECK_EQ(odom_datas.size(), imu_datas.size());
    const int th_id = 3; // quaternion
    const int v_id = 6;  // velocity
    const int bg_id = 9;  // bg
    const int ba_id = 12;  // ba

    const Eigen::Vector3d g(0.0, 0.0, 9.7940);
    vins_core::OdomPropagator odom_propagator;
    Eigen::Matrix<double, 3, 3> phi_r_bg = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Vector3d residual_bg = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 3> phi_v_ba = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Vector3d residual_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_k;
    for (size_t i = 0u; i < imu_datas.size(); ++i) {
        common::State state_odom;
        odom_propagator.Propagate(odom_datas[i], &state_odom);

        common::State state_imu;
        Eigen::Matrix<double, 15, 15> Phi;
        Eigen::Matrix<double, 15, 15> Q;
        Propagate(imu_datas[i], &state_imu, &Phi, &Q);

        const Eigen::Quaterniond& q_odom = state_odom.T_OtoG.getEigenQuaternion();
        const Eigen::Quaterniond& q_imu = state_imu.T_OtoG.getEigenQuaternion();
        const Eigen::Matrix<double, 3, 3> phi_r_bg_i = Phi.block<3, 3>(th_id, bg_id);
        const Eigen::Vector3d residual_bg_i = 2.0 * (q_imu.inverse() * q_odom).vec();

        phi_r_bg += phi_r_bg_i.transpose() * phi_r_bg_i;
        residual_bg += phi_r_bg_i.transpose() * residual_bg_i;

        const Eigen::Vector3d& p_odom = state_odom.T_OtoG.getPosition();
        const double delta_t = common::NanoSecondsToSeconds(
            odom_datas[i].back().timestamp_ns - odom_datas[i].front().timestamp_ns);
        const Eigen::Vector3d v_odom = p_odom / delta_t;
        if (i == 0u) {
            v_k = v_odom;
        } else {
            Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
            for (size_t j = 0u; j < imu_datas[i].size(); ++j) {
                acc_mean += (imu_datas[i][j].acc - g);
            }
            acc_mean /= static_cast<double>(imu_datas[i].size());
            
            const Eigen::Matrix<double, 3, 3> phi_v_ba_j = Eigen::Matrix<double, 3, 3>::Identity();
            const Eigen::Vector3d delta_v = v_odom - v_k;
            const Eigen::Vector3d residual_ba_j = acc_mean - delta_v / delta_t;

            phi_v_ba += phi_v_ba_j.transpose() * phi_v_ba_j;
            residual_ba += phi_r_bg_i.transpose() * residual_ba_j;

            v_k = v_odom;
        }
    }

    Eigen::Vector3d v_init = v_k;
    Eigen::Vector3d bg_init = phi_r_bg.ldlt().solve(residual_bg);
    Eigen::Vector3d ba_init = phi_v_ba.ldlt().solve(residual_ba);
    LOG(INFO) << "Init IMU state velocity: " << v_init.transpose();
    LOG(INFO) << "Init IMU state bg: " << bg_init.transpose();
    LOG(INFO) << "Init IMU state ba: " << ba_init.transpose();
    first_state.velocity = v_init;
    first_state.bg = bg_init;
    first_state.ba = ba_init;

    return true;
}

bool ImuPropagator::DynamicInitialize(const std::vector<common::OdomDatas>& odom_datas,
                                      const std::vector<common::ImuDatas>& imu_datas,
                                      common::KeyFrame* keyframe_ptr) {
    common::KeyFrame& keyframe = *CHECK_NOTNULL(keyframe_ptr);
    
    return DynamicInitialize(odom_datas, imu_datas, &keyframe.state);
}
}  // namespace imu_propagator

