#include "cost_function/camera_reprojection_cost.h"

#include "data_common/constants.h"
#include "math_common/math.h"

namespace vins_core {
CameraReprojectionCost::CameraReprojectionCost(
        const aslam::NCamera::Ptr& cameras,
        const Eigen::Vector2d& keypoint_anchor,
        const Eigen::Vector2d& keypoint_current,
        const Eigen::Vector2d& velocity_anchor,
        const Eigen::Vector2d& velocity_current,
        const double use_depth,
        const int cam_idx_anchor,
        const int cam_idx_currnet,
        const double depth_current,
        const double td_anchor,
        const double td_current,
        const double sigma_pixel,
        const double sigma_depth)
    : cameras_(cameras),
      meas_anchor_pixel_(keypoint_anchor),
      meas_current_pixel_(keypoint_current),
      velocity_anchor_pixel_(velocity_anchor),
      velocity_current_pixel_(velocity_current),
      use_depth_(use_depth),
      cam_idx_anchor_(cam_idx_anchor),
      cam_idx_current_(cam_idx_currnet),
      meas_depth_current_(depth_current),
      meas_td_anchor_(td_anchor),
      meas_td_current_(td_current) {
    // Parameters:
    // 1) Landmark inverse depth in anchor camera frame.
    // 2) Anchor pose in global frame.
    // 3) current pose in global frame.
    // 4) time drift between camera and odom/IMU.
    // 5) 6) camera extrinsics.
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(1);
    if (cam_idx_anchor_ == cam_idx_current_) {
        mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    } else {
        const auto type_anchor = cameras_->getCamera(cam_idx_anchor_).getType();
        const auto type_current = cameras_->getCamera(cam_idx_current_).getType();
        if (type_anchor != type_current) {
            LOG(FATAL) << "Can not perform re-projection between different camera type.";
        }
        mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
        mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    }

    if (use_depth_) {
        set_num_residuals(3);
        sqrt_info_ = Eigen::Matrix3d::Zero();
        sqrt_info_.topLeftCorner(2, 2) = Eigen::Matrix2d::Identity() / sigma_pixel;
        sqrt_info_(2, 2) = 1.0 / sigma_depth;
    } else {
        set_num_residuals(2);
        sqrt_info_.resize(2, 2);
        sqrt_info_ = Eigen::Matrix2d::Identity() / sigma_pixel;
    }
}

bool CameraReprojectionCost::Evaluate(double const * const *parameters, double *residuals,
                                      double **jacobians) const {
    const double inv_depth_anchor(parameters[0][0]);

    Eigen::Map<const Eigen::Vector3d> p_OinG_anchor(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_anchor(parameters[1] + 3);
    const Eigen::Matrix3d R_OtoG_anchor = q_OtoG_anchor.toRotationMatrix();

    Eigen::Map<const Eigen::Vector3d> p_OinG_current(parameters[2]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_current(parameters[2] + 3);
    const Eigen::Matrix3d R_OtoG_current = q_OtoG_current.toRotationMatrix();

    const double td(parameters[3][0]);

    Eigen::Map<const Eigen::Vector3d> p_OinC_anchor(parameters[4]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoC_anchor(parameters[4] + 3);
    const aslam::Transformation T_OtoC_anchor(q_OtoC_anchor, p_OinC_anchor);
    const int ex_para_idx_current = (cam_idx_anchor_ == cam_idx_current_) ?
        4 : 5;
    Eigen::Map<const Eigen::Vector3d> p_OinC_current(parameters[ex_para_idx_current]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoC_current(parameters[ex_para_idx_current] + 3);
    const Eigen::Matrix3d R_OtoC_current = q_OtoC_current.toRotationMatrix();
    const aslam::Transformation T_OtoC_current(q_OtoC_current, p_OinC_current);

    const aslam::Transformation T_CtoO_anchor = T_OtoC_anchor.inverse();
    const Eigen::Quaterniond& q_CtoO_anchor = T_CtoO_anchor.getEigenQuaternion();
    const Eigen::Matrix3d& R_CtoO_anchor = T_CtoO_anchor.getRotationMatrix();
    const Eigen::Vector3d& p_CinO_anchor = T_CtoO_anchor.getPosition();

    const Eigen::Vector2d meas_anchor_pixel_td = meas_anchor_pixel_ - (td - meas_td_anchor_) * velocity_anchor_pixel_;
    const Eigen::Vector2d meas_current_pixel_td = meas_current_pixel_ - (td - meas_td_current_) * velocity_current_pixel_;
    const aslam::Camera& camera_anchor = cameras_->getCamera(cam_idx_anchor_);
    Eigen::Vector3d bearing_3d_anchor;
    camera_anchor.backProject3(meas_anchor_pixel_td, &bearing_3d_anchor);
    const double z = bearing_3d_anchor(2);
    bearing_3d_anchor << bearing_3d_anchor(0) / z, bearing_3d_anchor(1) / z, 1.0;

    const Eigen::Vector3d p_LinC_anchor = bearing_3d_anchor / inv_depth_anchor;
    const Eigen::Vector3d p_LinO_anchor = q_CtoO_anchor * p_LinC_anchor + p_CinO_anchor;
    const Eigen::Vector3d p_LinG = q_OtoG_anchor * p_LinO_anchor + p_OinG_anchor;
    const Eigen::Vector3d p_LinO_current = q_OtoG_current.conjugate() * (p_LinG - p_OinG_current);
    const Eigen::Vector3d p_LinC_current = q_OtoC_current * p_LinO_current + p_OinC_current;

    Eigen::Vector2d predicted_reprojection_pixel;
    Eigen::Matrix<double, 2, 3> jaco_ins = Eigen::Matrix<double, 2, 3>::Zero();
    const aslam::Camera& camera_current = cameras_->getCamera(cam_idx_current_);
    camera_current.project3(p_LinC_current, &predicted_reprojection_pixel,
                    (jacobians) ? &jaco_ins : nullptr);

    const Eigen::Vector3d e(0., 0., 1.);
    if (use_depth_) {
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual.head<2>() = predicted_reprojection_pixel - meas_current_pixel_td;
        residual(2) = e.transpose() * p_LinC_current - meas_depth_current_;
        residual = sqrt_info_ * residual;
    } else {
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = sqrt_info_ * (predicted_reprojection_pixel - meas_current_pixel_td);
    }

    if (jacobians) {
        Eigen::MatrixXd dr_dh;
        if (use_depth_) {
            dr_dh.resize(3, 3);
            dr_dh.topLeftCorner(2, 3) = jaco_ins;
            dr_dh.bottomLeftCorner(1, 3) = e.transpose();
        } else {
            dr_dh.resize(2, 2);
            dr_dh = jaco_ins;
        }

        const Eigen::Matrix3d R_GtoC_current = R_OtoC_current * R_OtoG_current.transpose();
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dh_dp_anchor =
                R_GtoC_current * R_OtoG_anchor * R_CtoO_anchor;
        if (jacobians[0]) {
            Eigen::Matrix<double, 3, 1> dp_did_anchor =
                    -1.0 / (inv_depth_anchor * inv_depth_anchor) * p_LinC_anchor;
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, 1>> dr_did_anchor(
                            jacobians[0]);
                dr_did_anchor.setZero();

                dr_did_anchor = sqrt_info_ * dr_dh * dh_dp_anchor * dp_did_anchor;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, 1>> dr_did_anchor(
                        jacobians[0]);
                dr_did_anchor.setZero();

                dr_did_anchor = sqrt_info_ * dr_dh * dh_dp_anchor * dp_did_anchor;
            }
        }
        if (jacobians[1]) {
            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_anchor;
            dh_dT_anchor.leftCols<3>() =
                    R_GtoC_current;
            dh_dT_anchor.rightCols<3>() =
                    -R_GtoC_current * R_OtoG_anchor * common::skew_x(p_LinO_anchor);
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_anchor(
                            jacobians[1]);
                dr_dT_anchor.setZero();

                dr_dT_anchor.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_anchor;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_anchor(
                        jacobians[1]);
                dr_dT_anchor.setZero();

                dr_dT_anchor.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_anchor;
            }
        }
        if (jacobians[2]) {
            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_current;
            dh_dT_current.leftCols<3>() =
                   -R_GtoC_current;
            dh_dT_current.rightCols<3>() =
                   R_OtoC_current * common::skew_x(p_LinO_current);
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_current(
                            jacobians[2]);
                dr_dT_current.setZero();

                dr_dT_current.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_current;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_current(
                        jacobians[2]);
                dr_dT_current.setZero();

                dr_dT_current.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_current;
            }
        }
        if (jacobians[3]) {
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, 1>> dr_dtd(
                            jacobians[3]);
                dr_dtd.setZero();

                Eigen::Vector3d velocity3_anchor;
                velocity3_anchor << velocity_anchor_pixel_, 0.0;
                Eigen::Vector3d velocity3_current;
                velocity3_current << velocity_current_pixel_, 0.0;
                dr_dtd = sqrt_info_ * dr_dh * dh_dp_anchor * velocity3_anchor / (-inv_depth_anchor) +
                    sqrt_info_ * velocity3_current;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, 1>> dr_dtd(
                        jacobians[3]);
                dr_dtd.setZero();

                Eigen::Vector3d velocity3_anchor;
                velocity3_anchor << velocity_anchor_pixel_, 0.0;
                dr_dtd = sqrt_info_ * dr_dh * dh_dp_anchor * velocity3_anchor / (-inv_depth_anchor) +
                    sqrt_info_ * velocity_current_pixel_;
            }
        }
        if (jacobians[4]) {
            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dEx_anchor;
            dh_dEx_anchor.leftCols<3>() =
                    -dh_dp_anchor;
            dh_dEx_anchor.rightCols<3>() =
                    R_GtoC_current * R_OtoG_anchor * common::skew_x(R_CtoO_anchor * p_LinC_anchor);
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dEx_anchor(
                            jacobians[4]);
                dr_dEx_anchor.setZero();

                dr_dEx_anchor.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dEx_anchor;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dEx_anchor(
                        jacobians[4]);
                dr_dEx_anchor.setZero();

                dr_dEx_anchor.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dEx_anchor;
            }
        }
        if (cam_idx_anchor_ != cam_idx_current_ && jacobians[5]) {
            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dEx_current;
            dh_dEx_current.leftCols<3>() =
                    Eigen::Matrix3d::Identity();
            dh_dEx_current.rightCols<3>() =
                    -common::skew_x(p_LinO_current);
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dEx_current(
                            jacobians[5]);
                dr_dEx_current.setZero();

                dr_dEx_current.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dEx_current;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dEx_current(
                        jacobians[5]);
                dr_dEx_current.setZero();

                dr_dEx_current.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dEx_current;
            }
        }
    }

    return true;
}

CameraReprojectionFromMapCost::CameraReprojectionFromMapCost(
        const aslam::NCamera::Ptr& cameras,
        const Eigen::Vector3d& meas_p_LinM,
        const Eigen::Vector2d& meas_pixel,
        const int cam_idx,
        const bool use_depth,
        const double meas_depth,
        const double sigma_pixel,
        const double sigma_depth)
    : cameras_(cameras),
      meas_p_LinM_(meas_p_LinM),
      meas_pixel_(meas_pixel),
      cam_idx_(cam_idx),
      use_depth_(use_depth),
      meas_depth_(meas_depth) {

    // Parameters:
    // 1) Transfromation from global to map.
    // 2) Current pose in global frame.
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);

    if (use_depth_) {
        set_num_residuals(3);
        sqrt_info_ = Eigen::Matrix3d::Zero();
        sqrt_info_.topLeftCorner(2, 2) = Eigen::Matrix2d::Identity() / sigma_pixel;
        sqrt_info_(2, 2) = 1.0 / sigma_depth;
    } else {
        set_num_residuals(2);
        sqrt_info_.resize(2, 2);
        sqrt_info_ = Eigen::Matrix2d::Identity() / sigma_pixel;
    }
}

bool CameraReprojectionFromMapCost::Evaluate(double const * const *parameters, double *residuals,
                                      double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_GinM(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_GtoM(parameters[0] + 3);
    const Eigen::Matrix3d R_GtoM = q_GtoM.toRotationMatrix();
    Eigen::Map<const Eigen::Vector3d> p_OinG(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG(parameters[1] + 3);
    const Eigen::Matrix3d R_OtoG = q_OtoG.toRotationMatrix();

    const aslam::Transformation T_OtoC = cameras_->get_T_BtoC(cam_idx_);
    const Eigen::Matrix3d R_OtoC = T_OtoC.getRotationMatrix();
    const Eigen::Vector3d p_OinC = T_OtoC.getPosition();

    Eigen::Vector3d p_LinG = R_GtoM.transpose() * (meas_p_LinM_ - p_GinM);
    Eigen::Vector3d p_LinO = R_OtoG.transpose() * (p_LinG - p_OinG);
    Eigen::Vector3d p_LinC = R_OtoC * p_LinO + p_OinC;

    Eigen::Vector2d predicted_reprojection_pixel;
    Eigen::Matrix<double, 2, 3> jaco_ins = Eigen::Matrix<double, 2, 3>::Zero();
    const aslam::Camera& camera = cameras_->getCamera(cam_idx_);
    camera.project3(p_LinC, &predicted_reprojection_pixel,
                    (jacobians) ? &jaco_ins : nullptr);

    const Eigen::Vector3d e(0., 0., 1.);
    if (use_depth_) {
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual.head<2>() = predicted_reprojection_pixel - meas_pixel_;
        residual(2) = e.transpose() * p_LinC - meas_depth_;
        residual = sqrt_info_ * residual;
    } else {
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = sqrt_info_ * (predicted_reprojection_pixel - meas_pixel_);
    }

    if (jacobians) {
        Eigen::MatrixXd dr_dh;
        if (use_depth_) {
            dr_dh.resize(3, 3);
            dr_dh.topLeftCorner(2, 3) = jaco_ins;
            dr_dh.bottomLeftCorner(1, 3) = e.transpose();
        } else {
            dr_dh.resize(2, 2);
            dr_dh = jaco_ins;
        }

        const Eigen::Matrix3d R_CtoG = R_OtoG * R_OtoC.transpose();
        const Eigen::Matrix3d R_CtoM = R_GtoM * R_CtoG;

        if (jacobians[0]) {
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_GtoM(
                        jacobians[0]);
                dr_dT_GtoM.setZero();

                Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_GtoM;
                dh_dT_GtoM.leftCols<3>() = -R_CtoM.transpose();
                dh_dT_GtoM.rightCols<3>() = R_CtoG.transpose() * common::skew_x(p_LinG);
                dr_dT_GtoM.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_GtoM;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_GtoM(
                        jacobians[0]);
                dr_dT_GtoM.setZero();

                Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_GtoM;
                dh_dT_GtoM.leftCols<3>() = -R_CtoM.transpose();
                dh_dT_GtoM.rightCols<3>() = R_CtoG.transpose() * common::skew_x(p_LinG);
                dr_dT_GtoM.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_GtoM;
            }
        }
        if (jacobians[1]) {
            if (use_depth_) {
                Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_OtoG(
                        jacobians[1]);
                dr_dT_OtoG.setZero();

                Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_OtoG;
                dh_dT_OtoG.leftCols<3>() = -R_CtoG.transpose();
                dh_dT_OtoG.rightCols<3>() = R_OtoC * common::skew_x(p_LinO);
                dr_dT_OtoG.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_OtoG;
            } else {
                Eigen::Map<Eigen::Matrix<double, 2, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_OtoG(
                        jacobians[1]);
                dr_dT_OtoG.setZero();

                Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> dh_dT_OtoG;
                dh_dT_OtoG.leftCols<3>() = -R_CtoG.transpose();
                dh_dT_OtoG.rightCols<3>() = R_OtoC * common::skew_x(p_LinO);
                dr_dT_OtoG.leftCols<common::kLocalPoseSize>() = sqrt_info_ * dr_dh * dh_dT_OtoG;
            }
        }
    }
    return true;
}

BearingEuclideanCost::BearingEuclideanCost(
        const aslam::NCamera::Ptr& cameras,
        const Eigen::Vector2d& meas_bearing,
        const int cam_idx,
        const double sigma_pixel)
    : cameras_(cameras),
      meas_bearing_(meas_bearing),
      cam_idx_(cam_idx) {
    const double avg_img_size =
            0.5 * (cameras->getCamera(cam_idx_).imageHeight() +
            cameras->getCamera(cam_idx_).imageWidth());
    const double sigma_bearing = sigma_pixel / avg_img_size;
    sqrt_info_ = sigma_bearing * Eigen::Matrix2d::Identity();

    mutable_parameter_block_sizes()->push_back(2);
    set_num_residuals(2);
}

bool BearingEuclideanCost::Evaluate(double const * const *parameters, double *residuals,
                                    double **jacobians) const {
    Eigen::Map<const Eigen::Vector2d> bearing_estimated(parameters[0]);

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = sqrt_info_ * (bearing_estimated - meas_bearing_);

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> dr_db(
                    jacobians[0]);
            dr_db.setZero();

            dr_db = sqrt_info_ * Eigen::Matrix2d::Identity();
        }
    }
    return true;
}
}
