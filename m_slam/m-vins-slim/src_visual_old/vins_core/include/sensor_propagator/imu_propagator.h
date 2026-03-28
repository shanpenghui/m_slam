#ifndef MVINS_IMU_PROPAGATOR_H_
#define MVINS_IMU_PROPAGATOR_H_

#include "sensor_propagator/sensor_propagator.h"

namespace vins_core {

class ImuPropagator : public SensorPropagator {
public:
    /**
     * @brief Struct of our imu noise parameters
     */
    struct NoiseManager {

        NoiseManager(const common::SlamConfigPtr& config) {
            sigma_gyro = config->sigma_gyro;
            sigma_acc = config->sigma_acc;
            sigma_bg = config->sigma_bg;
            sigma_ba = config->sigma_ba;
            sigma_gyro_2 = std::pow(sigma_gyro, 2);
            sigma_acc_2 = std::pow(sigma_acc, 2);
            sigma_bg_2 = std::pow(sigma_bg, 2);
            sigma_ba_2 = std::pow(sigma_ba, 2);
        }

        /// Gyroscope white noise (rad/s/sqrt(hz))
        double sigma_gyro;

        /// Gyroscope white noise covariance
        double sigma_gyro_2;

        /// Gyroscope random walk (rad/s^2/sqrt(hz))
        double sigma_bg;

        /// Gyroscope random walk covariance
        double sigma_bg_2;

        /// Accelerometer white noise (m/s^2/sqrt(hz))
        double sigma_acc;

        /// Accelerometer white noise covariance
        double sigma_acc_2;

        /// Accelerometer random walk (m/s^3/sqrt(hz))
        double sigma_ba;

        /// Accelerometer random walk covariance
        double sigma_ba_2;
    };

    ImuPropagator(const NoiseManager& noises);

    bool StaticInitialize(const common::ImuDatas& imu_data,
                          bool wait_for_jerk,
                          common::State* first_state_ptr);

    bool StaticInitialize(common::KeyFrame* keyframe_ptr,
                          bool wait_for_jerk=true);

    bool DynamicInitialize(const std::vector<common::OdomDatas>& odom_datas,
                           const std::vector<common::ImuDatas>& imu_datas,
                           common::State* first_state_ptr);

    bool DynamicInitialize(const std::vector<common::OdomDatas>& odom_datas,
                           const std::vector<common::ImuDatas>& imu_datas,
                           common::KeyFrame* keyframe_ptr);

    void Propagate(const common::ImuDatas& prop_data,
                   common::State* state_ptr,
                   Eigen::Matrix<double, 15, 15>* Phi_summed_ptr = nullptr,
                   Eigen::Matrix<double, 15, 15>* Qd_summed_ptr = nullptr);
protected:
    void PredictMeanDiscrete(const double dt,
                             const Eigen::Vector3d& w_hat1, const Eigen::Vector3d& a_hat1,
                             const Eigen::Vector3d& w_hat2, const Eigen::Vector3d& a_hat2,
                             const common::State& state,
                             Eigen::Quaterniond* new_q_ptr,
                             Eigen::Vector3d* new_v_ptr,
                             Eigen::Vector3d* new_p_ptr);

    void PredictAndCompute(const common::ImuData& data_minus,
                           const common::ImuData& data_plus,
                           common::State* state_ptr,
                           Eigen::Matrix<double, 15, 15>* F_ptr = nullptr,
                           Eigen::Matrix<double, 15, 15>* Qd_ptr = nullptr);
private:
    // Container for the noise values.
    NoiseManager noises_;

    Eigen::Matrix<double,18,18> Qc_ = Eigen::Matrix<double,18,18>::Zero();

    // Gravity vector.
    Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, 9.805);

    // Init time window length (second).
    double init_window_length_ = 0.75;
    // Imu acc threshold for detect a jerk.
    double imu_excite_threshold_ = 0.75;

};
}  // namespace vins_core


#endif  // MVINS_IMU_PROPAGATOR_H_
