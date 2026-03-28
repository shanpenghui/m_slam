#ifndef MVINS_COST_FUNCTION_RELATIVE_POSE_COST_H_
#define MVINS_COST_FUNCTION_RELATIVE_POSE_COST_H_

#include <ceres/ceres.h>

#include "data_common/constants.h"

namespace vins_core {
class RelativePoseCost : public ceres::CostFunction {
public:
    RelativePoseCost(const Eigen::Quaterniond& delta_q,
                     const Eigen::Vector3d& delta_p,
                     const Eigen::Matrix<double, 6, 6>& sqrt_info);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    const Eigen::Quaterniond delta_q_measured_;
    const Eigen::Vector3d delta_p_measured_;
    const Eigen::Matrix<double, 6, 6> sqrt_info_;
};

class PosePriorCost : public ceres::CostFunction {
public:
    PosePriorCost(const Eigen::Quaterniond& q,
                  const Eigen::Vector3d& p,
                  const Eigen::Matrix<double, 6, 6>& sqrt_info);
    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    const Eigen::Quaterniond q_measured_;
    const Eigen::Vector3d p_measured_;
    const Eigen::Matrix<double, 6, 6> sqrt_info_;
};
}
#endif
