#ifndef MVINS_COST_FUNCTION_POSE_IN_PLANE_H_
#define MVINS_COST_FUNCTION_POSE_IN_PLANE_H_

#include <ceres/ceres.h>

namespace vins_core {
class PoseInPlaneCost : public ceres::CostFunction {
public:
    PoseInPlaneCost(const Eigen::Matrix2d& sqrt_info_R,
                    const double sqrt_info_t);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    Eigen::Matrix3d sqrt_info_;
};
}
#endif