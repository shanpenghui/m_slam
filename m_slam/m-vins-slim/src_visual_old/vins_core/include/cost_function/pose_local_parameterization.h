#ifndef MVINS_COST_FUNCTION_POSE_LOCAL_PARAMETERIZATION_H_
#define MVINS_COST_FUNCTION_POSE_LOCAL_PARAMETERIZATION_H_

#include <ceres/ceres.h>

#include "data_common/constants.h"

namespace vins_core {

class Pose3DLocalParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const {return common::kGlobalPoseSize;}
    virtual int LocalSize() const {return common::kLocalPoseSize;}
};

class Pose2DLocalParameterization : public ceres::LocalParameterization {
    virtual bool Plus(
        const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const {return common::kGlobalPoseSize;}
    virtual int LocalSize() const {return 3;}
};

}

#endif
