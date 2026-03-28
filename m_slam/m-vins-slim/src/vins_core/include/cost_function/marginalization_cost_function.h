#ifndef MAPPING_MARGINALIZATION_COST_FUNCTOR_H_
#define MAPPING_MARGINALIZATION_COST_FUNCTOR_H_

#include <vector>
#include <Eigen/Core>

#include "cost_function/schur_complement_problem.h"

namespace vins_core {
class MarginalizationCost : public ceres::CostFunction {
 public:
    MarginalizationCost(const schur::Problem* const prior_term);
    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    const schur::Problem* const prior_term_;

};
}
#endif
