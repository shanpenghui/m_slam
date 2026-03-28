#include "cost_function/switch_prior_cost.h"

namespace vins_core {

class SwitchPriorErrorCost {
 public:
  SwitchPriorErrorCost(const double prior, const double sqrt_info)
    : prior_(prior),
      sqrt_info_(sqrt_info) {
  }

  template <typename T>
  bool operator()(const T* const switch_variable, T* residual) const {
    CHECK_GE(*switch_variable, static_cast<T>(0));
    CHECK_LE(*switch_variable, static_cast<T>(1));

    *residual = (static_cast<T>(prior_) - (*switch_variable)) *
                static_cast<T>(sqrt_info_);
    return true;
  }

  static constexpr int residualBlockSize = 1;
  static constexpr int switchVariableBlockSize = 1;

 private:
  double prior_;
  double sqrt_info_;
};

ceres::CostFunction* CreateSwitchPriorCost(
        const double prior,
        const double sqrt_info) {
    return new ceres::AutoDiffCostFunction<
        SwitchPriorErrorCost,
        SwitchPriorErrorCost::residualBlockSize,
        SwitchPriorErrorCost::switchVariableBlockSize>(
        new SwitchPriorErrorCost(prior, sqrt_info));
}

}