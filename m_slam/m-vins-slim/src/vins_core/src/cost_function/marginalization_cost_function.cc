#include "cost_function/marginalization_cost_function.h"

namespace vins_core {
MarginalizationCost::MarginalizationCost(
        const schur::Problem* const prior_term)
        : prior_term_(prior_term)  {
    CHECK_NOTNULL(prior_term_);

    int total_size = 0;
    for (size_t i = 0u; i < prior_term_->keep_block_size_.size(); ++i) {
        const int block_size = prior_term_->keep_block_size_[i];

        mutable_parameter_block_sizes()->push_back(block_size);

        total_size += schur::LocalSize(block_size);
    }

    set_num_residuals(total_size);
}

bool MarginalizationCost::Evaluate(double const * const *parameters,
                                          double *residuals,
                                          double **jacobians) const {
    const int residual_size = num_residuals();

    // Evalute residuals.
    Eigen::VectorXd dx(residual_size);
    for (size_t i = 0u; i < prior_term_->keep_block_size_.size(); ++i) {
        const int block_size = prior_term_->keep_block_size_[i];
        const int start_idx = prior_term_->keep_block_idx_[i];

        const Eigen::VectorXd x =
                Eigen::Map<const Eigen::VectorXd>(parameters[i], block_size);
        const Eigen::VectorXd x_prior = Eigen::Map<const Eigen::VectorXd>(
                prior_term_->keep_block_data_[i], block_size);

        if (block_size == common::kGlobalPoseSize) {
            Eigen::Map<const Eigen::Quaterniond> q(parameters[i] + 3);
            Eigen::Map<const Eigen::Quaterniond> q_prior(
                        prior_term_->keep_block_data_[i] + 3);
            dx.segment<3>(start_idx) = x.head<3>() - x_prior.head<3>();
            dx.segment<3>(start_idx + 3) = 2.0 * (q_prior.inverse() * q).vec();
        } else {
            dx.segment(start_idx, block_size) = x - x_prior;
        }
    }
    Eigen::Map<Eigen::VectorXd> residual(residuals, residual_size);
    residual = prior_term_->linearized_residuals_ + prior_term_->linearized_jacobians_ * dx;

    if (jacobians) {
        for (size_t i = 0u; i < prior_term_->keep_block_size_.size(); ++i) {
            if (jacobians[i]) {
                const int global_size = prior_term_->keep_block_size_[i];
                const int local_size = schur::LocalSize(global_size);
                const int start_idx = prior_term_->keep_block_idx_[i];              

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        jacobian(jacobians[i], residual_size, global_size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = prior_term_->linearized_jacobians_.middleCols(start_idx, local_size);
            }
        }
    }
    return true;
}
}
