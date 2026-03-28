#include "cost_function/schur_complement_problem.h"

#include "time_common/time.h"

namespace schur {

template <typename T>
void EvaluteLoss(const T sq_norm,
                 const T* rho,
                 T* residual_scaling_ptr,
                 T* alpha_sq_norm_ptr,
                 T* sqrt_rho1_ptr) {
    T& residual_scaling = *CHECK_NOTNULL(residual_scaling_ptr);
    T& alpha_sq_norm = *CHECK_NOTNULL(alpha_sq_norm_ptr);
    T& sqrt_rho1 = *CHECK_NOTNULL(sqrt_rho1_ptr);
    sqrt_rho1 = std::sqrt(rho[1]);
    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
        residual_scaling = sqrt_rho1;
        alpha_sq_norm = 0.0;
    } else {
        const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        const double alpha = 1.0 - std::sqrt(D);
        residual_scaling = sqrt_rho1 / (1 - alpha);
        alpha_sq_norm = alpha / sq_norm;
    }
}

template <typename T>
void JacobiansScaling(const T alpha_sq_norm,
                      const T sqrt_rho1,
                      const int num_rows,
                      const int num_cols,
                      T* residual,
                      T* jacobian) {
    if (alpha_sq_norm == 0.0) {
        Eigen::Map<Eigen::VectorXd>(jacobian, num_rows * num_cols) *=
            sqrt_rho1;
        return;
    }
    for (int c = 0; c < num_cols; ++c) {
        double r_transpose_j = 0.0;
        for (int r = 0; r < num_rows; ++r) {
            r_transpose_j += jacobian[r * num_cols + c] * residual[r];
        }

        for (int r = 0; r < num_rows; ++r) {
            jacobian[r * num_cols + c] = sqrt_rho1 *
                (jacobian[r * num_cols + c] -
                    alpha_sq_norm * residual[r] * r_transpose_j);
        }
    }
}

template<class T>
bool FindInMap(
    const T& data,
    const std::unordered_map<T, size_t>& map) {
    bool found = false;
    const auto search = map.find(data);
    if (search != map.end()) {
        found = true;
    }
    return found;
}

int HaveSameAddressInMap(
    const int64_t address,
    const std::unordered_map<int64_t, size_t>& map_adress_to_idx) {
    int index = -1;
    const auto search = map_adress_to_idx.find(address);
    if (search != map_adress_to_idx.end()) {
        index = search->second;
    }
    return index;
}

int GetAddressedData(
    const int64_t address,
    const std::unordered_map<int64_t, size_t>& map_address_to_idx,
    const std::vector<AddressedIdx>& addressed_int_vec) {
    const int index = HaveSameAddressInMap(address, map_address_to_idx);
    CHECK_GE(index, 0) << "Index out of range, please check.";
    return addressed_int_vec[index].data;
}

void ResidualBlockInfo::Init() {
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());
}

void ResidualBlockInfo::Evaluate() {
    const std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    const int num_residuals = cost_function->num_residuals();

    for (size_t i = 0u; i < block_sizes.size(); ++i) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(),
                            raw_jacobians);

    if (loss_function) {
        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double residual_scaling, alpha_sq_norm, sqrt_rho1;
        EvaluteLoss(sq_norm, rho,
                    &residual_scaling,
                    &alpha_sq_norm,
                    &sqrt_rho1);

        for (size_t i = 0u; i < parameter_blocks.size(); ++i) {
            JacobiansScaling(alpha_sq_norm,
                             sqrt_rho1,
                             num_residuals,
                             block_sizes[i],
                             residuals.data(),
                             jacobians[i].data());
        }

        residuals *= residual_scaling;
    }
}

template <typename JacobianMat>
void ThreadsStruct::GetJacobian(
    const ResidualBlockInfo& sub_factor,
    const size_t block_idx,
    int* hessian_idx_ptr,
    int* block_size_ptr,
    JacobianMat* jacobian_ptr) {
    int& hessian_idx = *CHECK_NOTNULL(hessian_idx_ptr);
    int& block_size = *CHECK_NOTNULL(block_size_ptr);
    JacobianMat& jacobian = *CHECK_NOTNULL(jacobian_ptr);
    const int64_t block_address = reinterpret_cast<int64_t>(
        sub_factor.parameter_blocks[block_idx]);
    // Get the starting index of the parameter block in Hessian.
    hessian_idx = GetAddressedData(block_address,
                                    map_address_parameter_block_idx,
                                    parameter_block_idx);
    // Get the size of the parameter block.
    block_size = LocalSize(GetAddressedData(block_address,
                                             map_address_parameter_block_size,
                                             parameter_block_size));

    CHECK_LT(block_idx, sub_factor.jacobians.size());
    CHECK_LE(block_size, sub_factor.jacobians[block_idx].cols());

    // Get the Jacobian for the parameter block.
    jacobian = sub_factor.jacobians[block_idx].leftCols(block_size);
}

template <typename JacobianMat>
void ThreadsStruct::AddHessianBlock(
    const ResidualBlockInfo& sub_factor,
    JacobianMat* jacobian_i_ptr,
    JacobianMat* jacobian_j_ptr) {
    JacobianMat& jacobian_i = *CHECK_NOTNULL(jacobian_i_ptr);
    JacobianMat& jacobian_j = *CHECK_NOTNULL(jacobian_j_ptr);

    // The starting index and (square) block size in Hessian.
    int hessian_idx_i, hessian_idx_j, block_size_i, block_size_j;

    for (size_t i = 0u; i < sub_factor.parameter_blocks.size(); ++i) {
        // Get Jacobian for i-th block and related parameters.
        GetJacobian<JacobianMat>(sub_factor, i, &hessian_idx_i,
            &block_size_i, &jacobian_i);
        for (size_t j = i; j < sub_factor.parameter_blocks.size(); ++j) {
            // Get Jacobian for j-th block and related parameters.
            GetJacobian<JacobianMat>(sub_factor, j, &hessian_idx_j,
                &block_size_j, &jacobian_j);
            if (hessian_idx_i == hessian_idx_j) {
                // Diagonal block.
                A.block(hessian_idx_i, hessian_idx_j,
                    block_size_i, block_size_j).triangularView<Eigen::Upper>()
                        += jacobian_i.transpose() * jacobian_j;
            } else if (hessian_idx_i < hessian_idx_j) {
                // (idx_i, idx_j) is in upper triangle.
                A.block(hessian_idx_i, hessian_idx_j,
                    block_size_i, block_size_j).noalias() +=
                        jacobian_i.transpose() * jacobian_j;
            } else {
                // (idx_i, idx_j) is in lower triangle.
                A.block(hessian_idx_j, hessian_idx_i,
                    block_size_j, block_size_i).noalias() +=
                        jacobian_j.transpose() * jacobian_i;
            }
        }
        b.segment(hessian_idx_i, block_size_i).noalias() +=
            jacobian_i.transpose() * sub_factor.residuals;
    }
}

using JacobianMatrixDynamic =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
void ThreadsStruct::BuildHessianMatrix() {
    JacobianMatrixDynamic jacobian_i_dynamic, jacobian_j_dynamic;
    for (const auto& it : sub_factors) {
        const auto& sub_factor = *CHECK_NOTNULL(it);
        AddHessianBlock<JacobianMatrixDynamic>(sub_factor,
            &jacobian_i_dynamic, &jacobian_j_dynamic);
    }
}

Problem::~Problem() {
    for (auto it = parameter_block_data_.begin();
         it != parameter_block_data_.end(); ++it) {
        if (it->second != nullptr) {
            delete[] it->second;
        }
    }

    for (size_t i = 0u; i < blocks_.size(); ++i) {
        if (blocks_[i]->raw_jacobians != nullptr) {
            delete[] blocks_[i]->raw_jacobians;
        }
        if (blocks_[i]->cost_function != nullptr) {
            delete blocks_[i]->cost_function;
        }
        if (blocks_[i] != nullptr) {
            delete blocks_[i];
        }
    }

    parameter_block_data_.clear();
    parameter_block_size_.clear();
    parameter_block_idx_.clear();

    map_address_parameter_to_marginalize_.clear();
    map_address_parameter_set_constant_.clear();
    map_address_parameter_block_size_.clear();
    map_address_parameter_block_idx_.clear();
}

void Problem::AddResidualBlock(
    ceres::CostFunction* cost_function,
    ceres::LossFunction* loss_function,
    double* parameter_0,
    double* parameter_1,
    double* parameter_2,
    double* parameter_3,
    double* parameter_4,
    double* parameter_5,
    double* parameter_6,
    double* parameter_7,
    double* parameter_8,
    double* parameter_9) {
    CHECK(parameter_0 != nullptr) << "We should at least have one block.";

    std::vector<double*> para_vector;
    para_vector.emplace_back(parameter_0);

    if (parameter_1 != nullptr) {
        para_vector.emplace_back(parameter_1);
    }
    if (parameter_2 != nullptr) {
        para_vector.emplace_back(parameter_2);
    }
    if (parameter_3 != nullptr) {
        para_vector.emplace_back(parameter_3);
    }
    if (parameter_4 != nullptr) {
        para_vector.emplace_back(parameter_4);
    }
    if (parameter_5 != nullptr) {
        para_vector.emplace_back(parameter_5);
    }
    if (parameter_6 != nullptr) {
        para_vector.emplace_back(parameter_6);
    }
    if (parameter_7 != nullptr) {
        para_vector.emplace_back(parameter_7);
    }
    if (parameter_8 != nullptr) {
        para_vector.emplace_back(parameter_8);
    }
    if (parameter_9 != nullptr) {
        para_vector.emplace_back(parameter_9);
    }

    AddResidualBlock(cost_function,
                     loss_function,
                     para_vector);
}

void Problem::AddResidualBlock(
    ceres::CostFunction* cost_function,
    ceres::LossFunction* loss_function,
    const std::vector<double*>& parameters) {
    CHECK(cost_function != nullptr) << "Cost function must be not nullptr";

    ResidualBlockInfo* residual_block_info =
            new ResidualBlockInfo(cost_function, loss_function, parameters);
    blocks_.push_back(residual_block_info);

    AddResidualBlockInfo(blocks_.back());
}

void Problem::AddParameterBlock(
        double* data,
        size_t data_size,
        ceres::LocalParameterization*  param) {
    delete param;
}

void Problem::AddResidualBlockInfo(
        ResidualBlockInfo* residual_block_info) {
    CHECK_NOTNULL(residual_block_info);

    std::vector<double*>& parameter_blocks =
        residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes =
        residual_block_info->cost_function->parameter_block_sizes();

    for (size_t i = 0u; i < parameter_blocks.size(); ++i) {
        const int64_t addr = reinterpret_cast<int64_t>(parameter_blocks[i]);
        const int size = parameter_block_sizes[i];
        parameter_block_size_.emplace_back(addr, size);
        map_address_parameter_block_size_.insert(
            {addr, parameter_block_size_.size() - 1u});
    }
}

void Problem::AddMarginalizeAddress(
    const std::vector<double*>& parameter_blocks) {
    for (size_t i = 0u; i < parameter_blocks.size(); ++i) {
        const int64_t addr = reinterpret_cast<int64_t>(parameter_blocks[i]);

        if (!FindInMap(addr, map_address_parameter_to_marginalize_)) {
            parameter_to_marginalize_address_.push_back(addr);
            map_address_parameter_to_marginalize_.insert(
                {addr, parameter_to_marginalize_address_.size() - 1u});
        }
    }
}

void Problem::SetParameterBlockConstant(
    double* parameter_block) {
    CHECK_NOTNULL(parameter_block);
    const int64_t addr = reinterpret_cast<int64_t>(parameter_block);
    if (!FindInMap(addr, map_address_parameter_set_constant_)) {
        parameter_set_constant_address_.push_back(addr);
        map_address_parameter_set_constant_.insert(
            {addr, parameter_set_constant_address_.size() - 1u});
    }
}

void Problem::SetParameterLowerBound(
     double* parameter_block, int index, double lower_bound) {
    // Do nothing.
}

void Problem::SetParameterUpperBound(
     double* parameter_block, int index, double upper_bound) {
    // Do nothing.
}

void Problem::CopyBlockData() {
    for (auto it : blocks_) {
        std::vector<int> block_sizes =
            it->cost_function->parameter_block_sizes();
        for (size_t i = 0u; i < block_sizes.size(); ++i) {
            const int64_t addr =
                reinterpret_cast<int64_t>(it->parameter_blocks[i]);
            const int size = block_sizes[i];
            if (parameter_block_data_.find(addr) == parameter_block_data_.end()) {
                double* data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data_.emplace(addr, data);
            }
        }
    }
}

void Problem::Evaluate() {
    for (auto it : blocks_) {
        it->Evaluate();
    }
}

void Problem::Marginalize(const bool log_cov) {
    Eigen::MatrixXd A, S;
    Eigen::VectorXd b, s;

    Build(&A, &b, &S, &s);

    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt_mat;
    llt_mat.compute(S);

    if (llt_mat.info() == Eigen::ComputationInfo::Success) {
        linearized_jacobians_ = llt_mat.matrixL().transpose();
        linearized_residuals_.noalias() = llt_mat.matrixL().solve(s);
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(
            S.selfadjointView<Eigen::Upper>());

        const Eigen::VectorXd S_sqrt = Eigen::VectorXd(
            (saes.eigenvalues().array() > common::kEpsilon).select(
            saes.eigenvalues().array().sqrt(), 0));
        const Eigen::VectorXd S_inv_sqrt = Eigen::VectorXd(
            (saes.eigenvalues().array() > common::kEpsilon).select(
            saes.eigenvalues().array().inverse().sqrt(), 0));

        linearized_jacobians_ =
            (Eigen::MatrixXd(saes.eigenvectors()).array().rowwise() *
            S_sqrt.transpose().array()).transpose();
        linearized_residuals_.noalias() = S_inv_sqrt.cwiseProduct(
            (saes.eigenvectors().transpose() * s));
    }

    if (log_cov) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(
            S.selfadjointView<Eigen::Upper>());
        cov_.noalias() = saes.eigenvectors() *
            Eigen::VectorXd((saes.eigenvalues().array() > common::kEpsilon).select(
            saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
            saes.eigenvectors().transpose();
    }
}

void Problem::Build(Eigen::MatrixXd* A_ptr,
                    Eigen::VectorXd* b_ptr,
                    Eigen::MatrixXd* S_ptr,
                    Eigen::VectorXd* s_ptr) {
    Eigen::MatrixXd& A = *CHECK_NOTNULL(A_ptr);
    Eigen::VectorXd& b = *CHECK_NOTNULL(b_ptr);
    Eigen::MatrixXd& S = *CHECK_NOTNULL(S_ptr);
    Eigen::VectorXd& s = *CHECK_NOTNULL(s_ptr);

    const int total_size = HessianArrange();

    BuildHessian(total_size, &A, &b);

    BuildSchurComplement(A, b, &S, &s);
}

int Problem::HessianArrange() {
    int total_size = 0;
    // Parameter blocks will be saved in the following order:
    // Constant | To be marginalized | To be kept
    for (const auto& it : parameter_set_constant_address_) {
        const int index = HaveSameAddressInMap(
            it, map_address_parameter_block_size_);
        CHECK_GE(index, 0) << "Can not find address: " << it;
        parameter_block_idx_.emplace_back(it, total_size);
        map_address_parameter_block_idx_.emplace(
            it, parameter_block_idx_.size() - 1u);
        total_size += LocalSize(parameter_block_size_[index].data);
    }
    const int constant_size = total_size;

    marg_start_idx_ = total_size;
    for (const auto& it : parameter_to_marginalize_address_) {
        const int index = HaveSameAddressInMap(
            it, map_address_parameter_block_size_);
        CHECK_GE(index, 0) << "Can not find address: " << it;
        parameter_block_idx_.emplace_back(it, total_size);
        map_address_parameter_block_idx_.emplace(
            it, parameter_block_idx_.size() - 1u);
        total_size += LocalSize(parameter_block_size_[index].data);

    }
    marg_size_ = total_size - constant_size;

    keep_start_idx_ = total_size;
    for (const auto &it : parameter_block_size_) {
        if (HaveSameAddressInMap(
            it.address,
            map_address_parameter_block_idx_) < 0) {
            parameter_block_idx_.emplace_back(it.address, total_size);
            map_address_parameter_block_idx_.insert(
                {it.address, parameter_block_idx_.size() - 1u});
            total_size += LocalSize(it.data);
        }
    }
    keep_size_ = total_size - marg_size_ - constant_size;

    VLOG(2) << "Constant_size " << constant_size
            << ", Marginalized_size " << marg_size_
            << ", Keep_size " << keep_size_
            << " in Hessian.";
    return total_size;
}

void Problem::BuildHessian(const int total_size,
                           Eigen::MatrixXd* A_ptr,
                           Eigen::VectorXd* b_ptr) {
    Eigen::MatrixXd& A = *CHECK_NOTNULL(A_ptr);
    Eigen::VectorXd& b = *CHECK_NOTNULL(b_ptr);
    ThreadsStruct threadsstruct;
    for (auto it : blocks_) {
        threadsstruct.sub_factors.emplace_back(it);
    }

    threadsstruct.A = Eigen::MatrixXd::Zero(total_size, total_size);
    threadsstruct.b = Eigen::VectorXd::Zero(total_size);
    threadsstruct.parameter_block_size = parameter_block_size_;
    threadsstruct.parameter_block_idx = parameter_block_idx_;
    threadsstruct.map_address_parameter_block_size =
        map_address_parameter_block_size_;
    threadsstruct.map_address_parameter_block_idx =
        map_address_parameter_block_idx_;

    threadsstruct.BuildHessianMatrix();

    A = threadsstruct.A.selfadjointView<Eigen::Upper>();
    b = threadsstruct.b;
}

void Problem::BuildSchurComplement(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    Eigen::MatrixXd* S_ptr,
    Eigen::VectorXd* s_ptr) {
    auto& S = *CHECK_NOTNULL(S_ptr);
    auto& s = *CHECK_NOTNULL(s_ptr);
    if (marg_size_ > 0) {
        Eigen::MatrixXd Amr_bmm(marg_size_, keep_size_ + 1);

        Amr_bmm.rightCols(1) = b.segment(marg_start_idx_, marg_size_);
        Amr_bmm.leftCols(keep_size_) =
            A.block(marg_start_idx_, keep_start_idx_, marg_size_, keep_size_);

        SolveSchurComplement(A, b,
            marg_start_idx_, marg_size_, keep_start_idx_, keep_size_,
            &Amr_bmm, S_ptr, s_ptr);
    } else {
        S = A.block(keep_start_idx_, keep_start_idx_, keep_size_, keep_size_);
        s = b.segment(keep_start_idx_, keep_size_);
    }
}

void Problem::SolveSchurComplement(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const int marg_idx,
    const int marg_size,
    const int keep_idx,
    const int keep_size,
    Eigen::MatrixXd* Ab_ptr,
    Eigen::MatrixXd* S_ptr,
    Eigen::VectorXd* s_ptr) {
    auto& Ab = *CHECK_NOTNULL(Ab_ptr);
    auto& S = *CHECK_NOTNULL(S_ptr);
    auto& s = *CHECK_NOTNULL(s_ptr);

    const auto llt = A.block(marg_idx, marg_idx, marg_size, marg_size).
        selfadjointView<Eigen::Upper>().llt();
    if (llt.info() == Eigen::ComputationInfo::Success) {
        llt.solveInPlace(Ab);
    } else {
        A.block(marg_idx, marg_idx, marg_size, marg_size).
            selfadjointView<Eigen::Upper>().ldlt().solveInPlace(Ab);
    }

    S = A.block(keep_idx, keep_idx, keep_size, keep_size).
            triangularView<Eigen::Upper>();
    S.triangularView<Eigen::Upper>() -=
        A.block(marg_idx, keep_idx, marg_size, keep_size).
            transpose() * Ab.leftCols(keep_size);
    s = b.segment(keep_idx, keep_size) -
        A.block(marg_idx, keep_idx, marg_size, keep_size).
            transpose() * Ab.rightCols(1);
#ifdef DEBUG
    CHECK(!S.hasNaN()) << std::endl << S;
    CHECK(!s.hasNaN()) << std::endl << s;
#endif
}

std::vector<double*> Problem::GetParameterBlocks(
    const std::unordered_map<int64_t, double*>& addr_shift) {
    std::vector<double*> keep_block_addr;
    keep_block_size_.clear();
    keep_block_idx_.clear();
    keep_block_data_.clear();

    for (const auto& it : parameter_block_idx_) {
        if (it.data >= keep_start_idx_) {
            keep_block_size_.emplace_back(
                GetAddressedData(it.address,
                                 map_address_parameter_block_size_,
                                 parameter_block_size_));
            keep_block_idx_.emplace_back(
                GetAddressedData(
                    it.address,
                    map_address_parameter_block_idx_,
                    parameter_block_idx_) - keep_start_idx_);

            const auto& data_iter = parameter_block_data_.find(it.address);
            CHECK(data_iter != parameter_block_data_.end());
            keep_block_data_.emplace_back(data_iter->second);

            const auto& shifted_address_iter = addr_shift.find(it.address);
            if (shifted_address_iter != addr_shift.end()) {
                keep_block_addr.emplace_back(
                    shifted_address_iter->second);
            } else {
                keep_block_addr.emplace_back(
                    reinterpret_cast<double*>(it.address));
            }
        }
    }
    return keep_block_addr;
}

Eigen::MatrixXd Problem::GetCov(const size_t length,
                                const int64_t addr) {
    const int index = HaveSameAddressInMap(
        addr,
        map_address_parameter_block_idx_);
    if (index < 0) {
        LOG(FATAL) << "cannot find addr.";
    }

    const int idx = parameter_block_idx_[index].data - keep_start_idx_;
    CHECK_GE(idx, 0);
    Eigen::MatrixXd target_cov_block = cov_.block(idx, idx, length, length);

    return target_cov_block;
}

void Problem::PrintCov(const size_t length,
                       const uint64_t timestamp_ns,
                       const int64_t addr) {
    const Eigen::MatrixXd target_cov_block = GetCov(length, addr);
    const Eigen::VectorXd cov_diagonal_sqrt = target_cov_block.diagonal().cwiseSqrt();
    // Pose cov.
    if (length == 6u) {
        LOG(INFO) << std::setprecision(18)
                  << "PosStd: " << common::NanoSecondsToSeconds(timestamp_ns)
                  << ", " << cov_diagonal_sqrt(0)
                  << ", " << cov_diagonal_sqrt(1)
                  << ", " << cov_diagonal_sqrt(2);
        LOG(INFO) << std::setprecision(18)
                  << "RotStd: " << common::NanoSecondsToSeconds(timestamp_ns)
                  << ", " << cov_diagonal_sqrt(3)
                  << ", " << cov_diagonal_sqrt(4)
                  << ", " << cov_diagonal_sqrt(5);
    } else {
        std::string st = "";
        for (size_t i = 0u; i < length; ++i) {
            st += ", " + std::to_string(cov_diagonal_sqrt(i));
        }
        LOG(INFO) << std::setprecision(18)
                  << "OtherStd: " << common::NanoSecondsToSeconds(timestamp_ns)
                  << st;
    }
}
}
