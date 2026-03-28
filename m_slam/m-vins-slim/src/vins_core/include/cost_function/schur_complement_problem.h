#ifndef SCHUR_COMPLEMENT_PROBLEM_H_
#define SCHUR_COMPLEMENT_PROBLEM_H_

#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "data_common/constants.h"

namespace schur {

inline int LocalSize(const int size) {
    return size == common::kGlobalPoseSize ? common::kLocalPoseSize : size;
}


struct AddressedIdx {
    int64_t address;
    int data;

    AddressedIdx(
        const int64_t add,
        const int dat) {
        address = add;
        data = dat;
    }

    bool operator==(const AddressedIdx& a) {
        return address == a.address &&
            data == a.data;
    }
};

using AddressedSize = AddressedIdx;

using DynamicMatrixRowMajor =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct ResidualBlockInfo {
    ResidualBlockInfo(
        ceres::CostFunction* _cost_function,
        ceres::LossFunction* _loss_function,
        std::vector<double*> _parameter_blocks)
            : cost_function(_cost_function),
              loss_function(_loss_function),
              parameter_blocks(_parameter_blocks) {
        Init();
    }

    void Init();

    void Evaluate();

    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;

    double** raw_jacobians;
    std::vector<DynamicMatrixRowMajor,
        Eigen::aligned_allocator<DynamicMatrixRowMajor>> jacobians;
    Eigen::VectorXd residuals;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo*> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<AddressedSize> parameter_block_size;
    std::vector<AddressedIdx> parameter_block_idx;
    std::unordered_map<int64_t, size_t> map_address_parameter_block_size;
    std::unordered_map<int64_t, size_t> map_address_parameter_block_idx;

    //! Helper function to build Hessian matrix.
    //!
    //! \param sub_factor the sub-factor to build Hessian on.
    //! \param block_idx the index of parameter block in the list of
    //!        sub-factor's parameter block address.
    //! \param hessian_idx_ptr the starting index of the block in Hessian.
    //! \param block_size_ptr the size of the parameter block.
    //! \param jacobian_ptr pointer to the queried Jacobian.
    template <typename JacobianMat>
    void GetJacobian(
        const ResidualBlockInfo& sub_factor,
        const size_t block_idx,
        int* hessian_idx_ptr,
        int* block_size_ptr,
        JacobianMat* jacobian_ptr);

    //! Helper function to accumulate Hessian matrix.
    //!
    //! \param sub_factor the sub-factor to build Hessian on.
    //! \param jacobian_i_ptr pointer to the queried Jacobian for i-th block.
    //! \param jacobian_j_ptr pointer to the queried Jacobian for j-th block.
    template <typename JacobianMat>
    void AddHessianBlock(
        const ResidualBlockInfo& sub_factor,
        JacobianMat* jacobian_i_ptr,
        JacobianMat* jacobian_j_ptr);

    void BuildHessianMatrix();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class Problem {
public:
    Problem() = default;
    ~Problem();
    void AddResidualBlock(
        ceres::CostFunction* cost_function,
        ceres::LossFunction* loss_function,
        double* parameter_0,
        double* parameter_1 = nullptr,
        double* parameter_2 = nullptr,
        double* parameter_3 = nullptr,
        double* parameter_4 = nullptr,
        double* parameter_5 = nullptr,
        double* parameter_6 = nullptr,
        double* parameter_7 = nullptr,
        double* parameter_8 = nullptr,
        double* parameter_9 = nullptr);
    void AddResidualBlock(
        ceres::CostFunction* cost_function,
        ceres::LossFunction* loss_function,
        const std::vector<double*>& parameters);
    void AddParameterBlock(
        double* data,
        size_t data_size,
        ceres::LocalParameterization*  param);
    void SetParameterBlockConstant(
        double* parameter_block);
    void SetParameterUpperBound(
        double* parameter_block, int index, double upper_bound);
    void SetParameterLowerBound(
         double* parameter_block, int index, double lower_bound);
    void AddMarginalizeAddress(
        const std::vector<double*>& parameter_blocks);
    void CopyBlockData();
    void Evaluate();
    void Marginalize(const bool log_cov);
    std::vector<double*> GetParameterBlocks(
         const std::unordered_map<int64_t, double*>& addr_shift);
    Eigen::MatrixXd GetCov(const size_t length,
                           const int64_t addr);
    void PrintCov(const size_t length,
                  const uint64_t timestamp_ns,
                  const int64_t addr);

    std::vector<int> keep_block_size_;
    std::vector<int> keep_block_idx_;
    std::vector<double*> keep_block_data_;

    Eigen::MatrixXd linearized_jacobians_;
    Eigen::VectorXd linearized_residuals_;
private:
     void AddResidualBlockInfo(ResidualBlockInfo* residual_block_info);
     void Build(Eigen::MatrixXd* A_ptr,
                Eigen::VectorXd* b_ptr,
                Eigen::MatrixXd* S_ptr,
                Eigen::VectorXd* s_ptr);
     int HessianArrange();
     void BuildHessian(const int total_size,
                       Eigen::MatrixXd* A_ptr,
                       Eigen::VectorXd* b_ptr);
     void BuildSchurComplement(const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& b,
                                 Eigen::MatrixXd* S_ptr,
                                 Eigen::VectorXd* s_ptr);
     void SolveSchurComplement(const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& b,
                                 const int marg_idx,
                                 const int marg_size,
                                 const int keep_idx,
                                 const int keep_size,
                                 Eigen::MatrixXd* Ab_ptr,
                                 Eigen::MatrixXd* S_ptr,
                                 Eigen::VectorXd* s_ptr);

     std::vector<ResidualBlockInfo*> blocks_;

     Eigen::MatrixXd cov_;

     int keep_start_idx_;
     int marg_start_idx_;
     int keep_size_;
     int marg_size_;

     std::vector<AddressedSize> parameter_block_size_;
     std::vector<AddressedIdx> parameter_block_idx_;
     std::unordered_map<int64_t, double*> parameter_block_data_;

     std::unordered_map<int64_t, size_t> map_address_parameter_to_marginalize_;
     std::vector<int64_t> parameter_to_marginalize_address_;

     std::unordered_map<int64_t, size_t> map_address_parameter_set_constant_;
     std::vector<int64_t> parameter_set_constant_address_;

     std::unordered_map<int64_t, size_t> map_address_parameter_block_size_;
     std::unordered_map<int64_t, size_t> map_address_parameter_block_idx_;


};
}

#endif
