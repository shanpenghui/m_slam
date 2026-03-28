#ifndef LOOP_CLOSURE_KD_TREE_INDEX_INTERFACE_H_
#define LOOP_CLOSURE_KD_TREE_INDEX_INTERFACE_H_
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <loop_index/index_interface.h>
#include <loop_index/kd_tree_index.h>

namespace loop_closure {
using kd_tree_index::KDTreeIndex;
class KDTreeIndexInterface : public IndexInterface {
public:
    enum { kTargetDimensionality = 10 };
  typedef KDTreeIndex<kTargetDimensionality> Index;

  explicit KDTreeIndexInterface(const float knn_max_radius) {
      index_.reset(new Index(knn_max_radius));
  }

  template <typename DerivedQuery, typename DerivedIndices,
            typename DerivedDistances>
  inline void GetNNearestNeighbors(
      const Eigen::MatrixBase<DerivedQuery>& query_feature, int num_neighbors,
      const Eigen::MatrixBase<DerivedIndices>& indices_const,
      const Eigen::MatrixBase<DerivedDistances>& distances_const) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(
        DerivedQuery, static_cast<int>(kTargetDimensionality), 1);
    CHECK_EQ(indices_const.cols(), 1);
    CHECK_EQ(distances_const.cols(), 1);

    CHECK_EQ(indices_const.rows(), num_neighbors)
        << "The indices parameter must be pre-allocated to hold all results.";
    CHECK_EQ(distances_const.rows(), num_neighbors)
        << "The distances parameter must be pre-allocated to hold all results.";

    common::DescriptorsMatF32& query_feature_dyn = query_feature;

    std::lock_guard<std::mutex> lock(index_mutex_);
    CHECK_NOTNULL(index_);
    index_->GetNNearestNeighbors(
        query_feature_dyn, num_neighbors, indices_const, distances_const);
  }

  virtual void Clear();

  virtual void AddDescriptors(
           const common::DescriptorsMatF32& descriptors);

  virtual int GetNumDescriptorsInIndex() const;

  virtual void GetNNearestNeighborsForFeatures(
           const common::DescriptorsMatF32& query_features,
           const int num_neighbors,
           Eigen::MatrixXi* indices,
           Eigen::MatrixXf* distances) const;

  virtual void ProjectDescriptors(
           const common::DescriptorsMatUint8& raw_des,
           common::DescriptorsMatF32* projected_des) const;

private:
  std::shared_ptr<Index> index_;
  mutable std::mutex index_mutex_;
};
}  // namespace loop_closure
#endif  // LOOP_CLOSURE_KD_TREE_INDEX_INTERFACE_H_
