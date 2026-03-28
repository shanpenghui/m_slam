#include "loop_index/kd_tree_index_interface.h"

namespace loop_closure {

void KDTreeIndexInterface::Clear() {
  index_->Clear();
}

int KDTreeIndexInterface::GetNumDescriptorsInIndex() const {
  return index_->GetNumDescriptorsInIndex();
}

void KDTreeIndexInterface::AddDescriptors(
        const common::DescriptorsMatF32& descriptors) {
  CHECK_EQ(descriptors.rows(), kTargetDimensionality);
  std::lock_guard<std::mutex> lock(index_mutex_);
  CHECK_NOTNULL(index_);
  index_->AddDescriptors(descriptors);
}

void KDTreeIndexInterface::GetNNearestNeighborsForFeatures(
        const common::DescriptorsMatF32& query_features,
        int num_neighbors,
        Eigen::MatrixXi* indices,
        Eigen::MatrixXf* distances) const {
  CHECK_NOTNULL(indices);
  CHECK_NOTNULL(distances);
  std::lock_guard<std::mutex> lock(index_mutex_);
  CHECK_NOTNULL(index_);
  index_->GetNNearestNeighbors(
      query_features, num_neighbors, indices, distances);
}

void KDTreeIndexInterface::ProjectDescriptors(
         const common::DescriptorsMatUint8& raw_des,
        common::DescriptorsMatF32* projected_des) const {
    LOG(FATAL) << "Kd-tree index mode need not project descriptors,"
                  "because of we use descriptor mat f32 driectly.";
}

}
