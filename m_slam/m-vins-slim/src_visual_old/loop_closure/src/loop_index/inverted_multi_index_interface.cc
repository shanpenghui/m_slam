#include "loop_index/inverted_multi_index_interface.h"

#include "loop_index/helpers.h"

namespace loop_closure {
void InvertedMultiIndexInterface::Clear() {
    index_->Clear();
}
void InvertedMultiIndexInterface::AddDescriptors(
        const common::DescriptorsMatF32& descriptors) {
    CHECK_EQ(descriptors.rows(), kSubSpaceDimensionality * 2);
    CHECK_NOTNULL(index_);
    index_->AddDescriptors(descriptors);
}
void InvertedMultiIndexInterface::ProjectDescriptors(
        const common::DescriptorsMatUint8 &raw_des,
        common::DescriptorsMatF32 *projected_des) const {
    CHECK_NOTNULL(projected_des);
    internal::ProjectDescriptors(
        raw_des, vocabulary_.projection_matrix_,
        vocabulary_.target_dimensionality_, projected_des);
}

int InvertedMultiIndexInterface::GetNumDescriptorsInIndex() const {
    return index_->GetNumDescriptorsInIndex();
}

void InvertedMultiIndexInterface::GetNNearestNeighborsForFeatures(
         const common::DescriptorsMatF32& query_features,
         const int num_neighbors,
         Eigen::MatrixXi* indices,
         Eigen::MatrixXf* distances) const {
    CHECK_NOTNULL(indices);
    CHECK_NOTNULL(distances);
    GetNNearestNeighborsForFeaturesImpl(
                query_features, num_neighbors, *indices, *distances);
}
}
