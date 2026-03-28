#ifndef LOOP_CLOSURE_INDEX_INTERFACE_H_
#define LOOP_CLOSURE_INDEX_INTERFACE_H_
#include <vector>

#include <Eigen/Core>
#include <loop_index/helpers.h>

namespace loop_closure {

class IndexInterface {
public:
    virtual ~IndexInterface() {}

    virtual void Clear() = 0;

    // Add descriptors to the index. Can be done lazily.
    virtual void AddDescriptors(
             const common::DescriptorsMatF32& descriptors) = 0;

    // The number of individual descriptors in the index.
    virtual int GetNumDescriptorsInIndex() const = 0;

    // Return the indices and distances of the num_neighbors closest descriptors
    // for every descriptor from the query_features matrix.
    virtual void GetNNearestNeighborsForFeatures(
             const common::DescriptorsMatF32& query_features,
             const int num_neighbors,
             Eigen::MatrixXi* indices, Eigen::MatrixXf* distances) const = 0;

    // Use the projection matrix specific to the used index to project the
    // binary descriptors to a lower dimensional, real valued space.
    virtual void ProjectDescriptors(
             const common::DescriptorsMatUint8& raw_des,
             common::DescriptorsMatF32* projected_des) const = 0;
};
}  // namespace loop_closure
#endif  // LOOP_CLOSURE_INDEX_INTERFACE_H_
