#ifndef LOOP_CLOSURE_INVERTED_MULTI_INDEX_HELPERS_H_
#define LOOP_CLOSURE_INVERTED_MULTI_INDEX_HELPERS_H_
#include <vector>

#include <Eigen/Core>
#include <aslam/common/timer.h>

#include "data_common/visual_structures.h"
#include "descriptor_projection/descriptor_projection.h"

namespace loop_closure {
namespace internal {

// Zero copy passing of Eigen-Block Expressions.
// http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>
inline Eigen::MatrixBase<Derived>& CastConstEigenMatrixToNonConst(
    const Eigen::MatrixBase<Derived>& value) {
  return const_cast<Eigen::MatrixBase<Derived>&>(value);
}

template <typename IdType>
inline size_t GetNumberOfMatches(
    const loop_closure::IdToFrameKeyPointMatches<IdType>& id_to_matches) {
    size_t num_matches = 0u;
    for (const auto& id_and_matches : id_to_matches) {
        num_matches += id_and_matches.second.size();
    }
    return num_matches;
}

inline void ProjectDescriptors(
    const common::DescriptorsMatUint8& descriptors,
    const Eigen::MatrixXf& projection_matrix, int target_dimensionality,
    common::DescriptorsMatF32* projected_descriptors) {
  ProjectDescriptorBlock(descriptors, projection_matrix,
                        target_dimensionality, projected_descriptors);
}

}
}
#endif
