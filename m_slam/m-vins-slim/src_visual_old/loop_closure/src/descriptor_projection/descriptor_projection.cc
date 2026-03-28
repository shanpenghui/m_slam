#include "descriptor_projection/descriptor_projection.h"

#include <aslam/common/feature-descriptor-ref.h>
#include <Eigen/Core>

#include "time_common/time.h"

namespace loop_closure {

void ProjectDescriptorBlock(
        const common::DescriptorsMatUint8& raw_descriptors,
        const Eigen::MatrixXf& projection_matrix, int target_dimensions,
        common::DescriptorsMatF32* projected_descriptors_ptr) {
  if (raw_descriptors.cols() == 0) {
    return;
  }
  common::DescriptorsMatF32& projected_descriptors = *CHECK_NOTNULL(projected_descriptors_ptr);
  projected_descriptors_ptr->resize(target_dimensions, raw_descriptors.cols());
  Eigen::MatrixXf converted_descriptors;
  const int num_descriptor_bytes = raw_descriptors.rows();
  const int num_descriptor_bits = num_descriptor_bytes * 8;
  converted_descriptors.resize(num_descriptor_bits, raw_descriptors.cols());
  for (int i = 0; i < raw_descriptors.cols(); ++i) {
    DescriptorToEigenMatrix(
        raw_descriptors.block(0, i, raw_descriptors.rows(), 1),
        converted_descriptors.block(0, i, num_descriptor_bits, 1));
  }

  if (projection_matrix.cols() == 471) {
    CHECK_EQ(512, num_descriptor_bits)
        << "Projection matrix dimensions don't match the descriptor length. "
        << "Double check your setting for feature_descriptor_type.";
  } else {
    CHECK_EQ(projection_matrix.cols(), num_descriptor_bits)
        << "Projection matrix dimensions don't match the descriptor length. "
        << "Double check your setting for feature_descriptor_type.";
  }
  projected_descriptors =
      projection_matrix.block(
          0, 0, target_dimensions, projection_matrix.cols()) *
      converted_descriptors.block(
          0, 0, projection_matrix.cols(), projected_descriptors.cols());
}

}
