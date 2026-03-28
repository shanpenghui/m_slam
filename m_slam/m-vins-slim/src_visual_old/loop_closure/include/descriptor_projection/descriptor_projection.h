#ifndef LOOP_CLOSURE_DESCRIPTOR_PROJECTION_H_
#define LOOP_CLOSURE_DESCRIPTOR_PROJECTION_H_

#include "data_common/visual_structures.h"

namespace loop_closure {

void ProjectDescriptorBlock(
    const common::DescriptorsMatUint8& raw_descriptors,
    const Eigen::MatrixXf& projection_matrix, int target_dimensions,
    common::DescriptorsMatF32* projected_descriptors_ptr);

template <typename DerivedIn, typename DerivedOut>
void DescriptorToEigenMatrix(
    const Eigen::MatrixBase<DerivedIn>& descriptor,
    const Eigen::MatrixBase<DerivedOut>& matrix_const) {
  EIGEN_STATIC_ASSERT(
      !(Eigen::internal::traits<DerivedOut>::Flags & Eigen::RowMajorBit),
      "This method is only valid for column major matrices");
  CHECK_EQ(descriptor.cols(), 1);
  Eigen::MatrixBase<DerivedOut>& matrix =
      const_cast<Eigen::MatrixBase<DerivedOut>&>(matrix_const);
  const int num_descriptor_bytes = descriptor.rows();
  const int num_descriptor_bits = num_descriptor_bytes * 8;
  CHECK_EQ(matrix.rows(), num_descriptor_bits)
      << "The matrix passed must be preallocated to match the descriptor "
         "length in bits, which is "
      << num_descriptor_bits << ".";
  matrix.setZero();

  CHECK_EQ(num_descriptor_bytes % 16, 0);

// Define a set of macros to NEON and SSE instructions so we can use the same
// code further down for both platforms.
#ifdef __ARM_NEON__
#define VECTOR_SET vdupq_n_u8       // Set a vector from a single uint8.
#define VECTOR_LOAD(x) vld1q_u8(x)  // Set a vector from a mem location.
#define VECTOR_TYPE uint8x16_t      // The type of the vector element.
#define VECTOR_AND vandq_u8         // The vector AND instruction.
#define VECTOR_EXTRACT(x, i) vgetq_lane_u8(x, i)  // Get element from vector.
#else
#define VECTOR_SET _mm_set1_epi8
#define VECTOR_LOAD(x) _mm_load_si128(reinterpret_cast<const __m128i*>(x))
#define VECTOR_TYPE __m128i
#define VECTOR_AND _mm_and_si128
// Could use _mm_extract_epi8, but this requires SSE4.1.
#define VECTOR_EXTRACT(x, i) reinterpret_cast<const char*>(&x)[i]
#endif  // ANDROID

  VECTOR_TYPE mask[8];
  mask[0] = VECTOR_SET(static_cast<char>(1 << 0));
  mask[1] = VECTOR_SET(static_cast<char>(1 << 1));
  mask[2] = VECTOR_SET(static_cast<char>(1 << 2));
  mask[3] = VECTOR_SET(static_cast<char>(1 << 3));
  mask[4] = VECTOR_SET(static_cast<char>(1 << 4));
  mask[5] = VECTOR_SET(static_cast<char>(1 << 5));
  mask[6] = VECTOR_SET(static_cast<char>(1 << 6));
  mask[7] = VECTOR_SET(static_cast<char>(1 << 7));

  float* matrix_ref = matrix.derived().data();

  CHECK_EQ(descriptor.derived().cols(), 1);

  const unsigned char* descriptor_data = &descriptor.derived().coeffRef(0, 0);

  for (int pack = 0; pack < num_descriptor_bytes / 16; ++pack) {
    VECTOR_TYPE value = VECTOR_LOAD(descriptor_data + pack * 16);
    const int pack128 = pack << 7;
    for (int i = 0; i < 8; ++i) {  // Checks 16 bits at once with SSE/NEON.
      // Masks the i'th bit of the 16 uint8s.
      VECTOR_TYPE xmm1 = VECTOR_AND(value, mask[i]);
      if (VECTOR_EXTRACT(xmm1, 0))
        matrix_ref[pack128 + i + 0] = 1;
      if (VECTOR_EXTRACT(xmm1, 1))
        matrix_ref[pack128 + i + 8] = 1;
      if (VECTOR_EXTRACT(xmm1, 2))
        matrix_ref[pack128 + i + 16] = 1;
      if (VECTOR_EXTRACT(xmm1, 3))
        matrix_ref[pack128 + i + 24] = 1;
      if (VECTOR_EXTRACT(xmm1, 4))
        matrix_ref[pack128 + i + 32] = 1;
      if (VECTOR_EXTRACT(xmm1, 5))
        matrix_ref[pack128 + i + 40] = 1;
      if (VECTOR_EXTRACT(xmm1, 6))
        matrix_ref[pack128 + i + 48] = 1;
      if (VECTOR_EXTRACT(xmm1, 7))
        matrix_ref[pack128 + i + 56] = 1;
      if (VECTOR_EXTRACT(xmm1, 8))
        matrix_ref[pack128 + i + 64] = 1;
      if (VECTOR_EXTRACT(xmm1, 9))
        matrix_ref[pack128 + i + 72] = 1;
      if (VECTOR_EXTRACT(xmm1, 10))
        matrix_ref[pack128 + i + 80] = 1;
      if (VECTOR_EXTRACT(xmm1, 11))
        matrix_ref[pack128 + i + 88] = 1;
      if (VECTOR_EXTRACT(xmm1, 12))
        matrix_ref[pack128 + i + 96] = 1;
      if (VECTOR_EXTRACT(xmm1, 13))
        matrix_ref[pack128 + i + 104] = 1;
      if (VECTOR_EXTRACT(xmm1, 14))
        matrix_ref[pack128 + i + 112] = 1;
      if (VECTOR_EXTRACT(xmm1, 15))
        matrix_ref[pack128 + i + 120] = 1;
    }
  }
}

}

#endif
