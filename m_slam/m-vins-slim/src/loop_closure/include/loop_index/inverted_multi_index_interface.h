#ifndef LOOP_CLOSURE_INVERTED_MULTI_INDEX_INTERFACE_H_
#define LOOP_CLOSURE_INVERTED_MULTI_INDEX_INTERFACE_H_

#include "data_common/visual_structures.h"
#include "file_common/binary_serialization.h"
#include "loop_index/helpers.h"
#include "loop_index/index_interface.h"
#include "loop_index/inverted_multi_index.h"

namespace loop_closure {

class InvertedMultiIndexVocabulary {
 public:
#ifdef USE_CNN_FEATURE
  enum { kTargetDimensionality = 256 };
#else
  enum { kTargetDimensionality = 10 };
#endif
  enum { kSerializationVersion = 100 };
  InvertedMultiIndexVocabulary() {
    target_dimensionality_ = kTargetDimensionality;
  }
  inline void Save(std::ofstream* out_stream) const {
    CHECK_NOTNULL(out_stream);
    int serialized_version = kSerializationVersion;
    common::Serialize(serialized_version, out_stream);
    common::Serialize(target_dimensionality_, out_stream);
    common::Serialize(projection_matrix_, out_stream);
    common::Serialize(words_first_half_, out_stream);
    common::Serialize(words_second_half_, out_stream);
  }
  inline void Load(std::ifstream* in_stream) {
    CHECK_NOTNULL(in_stream);
    int deserialized_version;
    common::Deserialize(&deserialized_version, in_stream);

    int serialized_target_dimensionality;
    common::Deserialize(&serialized_target_dimensionality, in_stream);
    CHECK_EQ(serialized_target_dimensionality, target_dimensionality_);
#ifndef USE_CNN_FEATURE
    common::Deserialize(&projection_matrix_, in_stream);
#endif
    common::Deserialize(&words_first_half_, in_stream);
    common::Deserialize(&words_second_half_, in_stream);
  }
  int target_dimensionality_;
  Eigen::MatrixXf words_first_half_;
  Eigen::MatrixXf words_second_half_;
  Eigen::MatrixXf projection_matrix_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class InvertedMultiIndexInterface : public IndexInterface {
public:
#ifdef USE_CNN_FEATURE
    enum { kSubSpaceDimensionality = 128 };
#else
    enum { kSubSpaceDimensionality = 5 };
#endif
    typedef InvertedMultiIndex<kSubSpaceDimensionality> Index;

    InvertedMultiIndexInterface(const std::string& quantizer_filename,
                               int num_closest_words_for_nn_search) {
        std::ifstream in(quantizer_filename, std::ios_base::binary);
        CHECK(in.is_open()) << "Failed to read quantizer file from "
                            << quantizer_filename;

        vocabulary_.Load(&in);

        const Eigen::MatrixXf& words_1 = vocabulary_.words_first_half_;
        const Eigen::MatrixXf& words_2 = vocabulary_.words_second_half_;
        CHECK_GT(words_1.cols(), 0);
        CHECK_GT(words_2.cols(), 0);

        CHECK_EQ(kSubSpaceDimensionality, vocabulary_.target_dimensionality_ / 2);

        index_.reset(new Index(words_1, words_2, num_closest_words_for_nn_search));
    }
    ~InvertedMultiIndexInterface() = default;

    template <typename DerivedQuery, typename DerivedIndices,
              typename DerivedDistances>
    inline void GetNNearestNeighbors(
            const Eigen::MatrixBase<DerivedQuery>& query_feature, int num_neighbors,
            const Eigen::MatrixBase<DerivedIndices>& indices_const,
            const Eigen::MatrixBase<DerivedDistances>& distances_const) const {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(
            DerivedQuery, static_cast<int>(kSubSpaceDimensionality << 1), 1);
        CHECK_EQ(indices_const.cols(), 1);
        CHECK_EQ(distances_const.cols(), 1);

        CHECK_EQ(indices_const.rows(), num_neighbors)
                << "The indices parameter must be pre-allocated to hold all results.";
        CHECK_EQ(distances_const.rows(), num_neighbors)
                << "The distances parameter must be pre-allocated to hold all results.";
        index_->GetNNearestNeighbors(query_feature, num_neighbors, indices_const, distances_const);
    }

    template <typename DerivedQuery, typename DerivedIndices,
              typename DerivedDistances>
    inline void GetNNearestNeighborsForFeaturesImpl(
            const Eigen::MatrixBase<DerivedQuery>& query_features, int num_neighbors,
            const Eigen::MatrixBase<DerivedIndices>& indices_const,
            const Eigen::MatrixBase<DerivedDistances>& distances_const) const {
        Eigen::MatrixBase<DerivedIndices>& indices =
            internal::CastConstEigenMatrixToNonConst(indices_const);
        Eigen::MatrixBase<DerivedDistances>& distances =
            internal::CastConstEigenMatrixToNonConst(distances_const);

        CHECK_EQ(indices_const.rows(), num_neighbors)
                << "The indices parameter must be pre-allocated to hold all results.";
        CHECK_EQ(distances_const.rows(), num_neighbors)
                << "The distances parameter must be pre-allocated to hold all results.";
        CHECK_EQ(indices_const.cols(), query_features.cols())
                << "The indices parameter must be pre-allocated to hold all results.";
        CHECK_EQ(distances_const.cols(), query_features.cols())
                << "The distances parameter must be pre-allocated to hold all results.";
        for (int i = 0; i < query_features.cols(); ++i) {
            GetNNearestNeighbors(
                        query_features.template block<2 * kSubSpaceDimensionality, 1>(0, i),
                        num_neighbors, indices.block(0, i, num_neighbors, 1),
                        distances.block(0, i, num_neighbors, 1));
        }
    }

    virtual void Clear();

    virtual void AddDescriptors(
             const common::DescriptorsMatF32& descriptors);

    virtual int GetNumDescriptorsInIndex() const;

    virtual void GetNNearestNeighborsForFeatures(
             const common::DescriptorsMatF32& query_features,
             const int num_neighbors,
             Eigen::MatrixXi* indices, Eigen::MatrixXf* distances) const;

    virtual void ProjectDescriptors(
             const common::DescriptorsMatUint8& raw_des,
             common::DescriptorsMatF32* projected_des) const;
private:
    std::shared_ptr<Index> index_;
    InvertedMultiIndexVocabulary vocabulary_;
};

}
#endif
