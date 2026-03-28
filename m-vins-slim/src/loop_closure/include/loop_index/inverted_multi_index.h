#ifndef LOOP_CLOSURE_INVERTED_MULTI_INDEX_H_
#define LOOP_CLOSURE_INVERTED_MULTI_INDEX_H_

#include <Eigen/Core>
#include <aslam/common/memory.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "data_common/visual_structures.h"
#include "loop_index/inverted_multi_index_common.h"

namespace loop_closure {

template <int kDimSubVectors>
class InvertedMultiIndex {
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
typedef Eigen::Matrix<float, 2 * kDimSubVectors, 1> DescriptorType;
typedef Eigen::Matrix<float, kDimSubVectors, 1> SubDescriptorType;
typedef Eigen::Matrix<float, 2 * kDimSubVectors, Eigen::Dynamic>
    DescriptorMatrixType;
typedef InvertedFile<float, 2 * kDimSubVectors> InvFile;
// Creates the index from a given set of visual words. Each column in words_i
// specifies a cluster center coordinate.

InvertedMultiIndex(
    const Eigen::MatrixXf& words_1, const Eigen::MatrixXf& words_2,
    int num_closest_words_for_nn_search)
    : words_1_(words_1),
      words_2_(words_2),
      words_1_index_(
          NNSearch::createKDTreeLinearHeap(
              words_1_, kDimSubVectors, kCollectTouchStatistics)),
      words_2_index_(
          NNSearch::createKDTreeLinearHeap(
              words_2_, kDimSubVectors, kCollectTouchStatistics)),
      num_closest_words_for_nn_search_(num_closest_words_for_nn_search),
      max_db_descriptor_index_(0) {
  CHECK_EQ(words_1.rows(), kDimSubVectors);
  CHECK_GT(words_1.cols(), 0);
  CHECK_EQ(words_2.rows(), kDimSubVectors);
  CHECK_GT(words_2.cols(), 0);
  CHECK_GT(num_closest_words_for_nn_search_, 0);
}

// Clears the inverted index by removing all references to the database
// descriptors stored in it. Does NOT remove the underlying quantization.
inline void Clear() {
  word_index_map_.clear();
  max_db_descriptor_index_ = 0;
}

void AddDescriptors(const DescriptorMatrixType& descriptors) {
    const int num_descriptors = descriptors.cols();
    std::vector<std::pair<int, int> > closest_word;

    for (int i = 0; i < num_descriptors; ++i) {
        // Find closest visual words.
        FindClosestWords<kDimSubVectors>(
                    descriptors.col(i), 1, *words_1_index_, *words_2_index_,
                    words_1_.cols(), words_2_.cols(), &closest_word);
        CHECK(!closest_word.empty());

        // we have only one closest_word here
        const int word_index =
                closest_word[0].first * words_2_.cols() + closest_word[0].second;

        AddDescriptor<float, 2 * kDimSubVectors>(
                    descriptors.col(i), max_db_descriptor_index_, word_index,
                    &word_index_map_, &inverted_files_);
        ++max_db_descriptor_index_;
    }

    // TODO: Add pose map.
}

template <typename DerivedQuery, typename DerivedIndices,
          typename DerivedDistances>
inline void GetNNearestNeighbors(
        const Eigen::MatrixBase<DerivedQuery>& query_feature, int num_neighbors,
        const Eigen::MatrixBase<DerivedIndices>& out_indices,
        const Eigen::MatrixBase<DerivedDistances>& out_distances) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(
                DerivedQuery, static_cast<int>(2 * kDimSubVectors), 1);
    CHECK_EQ(out_indices.cols(), 1);
    CHECK_EQ(out_distances.cols(), 1);
    CHECK_GT(num_neighbors, 0);

    CHECK_EQ(out_indices.rows(), num_neighbors)
            << "The indices parameter must be pre-allocated to hold all results.";
    CHECK_EQ(out_distances.rows(), num_neighbors)
            << "The distances parameter must be pre-allocated to hold all results.";

    // Zero copy passing of Eigen-Block Expressions.
    // http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
    Eigen::MatrixBase<DerivedIndices>& indices =
            const_cast<Eigen::MatrixBase<DerivedIndices>&>(out_indices);
    Eigen::MatrixBase<DerivedDistances>& distances =
            const_cast<Eigen::MatrixBase<DerivedDistances>&>(out_distances);

    // Find the closest visual words.
    std::vector<std::pair<int, int> > closest_words;
        FindClosestWords<kDimSubVectors>(
            query_feature, num_closest_words_for_nn_search_, *words_1_index_,
            *words_2_index_, words_1_.cols(), words_2_.cols(), &closest_words);
    // Performs exhaustive search through all descriptors assigned to the
    // closest words.
    std::vector<std::pair<float, int>> nearest_neighbors;
    nearest_neighbors.reserve(num_neighbors + 1);
    const int num_words_to_use = static_cast<int>(closest_words.size());
    std::unordered_map<int, int>::const_iterator word_index_map_it;

    // for every closest words here
    for (int i = 0; i < num_words_to_use; ++i) {
        const int word_index =
                closest_words[i].first * words_2_.cols() + closest_words[i].second;

        // find the word index map to get the affiliated descriptor of this word
        word_index_map_it = word_index_map_.find(word_index);
        if (word_index_map_it == word_index_map_.end()) {
            continue;
        }

        const InvFile& inverted_file = inverted_files_[word_index_map_it->second];
        const size_t num_descriptors = inverted_file.descriptors_.size();

        for (size_t j = 0u; j < num_descriptors; ++j) {
            const float distance =
                (inverted_file.descriptors_[j] - query_feature).squaredNorm();
            InsertNeighbor(inverted_file.indices_[j], distance, num_neighbors,
                           &nearest_neighbors);
        }
    }

    for (size_t i = 0u; i < nearest_neighbors.size(); ++i) {
        indices(i, 0) = nearest_neighbors[i].second;
        distances(i, 0) = nearest_neighbors[i].first;
    }
    for (int i = static_cast<int>(nearest_neighbors.size()); i < num_neighbors; ++i) {
        indices(i, 0) = -1;
        distances(i, 0) = std::numeric_limits<float>::infinity();
    }
}

int GetNumDescriptorsInIndex() const {
    return max_db_descriptor_index_;
}

protected:
 // The two sets of cluster centers defining the quantization of the descriptor
 // space as the Cartesian product of the two sets of words.
 Eigen::MatrixXf words_1_;
 Eigen::MatrixXf words_2_;
 std::shared_ptr<NNSearch> words_1_index_;
 std::shared_ptr<NNSearch> words_2_index_;

 // The number of closest words from the product vocabulary that should be used
 // during nearest neighbor search.
 int num_closest_words_for_nn_search_;
 // Hashmap storing for each combined visual word the index in inverted_files_
 // in which all database descriptors assigned to that word can be found.
 // This allows us to easily add descriptors assigned to words that have not
 // been used previously without having to re-order large amounts of memory.
 std::unordered_map<int, int> word_index_map_;
 // Vector containing the inverted files, one for each visual word in the
 // product vocabulary. Each inverted file holds all descriptors assigned to
 // the corresponding word and their indices.
 Aligned<std::vector, InvFile> inverted_files_;
 // The maximum index of the descriptor indices.
 int max_db_descriptor_index_;
};

}
#endif
