#ifndef LOOP_CLOSURE_INVERTED_MULTI_INDEX_COMMON_H_
#define LOOP_CLOSURE_INVERTED_MULTI_INDEX_COMMON_H_

#include <aslam/common/memory.h>
#include <Eigen/Core>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nabo/nabo.h>
#include <opencv2/features2d.hpp>

const double kLoopClosureKnnEpsilon = 3.;
const double kKnnMaxRadius = 20.;

namespace loop_closure {

template <typename DescScalarType, int DescDim>
struct InvertedFile {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Eigen::Matrix<DescScalarType, DescDim, 1> Descriptor;
    // The descriptors stored in the InvertedFile.
    Aligned<std::vector, Descriptor> descriptors_;
    // Each stored descriptor has a corresponding index.
    std::vector<int> indices_;
};

typedef Nabo::NearestNeighbourSearch<float> NNSearch;
// Switch touch statistics (NNSearch::TOUCH_STATISTICS) off for performance.
static constexpr int kCollectTouchStatistics = 0;
// Kd-tree search options. ALLOW_SELF_MATCH means that a point which is
// equal to the query will be returned in the results.
static constexpr unsigned kSearchOptionsDefault =
    NNSearch::ALLOW_SELF_MATCH | NNSearch::SORT_RESULTS;

// Inserts a neighbor (defined by its id and distance to the query) in a list,
// which is sorted based on squared Euclidean distances. num_neighbors gives the
// maximum number of maintained neighbors. Neighbors not amongst the
// num_neighbors closest ones are not used.
inline static void InsertNeighbor(
    int index, float distance, int num_neighbors,
    std::vector<std::pair<float, int>>* const nearest_neighbors) {
    CHECK_NOTNULL(nearest_neighbors);
    const int num_found_neighbors = static_cast<int>(nearest_neighbors->size());
    if (num_found_neighbors >= num_neighbors) {
        if (nearest_neighbors->back().first < distance) {
            return;
        }
    }

    std::pair<float, int> neighbor(distance, index);
    std::vector<std::pair<float, int> >::iterator it = std::lower_bound(
        nearest_neighbors->begin(), nearest_neighbors->end(), neighbor);
    nearest_neighbors->insert(it, neighbor);

    if (num_found_neighbors >= num_neighbors) {
      nearest_neighbors->resize(num_neighbors);
    }

}

inline void MultiSequenceAlgorithm(
        const Eigen::VectorXi& indices_1, const Eigen::VectorXf& distances_1,
        const Eigen::VectorXi& indices_2, const Eigen::VectorXf& distances_2,
        int num_words, std::vector<std::pair<int, int> >* closest_words) {
    CHECK_NOTNULL(closest_words);
    CHECK_EQ(indices_1.rows(), distances_1.rows());
    CHECK_EQ(indices_2.rows(), distances_2.rows());
    const int num_dist1 = indices_1.rows();
    const int num_dist2 = indices_2.rows();
    const int max_num_words = num_dist1 * num_dist2;
    closest_words->reserve(std::min(num_words, max_num_words)); //WE have one here
    closest_words->clear();

    std::vector<bool> pair_used(num_dist1 * num_dist2, false);
    // We use a priority queue that stores the words from the product vocabulary
    // based on their distances to the query. Each product word is represented as
    // a tuple (d, i1, i2), where d is the squared distance to the query.
    // indices_1[i1] and indices_2[i2] are the indices of the corresponding word
    // from the two lower dimensional dictionaries that form the product word.
    std::priority_queue<std::tuple<float, int, int>,
            std::vector<std::tuple<float, int, int> >,
            std::greater<std::tuple<float, int, int> > >
            pqueue;
    pqueue.emplace(distances_1(0, 0) + distances_2(0, 0), 0, 0);

    while (!pqueue.empty() &&
           static_cast<int>(closest_words->size()) < num_words) {
        const int index1 = std::get<1>(pqueue.top());
        const int index2 = std::get<2>(pqueue.top());
        const int word_index = index1 * num_dist2 + index2;
        pair_used[word_index] = true;
        closest_words->emplace_back(indices_1(index1, 0), indices_2(index2, 0));
        pqueue.pop();

        if ((index1 + 1) < num_dist1) {
            if (index2 == 0 || pair_used[word_index + num_dist2 - 1]) {
                pqueue.emplace(
                            distances_1(index1 + 1, 0) + distances_2(index2, 0), index1 + 1,
                            index2);
            }
        }

        if ((index2 + 1) < num_dist2) {
            if (index1 == 0 || pair_used[word_index - num_dist2 + 1]) {
                pqueue.emplace(
                            distances_1(index1, 0) + distances_2(index2 + 1, 0), index1,
                            index2 + 1);
            }
        }
    }
}

template <int kDimSubVectors, typename DerivedQuery>
inline void FindClosestWords(
        const Eigen::MatrixBase<DerivedQuery>& query_feature, int num_closest_words,
        const NNSearch& words_1_index, const NNSearch& words_2_index,
        int num_words_1, int num_words_2,
        std::vector<std::pair<int, int> >* closest_words) {
    CHECK_NOTNULL(closest_words);
    CHECK_GT(num_closest_words, 0);
    CHECK_EQ(query_feature.rows(), 2 * kDimSubVectors);
    CHECK_EQ(query_feature.cols(), 1);

    // Computes the squared Euclidean distances to all visual words in the lower
    // dimensional vocabularies by splitting the query descriptor.
    // The words are then sorted based on their distances.
    int num_neighbors = std::min<int>(num_words_1, num_closest_words);
    CHECK_GT(num_neighbors, 0) << num_words_1 << " " << num_closest_words;
    NNSearch::IndexVector indices_1;
    indices_1.resize(num_neighbors, 1);
    NNSearch::Vector distances_1;
    distances_1.resize(num_neighbors, 1);
    Eigen::VectorXf query = query_feature.template head<kDimSubVectors>();
    words_1_index.knn(
        query, indices_1, distances_1, num_neighbors, kLoopClosureKnnEpsilon,
        kSearchOptionsDefault, kKnnMaxRadius);

    num_neighbors = std::min<int>(num_words_2, num_closest_words);
    CHECK_GT(num_neighbors, 0) << num_words_2 << " " << num_closest_words;
    NNSearch::IndexVector indices_2;
    indices_2.resize(num_neighbors, 1);
    NNSearch::Vector distances_2;
    distances_2.resize(num_neighbors, 1);
    query = query_feature.template tail<kDimSubVectors>();
    words_2_index.knn(
        query, indices_2, distances_2, num_neighbors, kLoopClosureKnnEpsilon,
        kSearchOptionsDefault, kKnnMaxRadius);

    // Given the distances to the lower dimensional words, finds the closest
    // words in the product vocabulary.
    MultiSequenceAlgorithm(
                indices_1, distances_1, indices_2, distances_2, num_closest_words,
                closest_words);
}

template <typename DescType, int DescDim>
void AddDescriptor(
        const Eigen::Matrix<DescType, DescDim, 1>& descriptor, int descriptor_id,
        int word_index, std::unordered_map<int, int>* word_index_map,
        Aligned<std::vector, InvertedFile<DescType, DescDim> >* inverted_files) {
    CHECK_NOTNULL(word_index_map);
    CHECK_NOTNULL(inverted_files);

    Aligned<std::vector, InvertedFile<DescType, DescDim> >& inverted_files_ref =
            *inverted_files;

    std::unordered_map<int, int>::const_iterator word_index_it;
    word_index_it = word_index_map->find(word_index);

    if (word_index_it == word_index_map->end()) {
        word_index_map->emplace(
                    word_index, static_cast<int>(inverted_files_ref.size()));

        InvertedFile<DescType, DescDim> new_inverted_file;
        new_inverted_file.descriptors_.emplace_back(descriptor);
        new_inverted_file.indices_.emplace_back(descriptor_id);
        inverted_files_ref.push_back(new_inverted_file);
    } else {
        inverted_files_ref[word_index_it->second].descriptors_.emplace_back(
                    descriptor);
        inverted_files_ref[word_index_it->second].indices_.emplace_back(
                    descriptor_id);
    }
}

}
#endif
