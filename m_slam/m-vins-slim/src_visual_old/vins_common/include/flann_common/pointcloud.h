#ifndef LOOP_CLOSURE_FLANN_POINTCLOUD_H_
#define LOOP_CLOSURE_FLANN_POINTCLOUD_H_

#include <memory>
#include <queue>
#include <vector>
#include <Eigen/Core>

#include "flann_common/nano_flann.hpp"

namespace nano_flann {
template <typename T>
struct PointCloud {
    struct Point {
        Point() {
            values.resize(0);
        }

        Point(T _x, T _y, T _z) {
            values.resize(3);
            values[0] = _x;
            values[1] = _y;
            values[2] = _z;
        }

        Point(Eigen::Matrix<T, Eigen::Dynamic, 1> descriptors) {
            values.resize(descriptors.rows());
            for (size_t i = 0u; i < descriptors.rows(); ++i) {
                values[i] = descriptors(i, 0);
            }
        }

        std::vector<T> values;
    };

    PointCloud() = default;

    PointCloud(const PointCloud& point_cloud) {
        points.assign(point_cloud.points.begin(), point_cloud.points.end());
    }

    std::vector<Point> points;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, int dim) const {
        return points[idx].values[dim];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop. Return true if the BBOX was already computed by
    // the class and returned in "bb" so it can be avoided to redo it again.
    // Look at bb.size() to find out the expected dimensionality
    // (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};

template<typename T, int DIM>
class DfsCluster final {
    enum ERROR_TYPE {
        SUCCESS = 0,
        FAILED = 1
    };

 public:
    DfsCluster() = default;
    explicit DfsCluster(const int& min_points) :
        minpts_(min_points) {}

    ~DfsCluster() = default;

    int Run(const std::shared_ptr<nano_flann::PointCloud<T>>& pointcloud,
            const T& eps);

    std::vector<std::vector<size_t>> clusters_;
    std::vector<size_t> noise_;

 private:
    std::vector<size_t> RegionQuery(const size_t& pid) const;

    void AddToCluster(const size_t& pid, const size_t& cid);

    void ExpandCluster(const size_t& cid,
            const std::vector<size_t>& neighbors);

    void AddToBorderSet(const size_t& pid) {
        this->borderset_.insert(pid);
    }

    void AddToBorderSet(const std::vector<size_t>& pids) {
        for (const auto& pid : pids) {
            this->borderset_.insert(pid);
        }
    }

    bool IsInBorderSet(const size_t& pid) const {
        return borderset_.count(pid);
    }

 private:
    // Temporary variables used during computation
    std::vector<bool> visited_;
    std::vector<bool> assigned_;
    std::set<size_t> borderset_;

    // Minimum and maximum points in a cluster.
    const size_t minpts_ = 2u;
    T epsilon_;
    std::shared_ptr<nano_flann::PointCloud<T>> data_;

    // KNN search.
    std::shared_ptr<nano_flann::KDTreeSingleIndexAdaptor<
            nano_flann::L2_Simple_Adaptor<T, nano_flann::PointCloud<T>>,
            nano_flann::PointCloud<T>, DIM>> knns_;

};

template<typename T, int DIM>
int DfsCluster<T, DIM>::Run(
        const std::shared_ptr<nano_flann::PointCloud<T>>& pointcloud,
        const T& eps) {
    if (pointcloud->points.empty()) {
        return ERROR_TYPE::FAILED;
    }

    const size_t datalen = pointcloud->points.size();
    this->visited_ = std::vector<bool>(datalen, false);
    this->assigned_ = std::vector<bool>(datalen, false);
    this->clusters_.clear();
    this->noise_.clear();
    this->data_ = pointcloud;
    this->epsilon_ = eps;

    knns_.reset(new nano_flann::KDTreeSingleIndexAdaptor<
            nano_flann::L2_Simple_Adaptor<T, nano_flann::PointCloud<T>>,
            nano_flann::PointCloud<T>, DIM>(
                    DIM, *pointcloud,
                    nano_flann::KDTreeSingleIndexAdaptorParams(10)));
    knns_->buildIndex();

    for (size_t pid = 0u; pid < datalen; ++pid) {
        this->borderset_.clear();
        if (!this->visited_[pid]) {
            this->visited_[pid] = true;

            const auto& neightbors = this->RegionQuery(pid);
            if (neightbors.size() >= this->minpts_ - 1u) {
                auto cid = this->clusters_.size();
                this->clusters_.emplace_back(std::vector<size_t>());
                this->AddToBorderSet(pid);
                this->AddToCluster(pid, cid);
                this->ExpandCluster(cid, neightbors);
            }
        }
    }

    for (size_t pid = 0u; pid < datalen; ++pid) {
        if (!this->assigned_[pid]) {
            this->noise_.emplace_back(pid);
        }
    }

    return ERROR_TYPE::SUCCESS;
}

template<typename T, int DIM>
std::vector<size_t> DfsCluster<T, DIM>::RegionQuery(const size_t& pid) const {

    T query_pt[DIM];
    for (int i = 0; i < DIM; ++i) {
        query_pt[i] = data_->kdtree_get_pt(pid, i);
    }

    std::vector<std::pair<size_t, T>> ret_matches;
    nano_flann::SearchParams params;
    const size_t nMatches = knns_->radiusSearch(
            &query_pt[0], this->epsilon_, ret_matches, params);

    std::vector<size_t> neighbors;
    // The first neighbour is itself.
    for (size_t i = 1u; i < nMatches; ++i) {
        neighbors.emplace_back(ret_matches[i].first);
    }

    return neighbors;
}

template<typename T, int DIM>
void DfsCluster<T, DIM>::ExpandCluster(const size_t& cid,
        const std::vector<size_t>& neighbors) {
    // It has unvisited , visited unassigned pts.
    // Visited assigned will not appear
    std::queue<size_t> border;
    for (const auto& pid : neighbors) {
        border.push(pid);
    }
    this->AddToBorderSet(neighbors);

    while (!border.empty()) {
        const size_t pid = border.front();
        border.pop();

        if (!this->visited_[pid]) {

            // Not been visited, great! , hurry to mark it visited
            this->visited_[pid] = true;
            const auto& pidneighbors = this->RegionQuery(pid);

            // Core point, the neighbors will be expanded
            if (pidneighbors.size() >= this->minpts_ - 1) {
                this->AddToCluster(pid, cid);
                for (const auto& pidnid : pidneighbors) {
                    if (!this->IsInBorderSet(pidnid)) {
                        border.push(pidnid);
                        this->AddToBorderSet(pidnid);
                    }
                }
            }
        }
    }
}

template<typename T, int DIM>
void DfsCluster<T, DIM>::AddToCluster(const size_t& pid, const size_t& cid) {
    this->clusters_[cid].emplace_back(pid);
    this->assigned_[pid] = true;
}

}

#endif
