#include "occ_common/pointcloud_kdtree.h"

#include <numeric>

#include "occ_common/common.h"


namespace common {
QueuePointCloudKDTree::QueuePointCloudKDTree() :
    cloud_map_(std::make_unique<common::PointCloudMap>()),
    position_map_(std::make_unique<nano_flann::PointCloud<double>>()),
    need_update_(false) {

}

void QueuePointCloudKDTree::reIndex() {
    if (need_update_) {
        cloud_knns_.reset(new lidar_kdtree(3, *cloud_map_,
                    nano_flann::KDTreeSingleIndexAdaptorParams(10)));
        cloud_knns_->buildIndex();

        position_knns_.reset(
            new position_kdtree(3, *position_map_,
                nano_flann::KDTreeSingleIndexAdaptorParams(10)));
        position_knns_->buildIndex();

        need_update_ = false;
    }
}

size_t QueuePointCloudKDTree::knnSearch(
        const Eigen::Vector3d& query_point,
        const size_t num_closest,
        Eigen::Vector3d* out_points,
        double* out_distances_sq) const {
    std::vector<size_t> ret_index(num_closest);
    const size_t num_results = cloud_knns_->knnSearch(
                query_point.data(),
                num_closest,
                &ret_index[0],
                out_distances_sq);

    for (size_t i = 0u; i < num_results; ++i) {
        out_points[i] = cloud_map_->points[ret_index[i]];
    }

    return num_results;
}


void QueuePointCloudKDTree::addPointclouds(
        const Eigen::Vector3d& position,
        const common::PointCloud& points) {

    constexpr size_t num_closest = 1u;
    std::vector<double> dis(num_closest);
    std::vector<size_t> index(num_closest);

    if (position_knns_ != nullptr) {
        position_knns_->knnSearch(
                    position.data(),
                    num_closest,
                    &index[0],
                    &dis[0]);
    }
    if (position_knns_ == nullptr || dis[0] > 2.0) {
        *cloud_map_ += points;
        position_map_->points.emplace_back(
                    nano_flann::PointCloud<double>::
                    Point(position(0),
                          position(1),
                          position(2)));
        need_update_ = true;
    }
}

void QueuePointCloudKDTree::removeOldest() {
    cloud_map_->RemoveFirstPointCloud();
    position_map_->points.erase(position_map_->points.begin());
}


common::EigenVector3dVec QueuePointCloudKDTree::getLocalMapPoints() const {
    common::EigenVector3dVec global_ps(cloud_map_->points.size());
    for (size_t idx = 0u; idx < cloud_map_->points.size(); ++idx) {
        global_ps[idx] = cloud_map_->points[idx];
    }

    return global_ps;
}

MultiPointCloudKDTree::MultiPointCloudKDTree() {

}

void MultiPointCloudKDTree::reIndex() {
    // NOTE: Do nothing.
}

size_t MultiPointCloudKDTree::knnSearch(
        const Eigen::Vector3d& query_point,
        const size_t num_closest,
        Eigen::Vector3d* out_points,
        double* out_distances_sq) const {

    std::vector<double> all_dis;
    common::EigenVector3dVec all_points;
    for (const auto& knn : tree_queue_) {
        std::vector<double> ret_dis(num_closest);
        common::EigenVector3dVec ret_points(num_closest);
        knn->knnSearch(
                    query_point,
                    num_closest,
                    &ret_points[0],
                    &ret_dis[0]);
        all_dis.insert(all_dis.end(),
                       ret_dis.begin(), ret_dis.end());
        all_points.insert(all_points.end(),
                          ret_points.begin(), ret_points.end());
    }

    std::vector<std::size_t> index(all_dis.size());
    std::iota(std::begin(index), std::end(index), 0u);
    std::partial_sort(std::begin(index),
                      std::begin(index) + num_closest, std::end(index),
                      [&all_dis](const size_t& lhs, const size_t& rhs)
    {return all_dis[lhs] < all_dis[rhs];});

    for (size_t i = 0u; i < num_closest; ++i) {
        out_distances_sq[i] = all_dis[index[i]];
        out_points[i] = all_points[index[i]];
    }

    return num_closest;
}

void MultiPointCloudKDTree::addPointclouds(
        const Eigen::Vector3d& position,
        const common::PointCloud& points) {
    std::shared_ptr<QueuePointCloudKDTree> single_tree_ptr =
        std::make_shared<QueuePointCloudKDTree>();
    single_tree_ptr->addPointclouds(position, points);
    single_tree_ptr->reIndex();

    tree_queue_.push_back(single_tree_ptr);
}

void MultiPointCloudKDTree::removeOldest() {
    tree_queue_.pop_front();
}

common::EigenVector3dVec MultiPointCloudKDTree::getLocalMapPoints() const {
    return tree_queue_.back()->getLocalMapPoints();
}

}
