#ifndef OCCUPANCY_GRID_POINTCLOUD_KDTREE_H_
#define OCCUPANCY_GRID_POINTCLOUD_KDTREE_H_

#include <Eigen/Core>
#include "data_common/state_structures.h"
#include "flann_common/nano_flann.hpp"
#include "flann_common/pointcloud.h"

namespace common {

typedef nano_flann::KDTreeSingleIndexAdaptor<
    nano_flann::L2_Simple_Adaptor<double,
    nano_flann::PointCloud<double>>,
    nano_flann::PointCloud<double>, 3> position_kdtree;

typedef nano_flann::KDTreeSingleIndexAdaptor<
    nano_flann::L2_Simple_Adaptor<double,
    common::PointCloudMap>, common::PointCloudMap, 3> lidar_kdtree;

class PointCloudKDTreeBase {
 public:
    PointCloudKDTreeBase() = default;

    virtual ~PointCloudKDTreeBase() = default;

    virtual void reIndex() = 0;

    virtual size_t knnSearch(
            const Eigen::Vector3d& query_point,
            const size_t num_closest,
            Eigen::Vector3d* out_points,
            double* out_distances_sq) const = 0;

    virtual void addPointclouds(
            const Eigen::Vector3d& position,
            const common::PointCloud& points) = 0;

    virtual size_t frameSize() const = 0;

    virtual void removeOldest() = 0;

    virtual common::EigenVector3dVec getLocalMapPoints() const = 0;
};

class QueuePointCloudKDTree: public PointCloudKDTreeBase {
 public:
    QueuePointCloudKDTree();

    ~QueuePointCloudKDTree() = default;

    void reIndex() override;

    size_t knnSearch(
            const Eigen::Vector3d& query_point,
            const size_t num_closest,
            Eigen::Vector3d* out_points,
            double* out_distances_sq) const override;

    void addPointclouds(
            const Eigen::Vector3d& position,
            const common::PointCloud& points) override;

    size_t frameSize() const override {
        return cloud_map_->pc_idx.size();
    }

    void removeOldest() override;

    common::EigenVector3dVec getLocalMapPoints() const override;

 private:
    std::unique_ptr<common::PointCloudMap> cloud_map_;
    std::unique_ptr<lidar_kdtree> cloud_knns_;

    std::unique_ptr<nano_flann::PointCloud<double>> position_map_;
    std::unique_ptr<position_kdtree> position_knns_;


    // Check do we have to update the kd_tree.
    bool need_update_;
};

class MultiPointCloudKDTree: public PointCloudKDTreeBase {
 public:
    MultiPointCloudKDTree();

    ~MultiPointCloudKDTree() = default;

    void reIndex() override;

    size_t knnSearch(
            const Eigen::Vector3d& query_point,
            const size_t num_closest,
            Eigen::Vector3d* out_points,
            double* out_distances_sq) const override;

    void addPointclouds(
            const Eigen::Vector3d& position,
            const common::PointCloud& points) override;

    size_t frameSize() const override {
        return tree_queue_.size();
    }

    void removeOldest() override;

    common::EigenVector3dVec getLocalMapPoints() const override;

 private:
    std::deque<std::shared_ptr<QueuePointCloudKDTree>> tree_queue_;
};

}

#endif
