#ifndef MVINS_OCTO_MAPPER_H_
#define MVINS_OCTO_MAPPER_H_

#include <mutex>

#include <aslam/cameras/ncamera.h>
#include <octomap/ColorOcTree.h>
#include <octomap/math/Pose6D.h>
#include <opencv2/opencv.hpp>

#include "data_common/sensor_structures.h"
#include "octomap_core/pointcloud_filter.h"

namespace octomap {

class OctoMapper {
public:
    OctoMapper(const std::string& setting_path,
                const double resolution,
                const int pick_step,
                const int pt_in_grid_per_metre_th);
    OctoMapper(const double resolution,
                const int pick_step,
                const aslam::NCamera::Ptr& cameras,
                const int pt_in_grid_per_metre_th);
    ~OctoMapper() = default;
    void GetPointCloud(
            const common::ImageData& img_data,
            octomap::PointCloudXYZRGB* point_cloud_ptr);
    octomap::OcTreeKey CoordToKey(const octomap::point3d& pt);
    void SetBbx(octomap::point3d& max,
                octomap::point3d& min);
    void CastRay(const octomap::point3d& origin,
                 const octomap::point3d& end,
                 octomap::KeySet* free_cells_ptr,
                 octomap::KeySet* occupied_cells_ptr);
    void UpdateOccupancy(octomap::KeySet* free_cells_ptr,
                           octomap::KeySet* occupied_cells_ptr);
    void UpdateColor(const octomap::Pointcloud& pt,
                      const octomap::Pointcloud& color);
    void UpdateInnerOccupancy();
    void DeleteNodes(const octomap::KeySet& free_cells,
                      const octomap::KeySet& occupied_cells);
    void Clear();
    void ResetOcTree();
    void Prune();
    void LoadPose(const std::string& pose_path);
    void LoadDepth(const std::string& bag_path,
                    const std::string& depth_topic_name);
    std::shared_ptr<octomap::ColorOcTree> GetOcTree();
    std::shared_ptr<octomap::ColorOcTree> GetOcTreeCopy();
    void SaveOctoMap(const std::string& path);
private:
    aslam::NCamera::Ptr cameras_;

    const double resolution_;
    const int pick_step_;
    const int pt_in_grid_per_metre_th_;
    std::shared_ptr<octomap::ColorOcTree> octree_;

    std::mutex octree_mutex_, removal_mutex_;
    std::unique_ptr<vins_core::RealTimeOutlierRemoval> outlier_removal_;
};

}
#endif

