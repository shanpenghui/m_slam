#include "octomap_core/octo_mapper.h"

#include <thread>

#include <aslam/cameras/camera.h>
#include <glog/logging.h>
#include <Eigen/Core>

#include "data_common/constants.h"
#include "time_common/time.h"

namespace octomap {

OctoMapper::OctoMapper(const std::string& setting_path,
                       const double resolution,
                       const int pick_step,
                       const int pt_in_grid_per_metre_th)
    : resolution_(resolution),
      pick_step_(pick_step),
      pt_in_grid_per_metre_th_(pt_in_grid_per_metre_th) {
    octree_.reset(new octomap::ColorOcTree(resolution_));
    octree_->setProbHit(0.6);
    octree_->setProbMiss(0.49);
    octree_->setClampingThresMin(0.12);
    octree_->setClampingThresMax(0.97);
    octree_->setOccupancyThres(0.7);

    try {
        cameras_ = aslam::NCamera::loadFromYaml(
                        setting_path);
        LOG(INFO) << "Loaded the camera YAML file "
                   << setting_path;
    } catch (const std::exception& ex) {
        LOG(FATAL) << "Failed to open and parse the camera YAML file "
                    << setting_path
                    << " with the error: " << ex.what();
    }

    outlier_removal_.reset(new vins_core::RealTimeOutlierRemoval(
                               static_cast<float>(resolution_)));
}

OctoMapper::OctoMapper(const double resolution,
                       const int pick_step,
                       const aslam::NCamera::Ptr& cameras,
                       const int pt_in_grid_per_metre_th)
    : cameras_(cameras),
      resolution_(resolution),
      pick_step_(pick_step),
      pt_in_grid_per_metre_th_(pt_in_grid_per_metre_th) {
    octree_.reset(new octomap::ColorOcTree(resolution_));
    octree_->setProbHit(0.6);
    octree_->setProbMiss(0.49);
    octree_->setClampingThresMin(0.12);
    octree_->setClampingThresMax(0.97);
    octree_->setOccupancyThres(0.7);

    outlier_removal_.reset(new vins_core::RealTimeOutlierRemoval(
                               static_cast<float>(resolution_)));
}

octomap::OcTreeKey OctoMapper::CoordToKey(const octomap::point3d& pt) {
    return octree_->coordToKey(pt);
}

void OctoMapper::SetBbx(octomap::point3d& max,
                        octomap::point3d& min) {
    octree_->setBBXMax(max);
    octree_->setBBXMin(min);
}

void OctoMapper::CastRay(const octomap::point3d& origin,
                         const octomap::point3d& end,
                         octomap::KeySet* free_cells_ptr,
                         octomap::KeySet* occupied_cells_ptr) {
    octomap::KeySet& free_cells = *CHECK_NOTNULL(free_cells_ptr);
    octomap::KeySet& occupied_cells = *CHECK_NOTNULL(occupied_cells_ptr);

    if ((end - origin).norm() <= common::kMaxRangeScan) {
        octomap::KeyRay key_ray;
        if (octree_->computeRayKeys(origin, end, key_ray)) {
            free_cells.insert(key_ray.begin(), key_ray.end());
        }
        octomap::OcTreeKey octree_key;
        if (octree_->coordToKeyChecked(end, octree_key)) {
            occupied_cells.insert(octree_key);
        }
    } else {
        octomap::point3d new_end = origin + (end - origin).normalized() * common::kMaxRangeScan;
        octomap::KeyRay key_ray;
        if (octree_->computeRayKeys(origin, new_end, key_ray)) {
            free_cells.insert(key_ray.begin(), key_ray.end());
        }
    }
}

void OctoMapper::UpdateOccupancy(octomap::KeySet* free_cells_ptr,
                                 octomap::KeySet* occupied_cells_ptr) {
    octomap::KeySet& free_cells = *CHECK_NOTNULL(free_cells_ptr);
    octomap::KeySet& occupied_cells = *CHECK_NOTNULL(occupied_cells_ptr);

    octree_mutex_.lock();
    for (octomap::KeySet::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
        octree_->updateNode(*it, octree_->getProbHitLog());

        if (free_cells.find(*it) != free_cells.end()) {
            free_cells.erase(*it);
        }
    }

    for (octomap::KeySet::iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
        octree_->updateNode(*it, octree_->getProbMissLog());
    }
    octree_mutex_.unlock();
}

void OctoMapper::UpdateColor(const octomap::Pointcloud& pt,
                             const octomap::Pointcloud& color) {
    CHECK_EQ(pt.size(), color.size());
    octree_mutex_.lock();
    for (size_t i = 0u; i < pt.size(); ++i) {
        octree_->integrateNodeColor(pt[i].x(), pt[i].y(), pt[i].z(),
                                    static_cast<uchar>(color[i].x()),
                                    static_cast<uchar>(color[i].y()),
                                    static_cast<uchar>(color[i].z()));
    }
    octree_mutex_.unlock();
}

void OctoMapper::UpdateInnerOccupancy() {
    octree_mutex_.lock();
    octree_->updateInnerOccupancy();
    octree_mutex_.unlock();
}

void OctoMapper::DeleteNodes(const octomap::KeySet& free_cells,
                             const octomap::KeySet& occupied_cells) {
    octree_mutex_.lock();
    for (octomap::KeySet::const_iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
        octree_->deleteNode(*it);
    }

    for (octomap::KeySet::const_iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
        octree_->deleteNode(*it);
    }

    octree_->prune();

    octree_mutex_.unlock();
}

void OctoMapper::Clear() {
    octree_mutex_.lock();
    octree_->clear();
    octree_mutex_.unlock();
}

void OctoMapper::ResetOcTree() {
    octree_mutex_.lock();
    octree_.reset();
    octree_ = nullptr;
    octree_mutex_.unlock();
}

void OctoMapper::GetPointCloud(
        const common::ImageData& img_data,
        octomap::PointCloudXYZRGB* point_cloud_ptr) {
    octomap::PointCloudXYZRGB& point_cloud = *CHECK_NOTNULL(point_cloud_ptr);

    if (img_data.depth == nullptr) {
        return;
    }

    octomap::PointCloudXYZRGB point_cloud_inc_noise;
    for (int m = 0; m < img_data.depth->rows; m += pick_step_) {
        for (int n = 0; n < img_data.depth->cols; n += pick_step_) {
            const float d = static_cast<float>(
                        img_data.depth->at<unsigned short>(m, n)) / 1000.f;
            if (d < common::kMinRange || d > common::kMaxRangeScan) {
                continue;
            }

            Eigen::Vector2d key_point(n, m);
            Eigen::Vector3d bearing_3d_anchor;
            cameras_->getCamera(0).backProject3(key_point, &bearing_3d_anchor);
            const double b_z = bearing_3d_anchor(2);
            bearing_3d_anchor << bearing_3d_anchor(0) / b_z, bearing_3d_anchor(1) / b_z, 1.0;

            Eigen::Vector3d p = d * bearing_3d_anchor;

            float x = static_cast<float>(p(0));
            float y = static_cast<float>(p(1));
            float z = static_cast<float>(p(2));

            point_cloud_inc_noise.xyz.push_back(x, y, z);
            if (!img_data.images.empty()) {
                if (img_data.images[0]->type() == CV_8UC3) {
                    cv::Vec3b cv_color = img_data.images[0]->at<cv::Vec3b>(m, n);
                    point_cloud_inc_noise.rgb.push_back(static_cast<float>(cv_color[0]),
                                                        static_cast<float>(cv_color[1]),
                                                        static_cast<float>(cv_color[2]));
                } else if (img_data.images[0]->type() == CV_8UC1) {
                    uchar grey = img_data.images[0]->at<uchar>(m, n);
                    point_cloud_inc_noise.rgb.push_back(static_cast<float>(grey),
                                                        static_cast<float>(grey),
                                                        static_cast<float>(grey));
                } else {
                    LOG(FATAL) << "Unsupport image type.";
                }
            } else {
                point_cloud_inc_noise.rgb.push_back(0.f, 0.f, 0.f);
            }
        }
    }
#if 1
    common::TicToc timer;
    removal_mutex_.lock();
    outlier_removal_->SetInput(&point_cloud_inc_noise);
    constexpr bool kDoVoxelFiltering = true;
    point_cloud = outlier_removal_->Filter(pt_in_grid_per_metre_th_, kDoVoxelFiltering);
    removal_mutex_.unlock();
    VLOG(5) << "Point coud outlier filtering cost time (ms): " << timer.toc();
#else
    point_cloud = point_cloud_inc_noise;
#endif
}

void OctoMapper::Prune() {
    octree_mutex_.lock();
    octree_->prune();
    octree_mutex_.unlock();
}

std::shared_ptr<octomap::ColorOcTree> OctoMapper::GetOcTree() {
    return octree_;
}

std::shared_ptr<octomap::ColorOcTree> OctoMapper::GetOcTreeCopy() {
    std::shared_ptr<octomap::ColorOcTree> octree_copy = nullptr;
    if (octree_ != nullptr) {
        octree_mutex_.lock();
        octree_copy = std::make_shared<octomap::ColorOcTree>(*octree_);
        octree_mutex_.unlock();
    }
    return octree_copy;
}

void OctoMapper::SaveOctoMap(const std::string& path) {
    octree_mutex_.lock();
    octree_->write(path);
    octree_mutex_.unlock();
}

}
