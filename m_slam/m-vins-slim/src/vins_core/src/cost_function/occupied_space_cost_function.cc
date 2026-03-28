#include "cost_function/occupied_space_cost_function.h"

#include <ceres/cubic_interpolation.h>

namespace vins_core {
class GridArrayAdapter {
 public:
    enum { DATA_DIMENSION = 1 };
    explicit GridArrayAdapter(const common::Grid2D& grid) : grid_(grid) {
    }
    void GetValue(const int row,
                  const int column,
                  double* const value) const {
        if (row < 0 || column < 0 ||
                row >= NumRows() ||
                column >= NumCols()) {
            *value = common::kMaxCorrespondenceCost;
        } else {
            *value = static_cast<double>(
                        grid_.GetCorrespondenceCost(
                            Eigen::Array2i(column, row)));
        }
    }
    int NumRows() const {
        return grid_.limits().cell_limits().num_y_cells;
    }
    int NumCols() const {
        return grid_.limits().cell_limits().num_x_cells;
    }
 private:
    const common::Grid2D& grid_;
};

class LocalOccupiedSpace2D {
public:
 LocalOccupiedSpace2D(const common::PointCloud& point_cloud,
                       const common::Grid2D& grid,
                       const double scan_sigma)
     : point_cloud_(point_cloud),
       grid_(grid),
       sqrt_info_(1. / scan_sigma) {}

 LocalOccupiedSpace2D(const LocalOccupiedSpace2D&) = delete;
 LocalOccupiedSpace2D& operator=(const LocalOccupiedSpace2D&) =
     delete;

 template <typename T>
 bool operator()(const T* const T_OtoG, T* const residual) const {
     Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_OinG(T_OtoG);
     Eigen::Map<const Eigen::Quaternion<T>> q_OtoG(T_OtoG + 3);
     const Eigen::Matrix<T, 3, 3> R_OtoG = q_OtoG.toRotationMatrix();

     const GridArrayAdapter adapter(grid_);
     ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);
     const common::MapLimits& limits = grid_.limits();

     for (size_t i = 0u; i < point_cloud_.points.size(); ++i) {
         const Eigen::Matrix<double, 3, 1> p_LinO = point_cloud_.points[i];
         const Eigen::Matrix<T, 3, 1> p_LinG =
                 R_OtoG * p_LinO.template cast<T>() + p_OinG;

         const T x = (limits.max().x() - p_LinG[0]) /
                 limits.resolution() - 0.5;
         const T y = (limits.max().y() - p_LinG[1]) /
                 limits.resolution() - 0.5;
         interpolator.Evaluate(x, y, &residual[i]);

         residual[i] = T(sqrt_info_) * residual[i];
         CHECK(!ceres::IsNaN(residual[i]));
     }
     return true;
 }
private:
  const common::PointCloud& point_cloud_;
  const common::Grid2D& grid_;
  const double sqrt_info_;
};


class GlobalOccupiedSpace2D {
public:
 GlobalOccupiedSpace2D(const common::PointCloud& point_cloud,
                        const common::Grid2D& grid,
                        const double scan_sigma)
     : point_cloud_(point_cloud),
       grid_(grid),
       sqrt_info_(1. / scan_sigma) {}

 GlobalOccupiedSpace2D(const GlobalOccupiedSpace2D&) = delete;
 GlobalOccupiedSpace2D& operator=(const GlobalOccupiedSpace2D&) =
     delete;

 template <typename T>
 bool operator()(const T* const T_OtoG, const T* const T_GtoM, T* const residual) const {
     Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_OinG(T_OtoG);
     Eigen::Map<const Eigen::Quaternion<T>> q_OtoG(T_OtoG + 3);
     Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_GinM(T_GtoM);
     Eigen::Map<const Eigen::Quaternion<T>> q_GtoM(T_GtoM + 3);
     const Eigen::Matrix<T, 3, 3> R_OtoG = q_OtoG.toRotationMatrix();
     const Eigen::Matrix<T, 3, 3> R_GtoM = q_GtoM.toRotationMatrix();

     const GridArrayAdapter adapter(grid_);
     ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);
     const common::MapLimits& limits = grid_.limits();

     for (size_t i = 0u; i < point_cloud_.points.size(); ++i) {
         const Eigen::Matrix<double, 3, 1> p_LinO = point_cloud_.points[i];
         const Eigen::Matrix<T, 3, 1> p_LinG =
                 R_OtoG * p_LinO.template cast<T>() + p_OinG;
         const Eigen::Matrix<T, 3, 1> p_LinM =
                 R_GtoM * p_LinG + p_GinM;

         const T x = (limits.max().x() - p_LinM[0]) /
                 limits.resolution() - 0.5;
         const T y = (limits.max().y() - p_LinM[1]) /
                 limits.resolution() - 0.5;
         interpolator.Evaluate(x, y, &residual[i]);

         residual[i] = T(sqrt_info_) * residual[i];
         CHECK(!ceres::IsNaN(residual[i]));
     }
     return true;
 }
private:
  const common::PointCloud& point_cloud_;
  const common::Grid2D& grid_;
  const double sqrt_info_;
};

ceres::CostFunction* CreateLocalOccupiedSpace2D(
        const common::PointCloud& point_cloud,
        const common::Grid2D& grid,
        const double scan_sigma) {
    return new ceres::AutoDiffCostFunction<
        LocalOccupiedSpace2D,
        ceres::DYNAMIC /*residuals*/, common::kGlobalPoseSize /*paramter block*/>(
        new LocalOccupiedSpace2D(point_cloud, grid, scan_sigma),
        point_cloud.points.size());
}

ceres::CostFunction* CreateGlobalOccupiedSpace2D(
        const common::PointCloud& point_cloud,
        const common::Grid2D& grid,
        const double scan_sigma) {
    return new ceres::AutoDiffCostFunction<
        GlobalOccupiedSpace2D,
        ceres::DYNAMIC /*residuals*/, common::kGlobalPoseSize /*paramter block*/,
            common::kGlobalPoseSize /*paramter block*/>(
        new GlobalOccupiedSpace2D(point_cloud, grid, scan_sigma),
        point_cloud.points.size());
}

}
