#include "occ_common/castrayer.h"

#include <cstdlib>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "glog/logging.h"

#include "occ_common/xy_index.h"
#include "occ_common/probability_values.h"

namespace common {
namespace {

// Factor for subpixel accuracy of start and end point for ray casts.
constexpr int kSubpixelScale = 1000;
constexpr double kMaxMissInsertRange = 15.0;

void GrowAsNeeded(const common::PointCloud& range_data,
                    const Eigen::Vector3d& origin,
                    ProbabilityGrid* const probability_grid) {
    Eigen::AlignedBox2d bounding_box(origin.head<2>());
    // Padding around bounding box to avoid numerical issues at cell boundaries.
    constexpr double kPadding = 1e-6;
    for (const Eigen::Vector3d& hit : range_data.points) {
        bounding_box.extend(hit.head<2>());
    }
    for (const Eigen::Vector3d& miss : range_data.miss_points) {
        bounding_box.extend(miss.head<2>());
    }

    probability_grid->GrowLimits(bounding_box.min() -
                                 kPadding * Eigen::Vector2d::Ones());
    probability_grid->GrowLimits(bounding_box.max() +
                                 kPadding * Eigen::Vector2d::Ones());
}

bool IsEqual(const Eigen::Array2i& lhs, const Eigen::Array2i& rhs) {
    return ((lhs - rhs).matrix().lpNorm<1>() == 0);
}

#if 0
bool IsNotIn(const Eigen::Array2i& search, std::vector<Eigen::Array2i>& ends) {
    if (ends.size() <= 0u) {
        return true;
    }
    for (const Eigen::Array2i& it : ends) {
        if(IsEqual(it, search)) {
            return false;
        }
    }

    return true;
}
#endif

void CastRays(const common::PointCloud& range_data,
              const Eigen::Vector3d& origin,
              const std::vector<uint16>& hit_table,
              const std::vector<uint16>& miss_table,
              ProbabilityGrid* probability_grid) {
    GrowAsNeeded(range_data, origin, probability_grid);

    const MapLimits& limits = probability_grid->limits();
    const MapLimits superscaled_limits(
                limits.resolution() / kSubpixelScale, limits.max(),
                CellLimits(limits.cell_limits().num_x_cells * kSubpixelScale,
                           limits.cell_limits().num_y_cells * kSubpixelScale));
    const Eigen::Array2i begin =
            superscaled_limits.GetCellIndex(origin.head<2>());
    // Compute and add the end points.
    std::vector<Eigen::Array2i> ends;
    ends.reserve(range_data.points.size());

#if 1
    for (const Eigen::Vector3d& hit : range_data.points) {
        ends.push_back(superscaled_limits.GetCellIndex(hit.head<2>()));
        probability_grid->ApplyLookupTable(ends.back() / kSubpixelScale,
                                           hit_table);
    }
#else
    Eigen::Array2i tmp_hit_point;
    for (size_t i = 0u; i<range_data.points.size() - 2u; ++i) {
        tmp_hit_point = superscaled_limits.GetCellIndex(
                    range_data.points[i].head<2>());
        Eigen::Array2i next_hit_point = superscaled_limits.GetCellIndex(
                    range_data.points[i + 1].head<2>());

        const int row = std::abs(tmp_hit_point.x() - next_hit_point.x());
        const int col = std::abs(tmp_hit_point.y() - next_hit_point.y());

        if (row == 0 && col <= 10) {
            size_t next = i + 2;
            for (; next < range_data.points.size(); ++next) {
                const Eigen::Array2i hit_point = superscaled_limits.GetCellIndex(
                            range_data.points[next].head<2>());
                const int row_tmp = std::abs(tmp_hit_point.x() - hit_point.x());
                const int col_tmp = std::abs(tmp_hit_point.y() - hit_point.y());
                if (std::abs(hit_point.x() - next_hit_point.x()) > 2 ||
                        std::abs(col_tmp) >  row_tmp / 20.0 + 1) {
                    break;
                }
                next_hit_point = hit_point;
            }

            if (next == i + 2u) {
                if (IsNotIn(tmp_hit_point, ends)) {
                    ends.push_back(tmp_hit_point);
                }
                if (IsNotIn(next_hit_point, ends)) {
                    ends.push_back(next_hit_point);
                }
                i += 1u;
                continue;
            }

            const int row_tmp = next_hit_point.x() - tmp_hit_point.x();
            const int col_tmp = tmp_hit_point.y() - next_hit_point.y();

            if (std::abs(col_tmp) >= 5) {
                for (size_t add_idx = i; add_idx <= next && add_idx < range_data.points.size(); ++add_idx) {
                    const Eigen::Array2i hit_point = superscaled_limits.GetCellIndex(
                                range_data.points[add_idx].head<2>());
                    const Eigen::Array2i hit_point_cr(tmp_hit_point.x() + row_tmp / 2, hit_point.y());
                    if (IsNotIn(hit_point_cr, ends)) {
                        ends.push_back(hit_point_cr);
                    }
                }
                i = next;
            } else if (IsNotIn(tmp_hit_point, ends)) {
                ends.push_back(tmp_hit_point);
            }
        } else if (col == 0 && row <= 10) {
            size_t next = i + 2u;
            for (; next<range_data.points.size(); next++) {
                Eigen::Array2i hit_point = superscaled_limits.GetCellIndex(
                            range_data.points[next].head<2>());
                const int row_tmp = tmp_hit_point.x() - hit_point.x();
                const int col_tmp = std::abs(tmp_hit_point.y() - hit_point.y());

                if (std::abs(hit_point.y() - next_hit_point.y()) > 2 ||
                        std::abs(row_tmp) > col_tmp / 20.0 + 1) {
                    break;
                }

                next_hit_point = hit_point;
            }
            if (next == i + 2u) {
                if (IsNotIn(tmp_hit_point, ends)) {
                    ends.push_back(tmp_hit_point);
                }
                if (IsNotIn(next_hit_point, ends)) {
                    ends.push_back(next_hit_point);
                }
                i += 1u;
                continue;
            }

            const int row_tmp = std::abs(tmp_hit_point.x() - next_hit_point.x());
            const int col_tmp = next_hit_point.y()-tmp_hit_point.y();

            if (row_tmp >= 5) {
                for (size_t add_idx = i; add_idx <= next && add_idx < range_data.points.size(); ++add_idx) {
                    const Eigen::Array2i hit_point = superscaled_limits.GetCellIndex(
                                range_data.points[add_idx].head<2>());
                    const Eigen::Array2i hit_point_cr(hit_point.x(), tmp_hit_point.y() + col_tmp / 2);

                    if (IsNotIn(hit_point_cr, ends)) {
                        ends.push_back(hit_point_cr);
                    }
                }
                i = next;
            } else if (IsNotIn(tmp_hit_point, ends)) {
                ends.push_back(tmp_hit_point);
            }
        } else if (IsNotIn(tmp_hit_point, ends)) {
            ends.push_back(tmp_hit_point);
        }
    }

    for (const Eigen::Array2i& it : ends) {
        probability_grid->ApplyLookupTable(it / kSubpixelScale, hit_table);
    }
#endif

    // Now add the misses.
    for (const Eigen::Array2i& end : ends) {
        std::vector<Eigen::Array2i> ray =
                RayToPixelMask(begin, end, kSubpixelScale);
        if (IsEqual(begin / kSubpixelScale, ray.back())) {
            std::reverse(ray.begin(), ray.end());
        }
        for (size_t i = 0; i < ray.size() - 1; ++i) {
            if ((origin.head(2) - 
                limits.GetCellCenter(ray[i])).norm() > kMaxMissInsertRange) {
                break;
            }
            probability_grid->ApplyLookupTable(ray[i], miss_table);
        }
    }

    // Finally, compute and add empty rays based on misses in the range data.
    for (const Eigen::Vector3d& miss : range_data.miss_points) {
        std::vector<Eigen::Array2i> ray = RayToPixelMask(
            begin, superscaled_limits.GetCellIndex(miss.head<2>()),
            kSubpixelScale);
        for (const Eigen::Array2i& cell_index : ray) {
            probability_grid->ApplyLookupTable(cell_index, miss_table);
        }
    }
}
}  // namespace

CastRayer::CastRayer()
    : hit_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
                    Odds(hit_probability_))),
      miss_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
                    Odds(miss_probability_))) {}

void CastRayer::Insert(
        const common::PointCloud& range_data,
        const Eigen::Vector3d& origin,
        Grid2D* grid) const {
    ProbabilityGrid* const probability_grid =
            static_cast<ProbabilityGrid*>(grid);
    CHECK_NOTNULL(probability_grid);
    // By not finishing the update after hits are inserted, we give hits
    // priority (i.e. no hits will be ignored because of a miss
    // in the same cell).
    CastRays(range_data,
             origin,
             hit_table_,
             miss_table_,
             probability_grid);
    probability_grid->FinishUpdate();
}

void CastRayer::SetHitMissProbability(
        double hit_probability, double miss_probability) {
    hit_probability_ = hit_probability;
    miss_probability_ = miss_probability;
}

}
