#include "occ_common/grid_2d.h"

namespace common {

namespace {

bool IsEqual(const Eigen::Array2i& lhs, const Eigen::Array2i& rhs) {
    return ((lhs - rhs).matrix().lpNorm<1>() == 0);
}
}  // namespace

// Compute all pixels that contain some part of the line segment connecting
// 'scaled_begin' and 'scaled_end'. 'scaled_begin' and 'scaled_end' are scaled
// by 'subpixel_scale'. 'scaled_begin' and 'scaled_end' are expected to be
// greater than zero. Return values are in pixels and not scaled.
std::vector<Eigen::Array2i> RayToPixelMask(
        const Eigen::Array2i& scaled_begin,
        const Eigen::Array2i& scaled_end,
        int subpixel_scale) {
    // For simplicity, we order 'scaled_begin' and 'scaled_end' by their x
    // coordinate.
    if (scaled_begin.x() > scaled_end.x()) {
        return RayToPixelMask(scaled_end, scaled_begin, subpixel_scale);
    }

    CHECK_GE(scaled_begin.x(), 0);
    CHECK_GE(scaled_begin.y(), 0);
    CHECK_GE(scaled_end.y(), 0);
    std::vector<Eigen::Array2i> pixel_mask;
    // Special case: We have to draw a vertical line in full pixels, as
    // 'scaled_begin' and 'scaled_end' have the same full pixel x coordinate.
    if (scaled_begin.x() / subpixel_scale == scaled_end.x() / subpixel_scale) {
        Eigen::Array2i current(
                    scaled_begin.x() / subpixel_scale,
                    std::min(scaled_begin.y(),
                             scaled_end.y()) / subpixel_scale);
        pixel_mask.push_back(current);
        const int end_y =
                std::max(scaled_begin.y(), scaled_end.y()) / subpixel_scale;
        for (; current.y() <= end_y; ++current.y()) {
            if (!IsEqual(pixel_mask.back(), current)) {
                pixel_mask.push_back(current);
            }
        }
        return pixel_mask;
    }

    const int64 dx = scaled_end.x() - scaled_begin.x();
    const int64 dy = scaled_end.y() - scaled_begin.y();
    const int64 denominator = 2 * subpixel_scale * dx;

    // The current full pixel coordinates. We scaled_begin at 'scaled_begin'.
    Eigen::Array2i current = scaled_begin / subpixel_scale;
    pixel_mask.push_back(current);

    // To represent subpixel centers, we use a factor of 2 * 'subpixel_scale' in
    // the denominator.
    // +-+-+-+ -- 1 = (2 * subpixel_scale) / (2 * subpixel_scale)
    // | | | |
    // +-+-+-+
    // | | | |
    // +-+-+-+ -- top edge of first subpixel = 2 / (2 * subpixel_scale)
    // | | | | -- center of first subpixel = 1 / (2 * subpixel_scale)
    // +-+-+-+ -- 0 = 0 / (2 * subpixel_scale)

    // The center of the subpixel part of 'scaled_begin.y()' assuming the
    // 'denominator', i.e., sub_y / denominator is in (0, 1).
    int64 sub_y = (2 * (scaled_begin.y() % subpixel_scale) + 1) * dx;

    // The distance from the from 'scaled_begin' to the right pixel border,
    // to be divided by 2 * 'subpixel_scale'.
    const int first_pixel =
            2 * subpixel_scale - 2 * (scaled_begin.x() % subpixel_scale) - 1;
    // The same from the left pixel border to 'scaled_end'.
    const int last_pixel = 2 * (scaled_end.x() % subpixel_scale) + 1;

    // The full pixel x coordinate of 'scaled_end'.
    const int end_x = std::max(scaled_begin.x(),
                               scaled_end.x()) / subpixel_scale;

    // Move from 'scaled_begin' to the next pixel border to the right.
    sub_y += dy * first_pixel;
    if (dy > 0) {
        while (true) {
            if (!IsEqual(pixel_mask.back(), current)) {
                pixel_mask.push_back(current);
            }
            while (sub_y > denominator) {
                sub_y -= denominator;
                ++current.y();
                if (!IsEqual(pixel_mask.back(), current)) {
                    pixel_mask.push_back(current);
                }
            }
            ++current.x();
            if (sub_y == denominator) {
                sub_y -= denominator;
                ++current.y();
            }
            if (current.x() == end_x) {
                break;
            }
            // Move from one pixel border to the next.
            sub_y += dy * 2 * subpixel_scale;
        }
        // Move from the pixel border on the right to 'scaled_end'.
        sub_y += dy * last_pixel;
        if (!IsEqual(pixel_mask.back(), current)) {
            pixel_mask.push_back(current);
        }
        while (sub_y > denominator) {
            sub_y -= denominator;
            ++current.y();
            if (!IsEqual(pixel_mask.back(), current)) {
                pixel_mask.push_back(current);
            }
        }
        CHECK_NE(sub_y, denominator);
        CHECK_EQ(current.y(), scaled_end.y() / subpixel_scale);
        return pixel_mask;
    }

    // Same for lines non-ascending in y coordinates.
    while (true) {
        if (!IsEqual(pixel_mask.back(), current)) {
            pixel_mask.push_back(current);
        }
        while (sub_y < 0) {
            sub_y += denominator;
            --current.y();
            if (!IsEqual(pixel_mask.back(), current)) {
                pixel_mask.push_back(current);
            }
        }
        ++current.x();
        if (sub_y == 0) {
            sub_y += denominator;
            --current.y();
        }
        if (current.x() == end_x) {
            break;
        }
        sub_y += dy * 2 * subpixel_scale;
    }
    sub_y += dy * last_pixel;
    if (!IsEqual(pixel_mask.back(), current)) {
        pixel_mask.push_back(current);
    }
    while (sub_y < 0) {
        sub_y += denominator;
        --current.y();
        if (!IsEqual(pixel_mask.back(), current)) {
            pixel_mask.push_back(current);
        }
    }
    CHECK_NE(sub_y, 0);
    CHECK_EQ(current.y(), scaled_end.y() / subpixel_scale);
    return pixel_mask;
}

const std::vector<float>* ValueConversionTables::GetConversionTable(
        float unknown_result, float lower_bound, float upper_bound) {
    std::tuple<float, float, float> bounds =
            std::make_tuple(unknown_result, lower_bound, upper_bound);
    auto lookup_table_iterator = bounds_to_lookup_table_.find(bounds);
    if (lookup_table_iterator == bounds_to_lookup_table_.end()) {
        auto insertion_result = bounds_to_lookup_table_.emplace(
                    bounds, PrecomputeValueToBoundedFloat(
                        0,
                        unknown_result,
                        lower_bound,
                        upper_bound));
        return insertion_result.first->second.get();
    } else {
        return lookup_table_iterator->second.get();
    }
}

Grid2D::Grid2D(
        int id,
        const MapLimits& limits,
        float min_correspondence_cost,
        float max_correspondence_cost,
        ValueConversionTables* conversion_tables)
    : id_(id),
      limits_(limits),
      correspondence_cost_cells_(
          limits_.cell_limits().num_x_cells * limits_.cell_limits().num_y_cells,
          kUnknownCorrespondenceValue),
      min_correspondence_cost_(min_correspondence_cost),
      max_correspondence_cost_(max_correspondence_cost),
      value_to_correspondence_cost_table_(
          conversion_tables->GetConversionTable(
              max_correspondence_cost,
              min_correspondence_cost,
              max_correspondence_cost)) {
    CHECK_LT(min_correspondence_cost_, max_correspondence_cost_);
}

// Finishes the update sequence.
void Grid2D::FinishUpdate() {
    while (!update_indices_.empty()) {
        DCHECK_GE(correspondence_cost_cells_[update_indices_.back()],
                kUpdateMarker);
        correspondence_cost_cells_[update_indices_.back()] -= kUpdateMarker;
        update_indices_.pop_back();
    }
}

// Fills in 'offset' and 'limits' to define a subregion of that contains all
// known cells.
void Grid2D::ComputeCroppedLimits(Eigen::Array2i* const offset,
                                  CellLimits* const limits) const {
    if (known_cells_box_.isEmpty()) {
        *offset = Eigen::Array2i::Zero();
        *limits = CellLimits(1, 1);
        return;
    }
    *offset = known_cells_box_.min().array();
    *limits = CellLimits(known_cells_box_.sizes().x() + 1,
                         known_cells_box_.sizes().y() + 1);
}

// Grows the map as necessary to include 'point'. This changes the meaning of
// these coordinates going forward. This method must be called immediately
// after 'FinishUpdate', before any calls to 'ApplyLookupTable'.
void Grid2D::GrowLimits(const Eigen::Vector2d& point) {
    GrowLimits(point, {mutable_correspondence_cost_cells()},
    {kUnknownCorrespondenceValue});
}

void Grid2D::GrowLimits(const Eigen::Vector2d& point,
                        const std::vector<std::vector<uint16>*>& grids,
                        const std::vector<uint16>& grids_unknown_cell_values) {
    CHECK(update_indices_.empty());
    while (!limits_.Contains(limits_.GetCellIndex(point))) {
        const int x_offset = limits_.cell_limits().num_x_cells / 2;
        const int y_offset = limits_.cell_limits().num_y_cells / 2;
        const MapLimits new_limits(
                    limits_.resolution(),
                    limits_.max() +
                    limits_.resolution() * Eigen::Vector2d(y_offset, x_offset),
                    CellLimits(2 * limits_.cell_limits().num_x_cells,
                               2 * limits_.cell_limits().num_y_cells));
        const int stride = new_limits.cell_limits().num_x_cells;
        const int offset = x_offset + stride * y_offset;
        const int new_size = new_limits.cell_limits().num_x_cells *
                new_limits.cell_limits().num_y_cells;

        for (size_t grid_index = 0u; grid_index < grids.size(); ++grid_index) {
            std::vector<uint16> new_cells(
                        new_size,
                        grids_unknown_cell_values[grid_index]);
            for (int i = 0; i < limits_.cell_limits().num_y_cells; ++i) {
                for (int j = 0; j < limits_.cell_limits().num_x_cells; ++j) {
                    new_cells[offset + j + i * stride] =
                            (*grids[grid_index])[j +
                            i * limits_.cell_limits().num_x_cells];
                }
            }
            *grids[grid_index] = new_cells;
        }
        limits_ = new_limits;
        if (!known_cells_box_.isEmpty()) {
            known_cells_box_.translate(Eigen::Vector2i(x_offset, y_offset));
        }
    }
}

}
