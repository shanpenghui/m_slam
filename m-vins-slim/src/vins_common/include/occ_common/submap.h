#ifndef OCCUPANCY_GRID_SUBMAP_H_
#define OCCUPANCY_GRID_SUBMAP_H_

#include <mutex>
#include <set>

#include "aslam/common/pose-types.h"

namespace common {
// An individual submap, which has a 'local_pose' in the local map frame, keeps
// track of how many range data were inserted into it, and sets
// 'insertion_finished' when the map no longer changes and is ready for loop
// closing.
class Submap {
 public:
    explicit Submap(const aslam::Transformation& local_submap_pose)
        : local_pose_(local_submap_pose) {}
    virtual ~Submap() {}

    // Pose of this submap in the local map frame.
    aslam::Transformation local_pose() const {
        return local_pose_;
    }

    void update_local_pose(const aslam::Transformation& local_pose) {
        local_pose_ = local_pose;
    }

    // Number of RangeData inserted.
    int num_range_data() const {
        return num_range_data_;
    }
    void set_num_range_data(const int num_range_data) {
        num_range_data_ = num_range_data;
    }

    bool insertion_finished() const {
        return insertion_finished_;
    }
    void set_insertion_finished(bool insertion_finished) {
        insertion_finished_ = insertion_finished;
    }

    bool HasScan(const int& scanid) {
        return scanid_in_submap_.count(scanid);
    }

    const std::set<int, std::less<int> >& scanid_in_submap() const {
        return scanid_in_submap_;
    }
    int submap_id_;

    std::mutex finish_mutex_;

 protected:
    std::set<int, std::less<int>> scanid_in_submap_;

 private:
    aslam::Transformation local_pose_;
    int num_range_data_ = 0;
    bool insertion_finished_ = false;
};
}

#endif
