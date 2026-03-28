#ifndef VINS_HANDLER_MAP_TOOL_H_
#define VINS_HANDLER_MAP_TOOL_H_

#include "data_common/state_structures.h"
#include "file_common/file_system_tools.h"
#include "occ_common/map_limits.h"

namespace vins_handler {

void MapPoseRectangulate(const cv::Mat& raw_map,
                         common::KeyFrames* key_frames_ptr,
                         std::vector<cv::Vec4i>* lines_ptr,
                         std::vector<int>* inlier_indices_ptr);

cv::Mat CreateShowMapMat(const cv::Mat& raw_map, 
                         const bool do_remove_obstacles,
                         const int contour_length);

void RemoveSmallObstacles(cv::Mat* input_map_ptr, const int contour_length);
}

#endif
