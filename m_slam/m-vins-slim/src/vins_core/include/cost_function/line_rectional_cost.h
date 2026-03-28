#ifndef MVINS_COST_FUNCTION_LINE_RECTIONAL_COST_H_
#define MVINS_COST_FUNCTION_LINE_RECTIONAL_COST_H_

#include <ceres/ceres.h>

#include <opencv2/opencv.hpp>

namespace vins_core {

ceres::CostFunction* CreateLineRectionalCost(
    const std::vector<cv::Vec4i>& lines);
}

#endif
