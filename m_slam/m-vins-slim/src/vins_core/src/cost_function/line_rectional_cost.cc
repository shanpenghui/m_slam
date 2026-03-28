#include "cost_function/line_rectional_cost.h"

namespace vins_core {

class LineRectionalCost {
 public:
    LineRectionalCost(const std::vector<cv::Vec4i>& lines)
     : lines_(lines) {}

    template <typename T>
    bool operator()(const T* const angle, T* residual) const {
        T cos_angle = cos(*angle);
        T sin_angle = sin(*angle);
        T total_length = T(0);
        T total_deviation = T(0);
        for (const auto& line : lines_) {
            T x1 = T(line[0]);
            T y1 = T(line[1]);
            T x2 = T(line[2]);
            T y2 = T(line[3]);

            T x1r = x1 * cos_angle + y1 * sin_angle;
            T y1r = -x1 * sin_angle + y1 * cos_angle;
            T x2r = x2 * cos_angle + y2 * sin_angle;
            T y2r = -x2 * sin_angle + y2 * cos_angle;

            T dx = x2r - x1r;
            T dy = y2r - y1r;
            T length = sqrt(dx * dx + dy * dy);
            T angle = atan2(dy, dx);

            T angle_degrees = angle * T(180.0 / M_PI);
            T tmp = angle_degrees / T(90.0);
            T angle_rounded = (tmp > T(0) ? ceres::floor(tmp + T(0.5)) : ceres::ceil(tmp - T(0.5))) * T(90.0);
            T angle_error = angle_rounded - angle_degrees;
            if (angle_error > T(45.0)) {
                angle_error -= T(90.0);
            } else if (angle_error < T(-45.0)) {
                angle_error += T(90.0);
            }

            total_length += length;
            total_deviation += abs(angle_error) * length;
        }

        residual[0] = total_length - total_deviation;
        return true;
    }

private:
    const std::vector<cv::Vec4i>& lines_;
};

ceres::CostFunction* CreateLineRectionalCost(
        const std::vector<cv::Vec4i>& lines) {
    return new ceres::AutoDiffCostFunction<LineRectionalCost, 1, 1>(
        new LineRectionalCost(lines));
}

}
