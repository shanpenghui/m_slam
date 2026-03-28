#ifndef ASLAM_CV_DETECTORS_LSD
#define ASLAM_CV_DETECTORS_LSD

#include <memory>

#include <aslam/common/macros.h>
#include <Eigen/Core>
#include <lsd/lsd-opencv.h>

#include "aslam/detectors/line.h"

namespace aslam {

class LineSegmentDetector {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Options {
        size_t min_segment_length_px;
        Options() :
            min_segment_length_px(20u) {}
    };

    // Use aslam default parameters for line segment detection.
    LineSegmentDetector(const Options& options);

    // Use custom parameters for line segment detection.
    // See lsd-opencv.cc : LineSegmentDetectorImpl for details.
    LineSegmentDetector(const Options& options,
                        const int refine,
                        const double scale,
                        const double sigma_scale,
                        const double quant,
                        const double ang_th,
                        const double log_eps,
                        const double density_th,
                        const int n_bins);

    ~LineSegmentDetector();

    void detect(const cv::Mat& image, Lines* lines);

    /// Draw a list of lines onto a color(!) image.
    void drawLines(const Lines& lines, cv::Mat* image);

 private:
    cv::Ptr<aslamcv::LineSegmentDetector> line_detector_;
    const Options options_;
};

}  // namespace aslam

#endif  // ASLAM_CV_DETECTORS_LSD
