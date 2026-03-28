#include "interface/interface.h"

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>

namespace mvins {

void GammaTransform(const cv::Mat& src,
                    const float gamma,
                    cv::Mat* dst_ptr) {
    cv::Mat& dst = *CHECK_NOTNULL(dst_ptr);

    uchar bin[256];
    for (int i = 0; i < 256; ++i) {
        bin[i] = static_cast<uchar>(std::pow(static_cast<float>(i) / 255.f, gamma) * 255.f);
    }
    
    dst = src.clone();
    cv::MatIterator_<uchar> it, end;
    for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it) {
        *it = bin[*it];
    }
}

//! Add image data from ROS messages.
void Interface::FillImageMsg(const ImageMsgPtr msg) {
    FillImageMsgImpl(msg);
}
void Interface::FillImageMsgImpl(const ImageMsgPtr msg) {
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        LOG(ERROR) << "cv_bridge exception: " << e.what();
        return;
    }

    // Need copy operation for images from rosbag.
    const cv::Mat image = cv_ptr->image.clone();
    
    RosTime t_msg = msg->header.stamp;
    uint64_t timestamp = t_msg.nanoseconds();

    common::CvMatConstPtrVec images;
    if (image.type() == CV_8UC1) {
        // Pre-process for IR gray image.
        cv::Mat image_gamma, image_clache;
        GammaTransform(image, 0.5f, &image_gamma);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4.0);
        clahe->setTilesGridSize(cv::Size(10, 10));
        clahe->apply(image_gamma, image_clache);
        images.push_back(std::make_shared<const cv::Mat>(image_clache));
    } else {
        images.push_back(std::make_shared<const cv::Mat>(image));
    }

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        double td_s;
        vins_handler_ptr_->GetNewTdCamera(&td_s);
        uint64_t td_ns = common::SecondsToNanoSeconds(std::abs(td_s));
        uint64_t timestamp_td_ns;
        if (td_s > 0) {
            timestamp_td_ns = timestamp + td_ns;
        } else {
            timestamp_td_ns = timestamp - td_ns;
        }
        vins_handler_ptr_->AddImage(common::ImageData(timestamp_td_ns, images));
    }
}

void Interface::FillDepthMsg(const ImageMsgPtr depth_msg) {
    FillDepthMsgImpl(depth_msg);
}

void Interface::FillDepthMsgImpl(const ImageMsgPtr depth_msg) {
    cv_bridge::CvImageConstPtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvShare(depth_msg);
    } catch (cv_bridge::Exception& e) {
        LOG(ERROR) << "cv_bridge exception: " << e.what();
        return;
    }
    const cv::Mat depth = cv_ptr->image.clone();
    // CHECK_EQ(depth.type(), CV_16UC1);
    RosTime t_msg = depth_msg->header.stamp;
    uint64_t timestamp = t_msg.nanoseconds();

    common::CvMatConstPtr depth_ptr = std::make_shared<const cv::Mat>(depth);

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);   
    if (vins_handler_ptr_ != nullptr) {
        double td_s;
        vins_handler_ptr_->GetNewTdCamera(&td_s);
        uint64_t td_ns = common::SecondsToNanoSeconds(std::abs(td_s));
        uint64_t timestamp_td_ns;
        if (td_s > 0) {
            timestamp_td_ns = timestamp + td_ns;
        } else {
            timestamp_td_ns = timestamp - td_ns;
        }
        common::DepthData depth_data(timestamp_td_ns, depth_ptr);
        vins_handler_ptr_->AddDepth(depth_data);
    } 
}
}
