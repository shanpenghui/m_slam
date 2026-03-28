#include "ros_handler/rosbag_parsing_handler.h"

#include "interface/interface.h"

namespace mvins {
RosbagParser::RosbagParser(const std::string& data_directory,
                           const double data_realtime_playback_rate,
                           const TopicMap& ros_topics,
                           Interface* interface_ptr)
    : interface_ptr_(CHECK_NOTNULL(interface_ptr)),
      ros_topics_(ros_topics),
      data_realtime_playback_rate_(data_realtime_playback_rate) {
    CHECK(!data_directory.empty());
#ifdef USE_ROS2
    rosbag2_cpp::StorageOptions storage_options;
    storage_options.uri = data_directory;
    storage_options.storage_id = "sqlite3";

    rosbag2_cpp::ConverterOptions converter_converter;
    converter_converter.input_serialization_format = "cdr";
    converter_converter.output_serialization_format = "cdr";

    auto sequential_reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
    reader_ptr_ = std::make_unique<rosbag2_cpp::Reader>(std::move(sequential_reader));
    reader_ptr_->open(storage_options, converter_converter);

    factory_ptr_ = std::make_unique<rosbag2_cpp::SerializationFormatConverterFactory>();
    cdr_deserializer_ptr_ = factory_ptr_->load_deserializer("cdr");
#else
    try {
        bag_.reset(new rosbag::Bag);
        bag_->open(data_directory, rosbag::bagmode::Read);
        VLOG(0) << "Get rosbag in " << data_directory;
    } catch (const std::exception& ex) {
        LOG(ERROR) << "Could not open the rosbag " << data_directory;
    }

    std::vector<std::string> all_topics;
    for (const auto& name_topic_pair : ros_topics_) {
        all_topics.push_back(name_topic_pair.second);
    }
    VLOG(0) << "Loading rosbag...";
    bag_view_.reset(new rosbag::View(*bag_,
                                     rosbag::TopicQuery(all_topics)));
#endif
}

RosbagParser::~RosbagParser() {
    if (streaming_thread_ptr_ != nullptr &&
        streaming_thread_ptr_->joinable()) {
        streaming_thread_ptr_->join();
    }
    VLOG(0) << "Shutdown rosbag parser.";
}

void RosbagParser::Start() {
    streaming_thread_ptr_.reset(
        new std::thread(&RosbagParser::RunThread, this));
}

void RosbagParser::ControlDataPlayRate(
    const TimePoint& time_start,
    const double timestamp_curr_frame,
    const double timestamp_next_frame) {
    std::chrono::duration<double> time_between_frame(
        timestamp_next_frame - timestamp_curr_frame);

    CHECK_GT(data_realtime_playback_rate_,
        std::numeric_limits<double>::epsilon())
        << "Negative play rate is not allowed!";

    time_between_frame *= 1.0 / data_realtime_playback_rate_;

    std::this_thread::sleep_until(time_start + time_between_frame);
}

#ifdef USE_ROS2
template <typename MessageType>
MessageType RosbagParser::ParseTopic(
    const SerializedBagMessagePtr& serialized_message,
    const rosbag2_cpp::ConverterTypeSupport& type_support) {
    MessageType msg;
    auto ros_message = std::make_shared<rosbag2_cpp::rosbag2_introspection_message_t>();
    ros_message->time_stamp = serialized_message->time_stamp;
    ros_message->allocator = rcutils_get_default_allocator();
    ros_message->message = &msg;
    try {
        cdr_deserializer_ptr_->deserialize(
            serialized_message, type_support.rmw_type_support, ros_message);
    } catch (std::exception &e) {
        LOG(ERROR) << e.what();
    }

    return msg;
}
#endif

void RosbagParser::RunThread() {
    VLOG(0) << "Start rosbag parsing ...";

    size_t message_count = 0u;
    bool first_mssage = true;
    TimePoint last_pub_finish_time_s;

#ifdef USE_ROS2

    rcutils_time_point_value_t last_message_time_ns = 0;
    rosbag2_cpp::ConverterTypeSupport type_support_img;
    type_support_img.type_support_library =
        rosbag2_cpp::get_typesupport_library("sensor_msgs/msg/Image", "rosidl_typesupport_cpp");
    type_support_img.rmw_type_support = rosbag2_cpp::get_typesupport_handle(
        "sensor_msgs/msg/Image", "rosidl_typesupport_cpp", type_support_img.type_support_library);
    
    rosbag2_cpp::ConverterTypeSupport type_support_odom;
    type_support_odom.type_support_library =
        rosbag2_cpp::get_typesupport_library("nav_msgs/msg/Odometry", "rosidl_typesupport_cpp");
    type_support_odom.rmw_type_support = rosbag2_cpp::get_typesupport_handle(
        "nav_msgs/msg/Odometry", "rosidl_typesupport_cpp", type_support_odom.type_support_library);
    
    rosbag2_cpp::ConverterTypeSupport type_support_imu;
    type_support_imu.type_support_library =
        rosbag2_cpp::get_typesupport_library("sensor_msgs/msg/Imu", "rosidl_typesupport_cpp");
    type_support_imu.rmw_type_support = rosbag2_cpp::get_typesupport_handle(
        "sensor_msgs/msg/Imu", "rosidl_typesupport_cpp", type_support_imu.type_support_library);
    
    rosbag2_cpp::ConverterTypeSupport type_support_scan;
    type_support_scan.type_support_library =
        rosbag2_cpp::get_typesupport_library("sensor_msgs/msg/LaserScan", "rosidl_typesupport_cpp");
    type_support_scan.rmw_type_support = rosbag2_cpp::get_typesupport_handle(
        "sensor_msgs/msg/LaserScan", "rosidl_typesupport_cpp", type_support_scan.type_support_library);

    rosbag2_cpp::ConverterTypeSupport type_support_scan_pc2;
    type_support_scan_pc2.type_support_library =
        rosbag2_cpp::get_typesupport_library("sensor_msgs/msg/PointCloud2", "rosidl_typesupport_cpp");
    type_support_scan_pc2.rmw_type_support = rosbag2_cpp::get_typesupport_handle(
        "sensor_msgs/msg/PointCloud2", "rosidl_typesupport_cpp", type_support_scan_pc2.type_support_library);

    while(reader_ptr_->has_next()) {
        if (interface_ptr_->IsDataFinished()) {
            break;
        }
        SerializedBagMessagePtr serialized_message = reader_ptr_->read_next();

        if (first_mssage) {
            first_mssage = false;
        } else {
#if 1
            rcutils_time_point_value_t current_message_time_ns =
                serialized_message->time_stamp;
            ControlDataPlayRate(last_pub_finish_time_s,
                                common::NanoSecondsToSeconds(last_message_time_ns),
                                common::NanoSecondsToSeconds(current_message_time_ns));
#endif
        }

        if (serialized_message->topic_name == ros_topics_.at(CameraTopicName)) {
            ImageMsg image_msg = ParseTopic<ImageMsg>(serialized_message,
                                                      type_support_img);
            interface_ptr_->FillImageMsg(std::make_shared<ImageMsg>(image_msg));
        } else if (serialized_message->topic_name == ros_topics_.at(CameraDepthTopicName)) {
            ImageMsg depth_msg = ParseTopic<ImageMsg>(serialized_message,
                                                      type_support_img);
            interface_ptr_->FillDepthMsg(std::make_shared<ImageMsg>(depth_msg));
        } else if (serialized_message->topic_name == ros_topics_.at(OdomTopicName)) {
            OdometryMsg odom_msg = ParseTopic<OdometryMsg>(serialized_message,
                                                           type_support_odom);
            interface_ptr_->FillOdomMsg(std::make_shared<OdometryMsg>(odom_msg));
        } else if (serialized_message->topic_name == ros_topics_.at(ImuTopicName)) {
            ImuMsg imu_msg = ParseTopic<ImuMsg>(serialized_message,
                                                type_support_imu);
            interface_ptr_->FillImuMsg(std::make_shared<ImuMsg>(imu_msg));           
        } else if (serialized_message->topic_name == ros_topics_.at(ScanTopicName)) {
            LaserScanMsg scan_msg = ParseTopic<LaserScanMsg>(serialized_message,
                                                             type_support_scan);
            interface_ptr_->FillScanMsg(std::make_shared<LaserScanMsg>(scan_msg));            
        } else if (serialized_message->topic_name == ros_topics_.at(ScanCloudTopicName)) {
            PointCloudMsg scan_pc2_msg = ParseTopic<PointCloudMsg>(serialized_message,
                                                                   type_support_scan_pc2);
            interface_ptr_->FillScanPc2Msg(std::make_shared<PointCloudMsg>(scan_pc2_msg));
        } else if (serialized_message->topic_name == ros_topics_.at(GroundTruthTopicName)) {
            OdometryMsg gt_msg = ParseTopic<OdometryMsg>(serialized_message,
                                                         type_support_odom);
            interface_ptr_->FillGroundTruthMsg(std::make_shared<OdometryMsg>(gt_msg));
        }
        
        last_message_time_ns = serialized_message->time_stamp;
        last_pub_finish_time_s = HighResolutionClock::now();
        ++message_count;
    }

#else
    uint64_t last_message_time_ns = 0u;
    rosbag::View::iterator it_message = bag_view_->begin();
    while (it_message != bag_view_->end()) {
        if (interface_ptr_->IsDataFinished()) {
            break;
        }

        const rosbag::MessageInstance& message = *it_message;

        if (first_mssage) {
            first_mssage = false;
        } else {
#if 1
            uint64_t current_message_time_ns = message.getTime().toNSec();
            ControlDataPlayRate(last_pub_finish_time_s,
                                common::NanoSecondsToSeconds(last_message_time_ns),
                                common::NanoSecondsToSeconds(current_message_time_ns));
#endif
        }

        const std::string& topic = message.getTopic();
        CHECK(!topic.empty());
        if (topic == ros_topics_.at(CameraTopicName)) {
            ImageMsgPtr image_msg = message.instantiate<ImageMsg>();
            interface_ptr_->FillImageMsg(image_msg);
        } else if (topic == ros_topics_.at(CameraDepthTopicName)) {
            ImageMsgPtr depth_msg = message.instantiate<ImageMsg>();
            interface_ptr_->FillDepthMsg(depth_msg);
        } else if (topic == ros_topics_.at(OdomTopicName)) {
            OdometryMsgPtr odom_msg = message.instantiate<OdometryMsg>();
            interface_ptr_->FillOdomMsg(odom_msg);
        } else if (topic == ros_topics_.at(ImuTopicName)) {
            ImuMsgPtr imu_msg = message.instantiate<ImuMsg>();
            interface_ptr_->FillImuMsg(imu_msg);           
        } else if (topic == ros_topics_.at(ScanTopicName)) {
            LaserScanMsgPtr scan_msg = message.instantiate<LaserScanMsg>();
            interface_ptr_->FillScanMsg(scan_msg);            
        } else if (topic == ros_topics_.at(ScanCloudTopicName)) {
            PointCloudMsgPtr scan_pc2_msg = message.instantiate<PointCloudMsg>();
            interface_ptr_->FillScanPc2Msg(scan_pc2_msg);            
        } else if (topic == ros_topics_.at(GroundTruthTopicName)) {
            OdometryMsgPtr gt_msg = message.instantiate<OdometryMsg>();
            interface_ptr_->FillGroundTruthMsg(gt_msg);
        }

        last_message_time_ns = message.getTime().toNSec();
        last_pub_finish_time_s = HighResolutionClock::now();
        ++it_message;
        ++message_count;
    }
#endif
    VLOG(0) << "Rosbag parsing finished! Processed totally "
            << message_count << " messages.";
    interface_ptr_->AddEndSignal();
}
}