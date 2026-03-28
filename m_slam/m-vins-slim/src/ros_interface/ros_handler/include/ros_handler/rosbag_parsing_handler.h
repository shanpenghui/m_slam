#ifndef MVINS_ROSBAG_PARSING_HANDLER_H_
#define MVINS_ROSBAG_PARSING_HANDLER_H_

#include <chrono>
#include <thread>
#include <rosbag2_cpp/converter.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>

#include "interface/interface.h"
#include "ros_handler/configurate_interface.h"

namespace mvins {
typedef rosbag2_cpp::converter_interfaces::SerializationFormatDeserializer Deserializer;
typedef std::shared_ptr<rosbag2_storage::SerializedBagMessage> SerializedBagMessagePtr;
typedef std::chrono::high_resolution_clock HighResolutionClock;
typedef std::chrono::high_resolution_clock::time_point TimePoint;

class RosbagParser {
public:
    explicit RosbagParser(const std::string& data_directory,
                          const double data_realtime_playback_rate,
                          const TopicMap& ros_topics,
                          Interface* interface_ptr);
    ~RosbagParser();
    void Start();
private:
    void ControlDataPlayRate(
        const TimePoint& time_start,
        const double timestamp_current_frame,
        const double timestamp_next_frame);
    template <typename MessageType>
    MessageType ParseTopic(const SerializedBagMessagePtr& serialized_message,
                           const rosbag2_cpp::ConverterTypeSupport& type_support);
    void RunThread();
    
    std::unique_ptr<std::thread> streaming_thread_ptr_;
    std::unique_ptr<rosbag2_cpp::SerializationFormatConverterFactory> factory_ptr_;
    
    std::unique_ptr<Deserializer> cdr_deserializer_ptr_;

    std::unique_ptr<rosbag2_cpp::Reader> reader_ptr_;
    Interface* interface_ptr_;

    TopicMap ros_topics_;

    const double data_realtime_playback_rate_;
};
}
#endif