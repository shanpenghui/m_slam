#include <iostream>
#include <memory>
#include <syscall.h>

#include "interface/interface.h"
#include "ros_handler/ros_subscriber_handler.h"
#include "ros_handler/ros_publisher_handler.h"
#include "ros_handler/ros_service_handler.h"
#include "ros_handler/ros_life_circle_handler.h"
#include "ros_handler/sig_crash.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    LOG(INFO) << "ROS main node started.";
    LOG(INFO) << "main thread pid: " << syscall(SYS_gettid);

    std::shared_ptr<NodeHandle> node_handler_ptr;
    std::string config_file_path = "";
    int log_level = 0;
    node_handler_ptr = NodeHandle::make_shared("m_vins_node");
    node_handler_ptr->declare_parameter<std::string>("slam_yaml_config", "");
    node_handler_ptr->declare_parameter<int>("log_level", 0);
    node_handler_ptr->get_parameter<std::string>("slam_yaml_config", config_file_path);
    node_handler_ptr->get_parameter<int>("log_level", log_level);
    int glog_argc = 2;
    std::string log_level_argv = "--v=" + std::to_string(log_level);
    char** glog_argv = new char*[glog_argc];
    glog_argv[0] = new char[std::strlen(argv[0]) + 1];
    std::strcpy(glog_argv[0], argv[0]);
    glog_argv[1] = new char[log_level_argv.size() + 1];
    std::strcpy(glog_argv[1], log_level_argv.c_str());
    common::GLogHelper gLogHelper(glog_argc, glog_argv);
    delete[] glog_argv[0];
    delete[] glog_argv;

    bool online = true;
    if (!(common::fileExists(config_file_path) ||
          common::pathExists(config_file_path))) {
        LOG(ERROR) << "ERROR: config file or path not exist, return.";
        return 1;
    } else if (common::fileExists(config_file_path)) {
        online = false;
    } else if (common::pathExists(config_file_path)) {
        online = true;
    }

    std::string complete_config_file_name_default;
    if (online) {
        complete_config_file_name_default = common::ConcatenateFilePathFrom(
            config_file_path, kConfigIdleFileName);
    } else {
        complete_config_file_name_default = config_file_path;
    }

    common::SlamConfigPtr config;
    config.reset(new common::SlamConfig(
        complete_config_file_name_default, online));

    gLogHelper.SetLogPath(config->log_path, "m_vins_node");
    gLogHelper.UseTimeTable(config->log_time_table);
    gLogHelper.ConfigForOnline(config->online);

    mvins::Interface interface(config_file_path, config);

    const mvins::TopicMap ros_topics = mvins::ConfigurateRosTopics();

    // Topic subscribers listen to online ros topics.
    mvins::TopicSubscriberOptions subscriber_options;
    mvins::TopicSubscribers topic_subscribers;
    mvins::TopicPublishers topic_publishers;
    mvins::ServiceOptions service_options;
    mvins::Services services;

    signal(SIGINT, InterruptCallBack);

    SetCrashFilePath(config->log_path);
    // Segmentation Fault
    signal(SIGSEGV, SigCrash);
    // Aborted
    signal(SIGABRT, SigCrash);

    // Set output query rate and start ros life circle.
    const int sleep_ms = 25;
    mvins::StartRosLifeCircle(sleep_ms,
                              config,
                              ros_topics,
                              node_handler_ptr,
                              &interface,
                              &subscriber_options,
                              &topic_subscribers,
                              &topic_publishers,
                              &service_options,
                              &services,
                              &received_end_signal);
    rclcpp::shutdown();
    return 0;
}
