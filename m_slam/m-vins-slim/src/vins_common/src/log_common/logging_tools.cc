
#include <algorithm>
#include <functional>
#include <iostream>

#include "log_common/logging_tools.h"
#include "time_common/time_table.h"

namespace common {

GLogHelper::GLogHelper(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Parse data from input.
    google::ParseCommandLineFlags(&argc, &argv, true);

    // Let glog output failure.
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
}

void GLogHelper::SetLogPath(const std::string& path_set,
                            const std::string& symlink_name) {
    // Log path can be specified.
    std::string path = path_set;
    if (common::GetEnvironmentVariable(path, "LOCATION_LOG_DIR")) {
        LOG(INFO) << "GetEnvironmentVariable "
                  << "LOCATION_LOG_DIR Successfully!";
    } else {
        LOG(INFO) << "GetEnvironmentVariable LOCATION_LOG_DIR Failed!";
    }

    // Add protection for human input error.
    if (path.back() != '/') {
        path += '/';
    }

    if (!common::pathExists(path)) {
        LOG(ERROR) << "Log directory can not be found: " << path;
    }

    google::SetLogDestination(google::INFO, path.c_str());
    LOG(INFO) << "Log path: " << path.c_str();

    google::SetLogSymlink(google::INFO, symlink_name.c_str());
    LOG(INFO) << "Symlink name: " << symlink_name.c_str();
}

void GLogHelper::UseTimeTable(const bool use_time_table) {
    if (use_time_table) {
        common::TimeTable::Reset();
    }
}

void GLogHelper::ConfigForOnline(const bool use_online_mode) {
    if (use_online_mode) {
        FLAGS_logbufsecs = 0; // second.
        FLAGS_max_log_size = 50; // MB.
        FLAGS_stop_logging_if_full_disk = true;
    }
}

GLogHelper::~GLogHelper() {
    common::TimeTable::PrintAll();
    google::ShutdownGoogleLogging();
}

bool GetEnvironmentVariable(std::string& str,
                            const char* environment_variable) {
    char* env_var_cstr = nullptr;
    env_var_cstr = std::getenv(environment_variable);
    if ( env_var_cstr ) {
        str = std::string(env_var_cstr);
        return true;
    } else {
        return false;
    }
}
}  // namespace common

