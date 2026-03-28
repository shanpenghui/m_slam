
#ifndef LOGGING_TOOLS_H_
#define LOGGING_TOOLS_H_

#include <sys/stat.h>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "file_common/file_system_tools.h"
namespace common {

bool GetEnvironmentVariable(std::string& str,
                            const char* environment_variable);

class GLogHelper {
 public:
    // Initialize glog.
    GLogHelper(int argc, char** argv);

    // Set the path to save log.
    void SetLogPath(const std::string& path_set,
                    const std::string& symlink_name);

    // Set whether to use time table.
    void UseTimeTable(const bool use_time_table);

    // Config for online mode.
    void ConfigForOnline(const bool use_online_mode);

    ~GLogHelper();
};

}  // namespace common
#endif  // LOGGING_TOOLS_H_

