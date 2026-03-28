
#ifndef SYSTEM_COMMON_CONFIG_BASE_H_
#define SYSTEM_COMMON_CONFIG_BASE_H_

#include <map>
#include <string>

namespace common {
class ConfigBase {
 public:
    //! Default constructor.
    ConfigBase() = default;

    //! Constructor with config file path.
    explicit ConfigBase(const std::string& config_file_path)
        : config_file_path_(config_file_path) {}

    virtual ~ConfigBase() = default;

    //! Getters.
    inline const std::string GetConfigFilePath() const {
        return config_file_path_;
    }

    //! Outputters.
    virtual void PrintConfigToGlog() const = 0;

 protected:
    //! Load configurations from config_file_path_;
    virtual void LoadConfigFromFile() = 0;

    //! Configuration file path.
    const std::string config_file_path_ = "";

    //! Load option.
    // 0: just load. 1: load and print to glog.
    enum LoadOption {LOAD_ONLY, LOAD_AND_PRINT_GLOG};
    std::map<LoadOption, std::string> LoadOptionString = {
        {LoadOption::LOAD_ONLY, "LOAD_ONLY"},
        {LoadOption::LOAD_AND_PRINT_GLOG, "LOAD_AND_PRINT_GLOG"}};
    LoadOption load_option_;
};
}  // namespace common

#endif  // SYSTEM_COMMON_SYSTEM_COMMON_CONFIG_BASE_H_
