#ifndef YDLIDAR_NODE_CONFIG_H
#define YDLIDAR_NODE_CONFIG_H

#include <string>
#include <vector>

#include "common/config_base.h"
#include "yaml-cpp/yaml.h"

namespace ydlidar_node {

constexpr char kLoadOptionString[] = "load_option";
constexpr char kPortString[] = "port";
constexpr char kNumString[] = "num";
constexpr char kSleepModeString[] = "sleep_mode";
constexpr char kSleepDelayString[] = "sleep_delay";
constexpr char kActionNameString[] = "action_name";
constexpr char kTopicNameString[] = "topic_name";
constexpr char kFrameIdString[] = "frame_id";
constexpr char kBaudRateString[] = "baud_rate";
constexpr char kIgnoreArrayString[] = "ignore_array";
constexpr char kFixedResolutionString[] = "fixed_resolution";
constexpr char kReversionString[] = "reversion";
constexpr char kInvertedString[] = "inverted";
constexpr char kAutoReconnectString[] = "auto_reconnect";
constexpr char kSingleChnString[] = "single_chn";
constexpr char kIntensityString[] = "intensity";
constexpr char kMotorDTRString[] = "motor_DTR";
constexpr char kHeartBeatString[] = "heart_beat";
constexpr char kAngleMinString[] = "angle_min";
constexpr char kAngleMaxString[] = "angle_max";
constexpr char kRangeMinString[] = "range_min";
constexpr char kRangeMaxString[] = "range_max";
constexpr char kFrequencyString[] = "frequency";
constexpr char kSampleRateString[] = "sample_rate";
constexpr char kAbnormalCkCntString[] = "abnormal_ck_cnt";
constexpr char kIntensityBitString[] = "intensity_bit";

struct ydlidar_dev_t {
  std::string model;
  int baud_rate;
  std::string ignore_array;
  bool fixed_resolution;
  bool reversion;
  bool inverted;
  bool auto_reconnect;
  bool single_chn;
  bool intensity;
  bool motor_DTR;
  bool heart_beat;
  float angle_min;
  float angle_max;
  float range_min;
  float range_max;
  float frequency;
  int sample_rate;
  int abnormal_ck_cnt;
  int intensity_bit;
};

class YdlidarNodeCfg {
public:
  std::vector<std::string> port;
  int num;
  bool sleep_mode;
  int sleep_delay;
  std::string action_name;
  std::string topic_name;
  std::string frame_id;

  std::vector<ydlidar_dev_t> dev;
};

class YdlidarConfig: public common::ConfigBase {
public:
  YdlidarConfig() = default;
  explicit YdlidarConfig(const std::string &config_file_path);
  virtual ~YdlidarConfig() = default;

  std::vector<YdlidarNodeCfg> node_cfg;
  std::vector<std::string> ydlidar_node_list;
  std::vector<std::vector<std::string>> ydlidar_dev_list;

  void PrintConfigToGlog() const override;
private:
  void LoadConfigFromFile() override;
};

typedef std::shared_ptr<YdlidarConfig> YdlidarConfigPtr;

}

#endif
