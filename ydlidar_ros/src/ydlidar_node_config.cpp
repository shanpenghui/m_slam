#include "ydlidar_node_config.h"

#include "glog/logging.h"

namespace ydlidar_node {

YdlidarConfig::YdlidarConfig(const std::string &config_file_path)
  : ConfigBase(config_file_path) {
  ydlidar_node_list.push_back("chassis_lidar");
  ydlidar_node_list.push_back("chassis_line_lidar");
  ydlidar_dev_list.push_back({"G7", "T-mini Pro", "TGx"});
  ydlidar_dev_list.push_back({"GS2"});

  LoadConfigFromFile();
}

void YdlidarConfig::LoadConfigFromFile() {
  YAML::Node file_node;
  try {
    file_node = YAML::LoadFile(config_file_path_.c_str());
  } catch (const std::exception &ex) {
    LOG(ERROR) << "load " << config_file_path_.c_str() << " file failed: " << ex.what();
    return;
  }

  load_option_ = static_cast<LoadOption>(file_node[kLoadOptionString].as<int>());

  for (int i = 0; i < ydlidar_node_list.size(); ++i) {
    std::string sub_node = ydlidar_node_list[i];
    YdlidarNodeCfg node_cfg;
    YAML::Node yaml_node = file_node[sub_node.c_str()];

    for (int j = 0; j < yaml_node[kPortString].size(); ++j) {
      node_cfg.port.push_back(yaml_node[kPortString][j].as<std::string>());
    }
    node_cfg.num = yaml_node[kNumString].as<int>();
    node_cfg.sleep_mode = yaml_node[kSleepModeString].as<bool>();
    node_cfg.sleep_delay = yaml_node[kSleepDelayString].as<int>();
    node_cfg.action_name = yaml_node[kActionNameString].as<std::string>();
    node_cfg.topic_name = yaml_node[kTopicNameString].as<std::string>();
    node_cfg.frame_id = yaml_node[kFrameIdString].as<std::string>();

    for (auto sub_dev: ydlidar_dev_list[i]) {
      ydlidar_dev_t dev;
      YAML::Node sub_dev_node = yaml_node[sub_dev.c_str()];

      dev.model = sub_dev;
      dev.baud_rate = sub_dev_node[kBaudRateString].as<int>();
      dev.ignore_array = sub_dev_node[kIgnoreArrayString].as<std::string>();
      dev.fixed_resolution = sub_dev_node[kFixedResolutionString].as<bool>();
      dev.reversion = sub_dev_node[kReversionString].as<bool>();
      dev.inverted = sub_dev_node[kInvertedString].as<bool>();
      dev.auto_reconnect = sub_dev_node[kAutoReconnectString].as<bool>();
      dev.single_chn = sub_dev_node[kSingleChnString].as<bool>();
      dev.intensity = sub_dev_node[kIntensityString].as<bool>();
      dev.motor_DTR = sub_dev_node[kMotorDTRString].as<bool>();
      dev.heart_beat = sub_dev_node[kHeartBeatString].as<bool>();
      dev.angle_min = sub_dev_node[kAngleMinString].as<float>();
      dev.angle_max = sub_dev_node[kAngleMaxString].as<float>();
      dev.range_min = sub_dev_node[kRangeMinString].as<float>();
      dev.range_max = sub_dev_node[kRangeMaxString].as<float>();
      dev.frequency = sub_dev_node[kFrequencyString].as<float>();
      dev.sample_rate = sub_dev_node[kSampleRateString].as<int>();
      dev.abnormal_ck_cnt = sub_dev_node[kAbnormalCkCntString].as<int>();
      dev.intensity_bit = sub_dev_node[kIntensityBitString].as<int>();

      node_cfg.dev.push_back(dev);
    }
    this->node_cfg.push_back(node_cfg);
  }

  if (LOAD_AND_PRINT_GLOG == load_option_) {
    PrintConfigToGlog();
  }
  return;
}

void YdlidarConfig::PrintConfigToGlog() const {
  std::string log_str = "";
  log_str.append("\ncfg file info: \n");
  log_str.append("path: " + config_file_path_);
  for (int i = 0; i < ydlidar_node_list.size(); ++i) {
    log_str.append("\n" + ydlidar_node_list[i] + ":\n" + 
                    "  " + kPortString + ": ");
    for (auto port: node_cfg[i].port) {
      log_str.append(port + " ");
    }
    log_str.append("\n  " + std::string(kNumString) + ": " + std::to_string(node_cfg[i].num) + "\n" +
                  "  " + kSleepModeString + ": " + std::to_string(node_cfg[i].sleep_mode) + "\n" +
                  "  " + kSleepDelayString + ": " + std::to_string(node_cfg[i].sleep_delay) + "\n" +
                  "  " + kActionNameString + ": " + node_cfg[i].action_name + "\n" +
                  "  " + kTopicNameString + ": " + node_cfg[i].topic_name + "\n" +
                  "  " + kFrameIdString + ": " + node_cfg[i].frame_id);

    for (auto dev: node_cfg[i].dev) {
      log_str.append("\n  " + dev.model + ":\n" +
                    "    " + kBaudRateString + ": " + std::to_string(dev.baud_rate) + "\n" +
                    "    " + kIgnoreArrayString + ": " + dev.ignore_array + "\n" +
                    "    " + kFixedResolutionString + ": " + std::to_string(dev.fixed_resolution) + "\n" +
                    "    " + kReversionString + ": " + std::to_string(dev.reversion) + "\n" +
                    "    " + kInvertedString + ": " + std::to_string(dev.inverted) + "\n" +
                    "    " + kAutoReconnectString + ": " + std::to_string(dev.auto_reconnect) + "\n" +
                    "    " + kSingleChnString + ": " + std::to_string(dev.single_chn) + "\n" +
                    "    " + kIntensityString + ": " + std::to_string(dev.intensity) + "\n" +
                    "    " + kMotorDTRString + ": " + std::to_string(dev.motor_DTR) + "\n" +
                    "    " + kHeartBeatString + ": " + std::to_string(dev.heart_beat) + "\n" +
                    "    " + kAngleMinString + ": " + std::to_string(dev.angle_min) + "\n" +
                    "    " + kAngleMaxString + ": " + std::to_string(dev.angle_max) + "\n" +
                    "    " + kRangeMinString + ": " + std::to_string(dev.range_min) + "\n" +
                    "    " + kRangeMaxString + ": " + std::to_string(dev.range_max) + "\n" +
                    "    " + kFrequencyString + ": " + std::to_string(dev.frequency) + "\n" +
                    "    " + kSampleRateString + ": " + std::to_string(dev.sample_rate) + "\n" +
                    "    " + kAbnormalCkCntString + ": " + std::to_string(dev.abnormal_ck_cnt) + "\n" +
                    "    " + kIntensityBitString + ": " + std::to_string(dev.intensity_bit));
    }
  }

  LOG(INFO) << log_str;

  return;
}

}
