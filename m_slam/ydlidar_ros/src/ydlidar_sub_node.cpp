#include "ydlidar_sub_node.h"

#include <list>
#include <functional>

#include "glog/logging.h"
#include "YDlidarDriver.h"
#include "GS2LidarDriver.h"
#include "core/common/ydlidar_help.h"
#include "core/common/DriverInterface.h"

namespace ydlidar_node {

YdlidarSubNode::YdlidarSubNode(std::string name,
                              const YdlidarNodeCfg &node_cfg,
                              std::shared_ptr<rclcpp::Node> nh)
  : name_(name), sleep_state_(true), sleep_mode_(node_cfg.sleep_mode),
    sleep_delay_(node_cfg.sleep_delay), dev_num_(node_cfg.num),
    cascade_(false), processor_flg_(false), monitor_flg_(false) {
  if (dev_num_ > 1 && dev_num_ > node_cfg.port.size()) {
    cascade_ = true;
  }
  ydlidar_dev_t dev_cfg;
  std::string model = "";
  int ydlidar_type;
  int ydlidar_dev_type = YDLIDAR_TYPE_SERIAL;
  for (auto port: node_cfg.port) {
    model = "";
    for (auto dev: node_cfg.dev) {
      ydlidar::DriverInterface *lidar_drv;
      std::string cfg_model = "";
      if ("G7" == dev.model ||
          "T-mini Pro" == dev.model ||
          "TGx" == dev.model) {
        lidar_drv = new ydlidar::YDlidarDriver();
      }
      else if ("GS2" == dev.model) {
        lidar_drv = new ydlidar::GS2LidarDriver();
      }
      else {
        continue;
      }
      if (RESULT_OK == lidar_drv->connect(port.c_str(), dev.baud_rate)) {
        device_info dev_info;
        if (RESULT_OK == lidar_drv->getDeviceInfo(dev_info)) {
          model = ydlidar::lidarModelToString(dev_info.model);
          switch (dev_info.model) {
          case ydlidar::YDlidarDriver::YDLIDAR_G7:
          case ydlidar::DriverInterface::YDLIDAR_TminiPRO:
            cfg_model = model;
            ydlidar_type = TYPE_TRIANGLE;
            break;
          case ydlidar::YDlidarDriver::YDLIDAR_TG15:
          case ydlidar::YDlidarDriver::YDLIDAR_TG30:
            cfg_model = "TGx";
            ydlidar_type = TYPE_TOF;
            break;
          case ydlidar::DriverInterface::YDLIDAR_GS2:
            cfg_model = model;
            ydlidar_type = TYPE_GS;
            break;
          default:
            break;
          }
          bool valid_cfg = false;
          for (auto dev: node_cfg.dev) {
            if (cfg_model == dev.model) {
              dev_cfg = dev;
              valid_cfg = true;
              break;
            }
          }
          if (!valid_cfg) {
            model = "";
          }
          lidar_drv->disconnect();
          break;
        }
      }
      lidar_drv->disconnect();
    }
    if ("" == model) {
      continue;
    }

    CYdLidar *lidar = new CYdLidar();
    lidar->setlidaropt(LidarPropSerialPort, port.c_str(), port.size());
    lidar->setlidaropt(LidarPropSerialBaudrate, &dev_cfg.baud_rate, sizeof(int));
    lidar->setlidaropt(LidarPropIgnoreArray, dev_cfg.ignore_array.c_str(),
                    dev_cfg.ignore_array.size());
    lidar->setlidaropt(LidarPropLidarType, &ydlidar_type, sizeof(int));
    lidar->setlidaropt(LidarPropDeviceType, &ydlidar_dev_type, sizeof(int));
    lidar->setlidaropt(LidarPropSampleRate, &dev_cfg.sample_rate, sizeof(int));
    lidar->setlidaropt(LidarPropAbnormalCheckCount, &dev_cfg.abnormal_ck_cnt, sizeof(int));
    lidar->setlidaropt(LidarPropIntenstiyBit, &dev_cfg.intensity_bit, sizeof(int));
    lidar->setlidaropt(LidarPropFixedResolution, &dev_cfg.fixed_resolution, sizeof(bool));
    lidar->setlidaropt(LidarPropReversion, &dev_cfg.reversion, sizeof(bool));
    lidar->setlidaropt(LidarPropInverted, &dev_cfg.inverted, sizeof(bool));
    lidar->setlidaropt(LidarPropAutoReconnect, &dev_cfg.auto_reconnect, sizeof(bool));
    lidar->setlidaropt(LidarPropSingleChannel, &dev_cfg.single_chn, sizeof(bool));
    lidar->setlidaropt(LidarPropIntenstiy, &dev_cfg.intensity, sizeof(bool));
    lidar->setlidaropt(LidarPropSupportMotorDtrCtrl, &dev_cfg.motor_DTR, sizeof(bool));
    lidar->setlidaropt(LidarPropSupportHeartBeat, &dev_cfg.heart_beat, sizeof(bool));
    lidar->setlidaropt(LidarPropMaxAngle, &dev_cfg.angle_max, sizeof(float));
    lidar->setlidaropt(LidarPropMinAngle, &dev_cfg.angle_min, sizeof(float));
    lidar->setlidaropt(LidarPropMaxRange, &dev_cfg.range_max, sizeof(float));
    lidar->setlidaropt(LidarPropMinRange, &dev_cfg.range_min, sizeof(float));
    lidar->setlidaropt(LidarPropScanFrequency, &dev_cfg.frequency, sizeof(float));

    lidar->enableGlassNoise(false);
    lidar->enableSunNoise(false);

    for (int i = 0; i < 3; i++) {
      if (lidar->initialize()) {
        LidarVersion version;
        float max_range;
        float freq;
        int sample_rate;
        lidar->GetLidarVersion(version);
        lidar->getlidaropt(LidarPropMaxRange, &max_range, sizeof(float));
        lidar->getlidaropt(LidarPropScanFrequency, &freq, sizeof(float));
        lidar->getlidaropt(LidarPropSampleRate, &sample_rate, sizeof(int));
        std::string sn_str = "";
        for (auto chr: version.sn) {
          sn_str += std::to_string(chr);
        }
        LOG(INFO) << "\n" << name_ << " find new ydlidar:\n"
                << "  " << "port: " << port << "\n"
                << "  " << "baudrate: " << dev_cfg.baud_rate << "\n"
                << "  " << "model: " << model << "\n"
                << "  " << "sw ver: " << std::to_string(version.soft_major) 
                << "." << std::to_string(version.soft_minor)
                << "." << std::to_string(version.soft_patch) << "\n"
                << "  " << "hw ver: " << std::to_string(version.hardware) << "\n"
                << "  " << "sn: " << sn_str << "\n"
                << "  " << "max range: " << std::to_string(max_range) << "\n"
                << "  " << "freq: " << std::to_string(freq) << "\n"
                << "  " << "sample rate: " << std::to_string(sample_rate);

        lidar_.push_back(lidar);
        lidar_mutex_.push_back(new std::mutex());
        model_.push_back(model);
        version_.push_back(version);
        max_range_.push_back(max_range);
        freq_.push_back(freq);
        sample_rate_.push_back(sample_rate);
        fail_cnt_.push_back(0);
        break;
      }
    }
  }

  if (lidar_.size() <= 0) {
    return;
  }

  float processor_prd = 0;
  for (int i = 0; i < lidar_.size(); ++i) {
    processor_prd = processor_prd >= 1 / freq_[i]? processor_prd_ : 1 / freq_[i];
  }
  processor_prd_ = processor_prd == 0? 100000 : processor_prd * 1000000;
  monitor_prd_ = 1000000;
  
  activate_time_ = std::chrono::steady_clock::now();

  if (dev_num_ > 1) {
    for (int i = 0; i < dev_num_; ++i) {
      rclcpp::Publisher<LaserScanMsg>::SharedPtr scan_pub 
        = nh->create_publisher<LaserScanMsg>(node_cfg.topic_name + std::to_string(i), 1);
      scan_pub_.push_back(scan_pub);
      frame_id_.push_back(node_cfg.frame_id + std::to_string(i));
    }
  }
  else {
    rclcpp::Publisher<LaserScanMsg>::SharedPtr scan_pub 
      = nh->create_publisher<LaserScanMsg>(node_cfg.topic_name, 1);
    scan_pub_.push_back(scan_pub);
    frame_id_.push_back(node_cfg.frame_id);
  }
  action_srv_ = nh->create_service<mirobot_msgs::srv::YdlidarRosAction>(
                  node_cfg.action_name, 
                  std::bind(&YdlidarSubNode::ActionCallBack,
                            this,
                            std::placeholders::_1,
                            std::placeholders::_2));
}

YdlidarSubNode::~YdlidarSubNode() {
  if (lidar_.size() <= 0) {
    return;
  }
  
  if (monitor_flg_){
    monitor_flg_ = false;
    monitor_.join();
  }
  if (processor_flg_) {
    processor_flg_ = false;
    processor_.join();
  }

  for (auto lidar: lidar_) {
    lidar->turnOff();
    lidar->disconnecting();
    delete lidar;
  }

  for (auto lidar_mutex: lidar_mutex_) {
    delete lidar_mutex;
  }
}

bool YdlidarSubNode::ActionCallBack(YdlidarRosActionRequestPtr req_ptr,
                                    YdlidarRosActionResponsePtr res_ptr) {
  auto &req = *req_ptr;
  auto &res = *res_ptr;
  if ("get_state" == req.action) {
    for (auto fail_cnt: fail_cnt_) {
      if (fail_cnt < 5) {
        res.param.push_back(std::to_string(1));
      }
      else {
        res.param.push_back(std::to_string(0));
      }
    }
    res.feedback = 1;
  }
  else if ("get_version" == req.action) {
    for (int i = 0; i < model_.size(); ++i) {
      res.param.push_back(model_[i]);
      res.param.push_back(std::to_string((int)version_[i].soft_major) 
                          + "." + std::to_string((int)version_[i].soft_minor)
                          + "." + std::to_string((int)version_[i].soft_patch));  
      res.param.push_back(std::to_string((int)version_[i].hardware));
      std::string sn_str = "";
      for (auto chr: version_[i].sn)
      {
          sn_str += std::to_string(chr);
      }
      res.param.push_back(sn_str);
    }
    res.feedback = 1;
  }
  else if ("get_param" == req.action) {
    for (int i = 0; i < max_range_.size(); ++i) {
      res.param.push_back(std::to_string(max_range_[i]));
      res.param.push_back(std::to_string(freq_[i]));
      res.param.push_back(std::to_string(sample_rate_[i]));
    }
    res.feedback = 1;
  }
  else {
    LOG(ERROR) << name_ << " unknow ros action: " << req.action;
    res.feedback = 0;
  }

  return true;
}

void YdlidarSubNode::Monitor() {
  while (monitor_flg_) {
    bool get_subscribers = false;
    for (auto scan_pub: scan_pub_) {
      if (scan_pub->get_subscription_count() > 0) {
        get_subscribers |= true;
      }
    }

    if (get_subscribers) {
      if (sleep_state_) {
        LOG(INFO) << name_ << " get subscribers";
        bool turn_on_lidar = true;
        for (int i = 0; i < lidar_.size(); ++i) {
          lidar_mutex_[i]->lock();
          if (!lidar_[i]->turnOn()) {
            turn_on_lidar &= false;
          }
          lidar_mutex_[i]->unlock();
        }
        if (turn_on_lidar) {
          LOG(INFO) << name_ << " turn on ydlidar";
          processor_flg_ = true;
          processor_ = std::thread([&](){Processor();});
          sleep_state_ = false;
          LOG(INFO) << name_ << " exit sleep mode";
        }
        else {
          LOG(ERROR) << name_ << " turn on ydlidar failed";
        }
      }
      activate_time_ = std::chrono::steady_clock::now();
    }
    else {
      if (!sleep_state_ && sleep_mode_) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - activate_time_);
        if (duration.count() > sleep_delay_) {
          LOG(INFO) << name_ << " has no subscribers";
          if (processor_flg_) {
            processor_flg_ = false;
            processor_.join();
          }
          LOG(INFO) << name_ << " stop processor";
          bool turn_off_lidar = true;
          for (int i = 0; i < lidar_.size(); ++i) {
            lidar_mutex_[i]->lock();
            if (!lidar_[i]->turnOff()) {
              turn_off_lidar &= false;
            }
            lidar_mutex_[i]->unlock();
          }
          if (turn_off_lidar) {
            sleep_state_ = true;
            LOG(INFO) << name_ << " entry sleep state";
          }
          else {
            LOG(ERROR) << name_ << " turn off ydlidar failed";
          }
        }
      }
    }
    usleep(monitor_prd_);
  }
}

void YdlidarSubNode::Processor() {
  while (processor_flg_) {
    auto start_time = std::chrono::steady_clock::now();
    for (int lidar_index = 0; lidar_index < dev_num_; ++lidar_index) {
      rclcpp::Publisher<LaserScanMsg>::SharedPtr scan_pub;
      LaserScan scan;
      bool rslt;
      std::string frame_id;
      if (cascade_) {
        lidar_mutex_[0]->lock();
        rslt = lidar_[0]->doProcessSimple(scan);
        lidar_mutex_[0]->unlock();
      }
      else {
        if (lidar_index >= lidar_.size())
          break;
        lidar_mutex_[lidar_index]->lock();
        rslt = lidar_[lidar_index]->doProcessSimple(scan);
        lidar_mutex_[lidar_index]->unlock();
      }
      if (rslt) {
        LaserScanMsg scan_msg;
        scan_msg.header.stamp = rclcpp::Time(scan.stamp);
        if (cascade_) {
          if (scan.moduleNum < scan_pub_.size()) {
            scan_pub = scan_pub_[scan.moduleNum];
            scan_msg.header.frame_id = frame_id_[scan.moduleNum];
          }
          else {
            LOG(WARNING) << name_ << " unexpected module num: " << scan.moduleNum;
            continue;
          }
        }
        else {
          scan_pub = scan_pub_[lidar_index];
          scan_msg.header.frame_id = frame_id_[lidar_index];
        }
        scan_msg.angle_min = (scan.config.min_angle);
        scan_msg.angle_max = (scan.config.max_angle);
        scan_msg.angle_increment = (scan.config.angle_increment);
        scan_msg.scan_time = scan.config.scan_time;
        scan_msg.time_increment = scan.config.time_increment;
        scan_msg.range_min = (scan.config.min_range);
        scan_msg.range_max = (scan.config.max_range);

        int size = (scan.config.max_angle - scan.config.min_angle) / scan.config.angle_increment + 1;
        scan_msg.ranges.resize(size);
        scan_msg.intensities.resize(size);
        for (int i = 0; i < scan.points.size(); ++i) {
          int index = std::ceil((scan.points[i].angle - scan.config.min_angle) / scan.config.angle_increment);
          if (index >= 0 && index < size) {
            scan_msg.ranges[index] = scan.points[i].range;
            scan_msg.intensities[index] = scan.points[i].intensity;
          }
        }
        scan_pub->publish(scan_msg);
        fail_cnt_[lidar_index] = 0;
      }
      else {
        if (++fail_cnt_[lidar_index] >= 5) {
          fail_cnt_[lidar_index] = 5;
        }
      }
    }
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    int usec = processor_prd_ - duration.count();
    if (usec > 0) {
      usleep(usec);
    }
  }
}

void YdlidarSubNode::Start() {
  if (lidar_.size() <= 0) {
    return;
  }

  if (!monitor_flg_) {
    monitor_flg_ = true;
    monitor_ = std::thread([&](){Monitor();});
  }
  LOG(INFO) << name_  << " monitor start";
}

void YdlidarSubNode::Stop() {
  if (lidar_.size() <= 0) {
    return;
  }

  if (monitor_flg_){
    monitor_flg_ = false;
    monitor_.join();
  }
  if (processor_flg_) {
    processor_flg_ = false;
    processor_.join();
  }

  for (int i = 0; i < lidar_.size(); ++i) {
    lidar_mutex_[i]->lock();
    lidar_[i]->turnOff();
    lidar_mutex_[i]->unlock();
  }

  LOG(INFO) << name_  << " stop";
}

}
