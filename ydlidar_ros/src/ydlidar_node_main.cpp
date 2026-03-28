#include "ydlidar_node.h"

#include <csignal>
#include <errno.h>
#include <execinfo.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

std::string ydlidar_node_version = "A.01.01.02";

DEFINE_string(log_file_dir, "log/", "Directory to save log files.");
DEFINE_string(cfg_file_path, "cfg/config.yaml", "The yaml of ydlidar node config.");

void GetCompileDate(int16_t *yyyy, int8_t *mm, int8_t *dd) {
  const char *Months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  const char Date[12] = __DATE__;
  uint8_t i;

  for(i = 0; i < 12; i++) {
    if(memcmp(Date, Months[i], 3) == 0) {
      *mm = i + 1;
    }
  }
  *yyyy = atoi(Date + 7);
  *dd = atoi(Date + 4);
}

void sig_crash(int sig) {
  const int BACKTRACE_SIZE = 128;
  void *array[BACKTRACE_SIZE]={0};

  signal(sig, SIG_DFL);

  int size = backtrace(array, BACKTRACE_SIZE);
  char **strings = backtrace_symbols(array, size);
  for (int i = 0; i < size; ++i) {
    LOG(ERROR) << strings[i];
  }
  free(strings);

  signal(sig, SIG_DFL);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO, FLAGS_log_file_dir.c_str());
  google::SetLogFilenameExtension("ydlidar_");

  int16_t yyyy;
  int8_t mm;
  int8_t dd;
  GetCompileDate(&yyyy, &mm, &dd);
  LOG(INFO) << "\n*******************************\n"
            << "ydlidar node\n"
            << "Version: " << ydlidar_node_version << "\n"
            << "Compile time: " << yyyy << "-" << (int)mm << "-" << (int)dd << " " << __TIME__ << "\n"
            << "Copyright(c): Midea. Co. Ltd\n"
            << "*******************************";

  signal(SIGSEGV, sig_crash);
  signal(SIGABRT, sig_crash);

  ydlidar_node::YdlidarConfig ydlidar_config(FLAGS_cfg_file_path);

  rclcpp::init(argc, argv);
  auto nh = rclcpp::Node::make_shared("ydlidar_node");

  ydlidar_node::YdlidarNode ydlidar_node(ydlidar_config, nh);
  ydlidar_node.Start();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(nh);
  executor.spin();

  ydlidar_node.Stop();
  rclcpp::shutdown();
  google::ShutdownGoogleLogging();

  return 0;
}
