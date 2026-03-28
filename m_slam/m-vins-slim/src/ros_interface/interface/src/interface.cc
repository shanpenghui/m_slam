#include "interface/interface.h"

#include <gflags/gflags.h>

#include "file_common/file_system_tools.h"
#include "vins_handler/map_tool.h"
#include "yaml_common/yaml_util.h"

namespace mvins {

constexpr char kSlamShutdown[] = "shutdown";
constexpr char kSlamStart[] = "start";
constexpr char kSlamRestart[] = "restart";
constexpr char kSlamContinue[] = "continue";
constexpr char kSlamStop[] = "stop";
constexpr char kSlamLoadMap[] = "load_map";
constexpr char kSlamSaveMap[] = "save_map";
constexpr char kSlamRemoveMap[] = "remove_map";
constexpr char kSlamResetPose[] = "reset_pose";
constexpr char kSlamOperateMask[] = "operate_mask";
constexpr char kSlamGetDockerPose[] = "get_docker_pose";

enum SlamService {
    Shutdown = 0,
    Start = 1,
    Restart = 2,
    Continue = 3,
    Stop = 4,
    LoadMap = 5,
    SaveMap = 6,
    RemoveMap = 7,
    ResetPose = 8,
    OperateMask = 9,
    GetDockerPose = 10,
    Invalid = 11
};

inline SlamService StringToSlamService(const std::string type_str) {
    if (type_str == kSlamShutdown) {
        return SlamService::Shutdown;
    } else if (type_str == kSlamStart) {
        return SlamService::Start;
    } else if (type_str == kSlamRestart) {
        return SlamService::Restart;
    } else if (type_str == kSlamContinue) {
        return SlamService::Continue;
    } else if (type_str == kSlamStop) {
        return SlamService::Stop;
    } else if (type_str == kSlamLoadMap) {
        return SlamService::LoadMap;
    } else if (type_str == kSlamSaveMap) {
        return SlamService::SaveMap;
    } else if (type_str == kSlamRemoveMap) {
        return SlamService::RemoveMap;
    } else if (type_str == kSlamResetPose) {
        return SlamService::ResetPose;
    } else if (type_str == kSlamOperateMask) {
        return SlamService::OperateMask;
    } else if (type_str == kSlamGetDockerPose) {
        return SlamService::GetDockerPose;
    } else {
        LOG(ERROR) << "Unknow service type.";
        return SlamService::Invalid;
    }
}

//! Interface constructor.
Interface::Interface(const std::string& slam_yamls_path,
                     const common::SlamConfigPtr& config)
    : config_(config),
      slam_yamls_path_(slam_yamls_path) {
    if (config_->online) {
        slam_status_ = SLAM_STATUS::IDLE;
        LOG(WARNING) << "SLAM Status change to IDLE.";
        if (CheckMapExistence()) {
            interface_manager_ptr_.reset(new InterfaceManager(config_));
        } else {
            interface_manager_ptr_.reset(nullptr);
        }
        vins_handler_ptr_.reset(nullptr);
    } else {
        slam_status_ = SLAM_STATUS::OFFLINE;
        interface_manager_ptr_.reset(new InterfaceManager(config_));
        vins_handler_ptr_.reset(
            new vins_handler::VinsHandler(config_,
                                          interface_manager_ptr_->scan_maps_ptr_,
                                          interface_manager_ptr_->summary_map_ptr_,
                                          interface_manager_ptr_->octree_map_ptr_));
        vins_handler_ptr_->Start();
    }
}

Interface::~Interface() {
    // TODO: touch thread.
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->Release();
    }
    LOG(INFO) << "Interface exit.";
}

void Interface::FillResetPoseMsg(
    const PoseWithCovarianceStampedMsgPtr reset_pose) {
    if (nullptr == reset_pose) {
        LOG(WARNING) << "WARNNING: Reset pose message is nullptr";
        return;
    }
    Eigen::Quaterniond rotation(reset_pose->pose.pose.orientation.w,
                                reset_pose->pose.pose.orientation.x,
                                reset_pose->pose.pose.orientation.y,
                                reset_pose->pose.pose.orientation.z);
    Eigen::Vector3d position(reset_pose->pose.pose.position.x,
                            reset_pose->pose.pose.position.y,
                            reset_pose->pose.pose.position.z);
    aslam::Transformation T_OtoM_prior(rotation, position);

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->SetResetPose(T_OtoM_prior, false/*force_reset*/);
    }
}

bool Interface::SlamServiceCall(ServiceRequestPtr req,
                                ServiceResponsePtr res) {
    ServiceRequest& req_ref = *req;
    ServiceResponse& res_ref = *res;
    const SlamService type = StringToSlamService(req_ref.action);
    switch (type) {
        case SlamService::Shutdown: {
            VLOG(0) << "Try shutdown SLAM system by service call.";
            ShutDown();
            res_ref.feedback = 1;
            break;
        }
        case SlamService::Start: {
            VLOG(0) << "Try start SLAM system by service call.";
            SlamStart(&res_ref);
            break;
        }
        case SlamService::Restart: {
            VLOG(0) << "Try restart SLAM system by service call.";
            SlamStart(&res_ref);
            break;
        }
        case SlamService::Continue: {
            VLOG(0) << "Try continue SLAM system by service call.";
            SlamStart(&res_ref);
            break;
        }
        case SlamService::Stop: {
            VLOG(0) << "Try stop SLAM system by service call.";
            SlamIdle();
            res_ref.feedback = 1;
            break;
        }
        case SlamService::LoadMap: {
            VLOG(0) << "Try load Map by service call.";
            LoadMap();
            res_ref.feedback = 1;
            break;
        }
        case SlamService::SaveMap: {
            VLOG(0) << "Try save Map by service call.";
            SaveMap();
            res_ref.feedback = 1;
            break;
        }
        case SlamService::RemoveMap: {
            VLOG(0) << "Try remove Map by service call.";
            RemoveMap();
            res_ref.feedback = 1;
            break;
        }
        case SlamService::ResetPose: {
            VLOG(0) << "Try reset pose by service call.";
            ResetPose(&res_ref);
            break;
        }
        case SlamService::OperateMask: {
            VLOG(0) << "Try operate Mask by service call.";
            OperateMask(req_ref, &res_ref);
            break;
        }
        case SlamService::GetDockerPose: {
            VLOG(0) << "Try get Docker Pose by service call.";
            GetDockerPose(&res_ref);
            break;
        }
        default: {
            VLOG(0) << "Unknow service call type.";
            res_ref.feedback = 0;
        }
    }
    return true;
}

void Interface::AddEndSignal() {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->SetDataFinished();
    }
}

bool Interface::IsNewPose() const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {  
        return vins_handler_ptr_->IsNewPose();
    }
    return false;
}

bool Interface::IsDataFinished() const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        return vins_handler_ptr_->IsDataFinished();
    }
    return false;
}

bool Interface::IsAllFinished() const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        return vins_handler_ptr_->AllThreadFinished();
    } 
    return false;
}

bool Interface::GetNewMap(cv::Mat* map_ptr,
                          Eigen::Vector2d* origin_ptr,
                          common::EigenVector3dVec* map_cloud_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    std::unique_lock<std::mutex> interface_manager_lock(interface_manager_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr && interface_manager_ptr_ != nullptr &&
        vins_handler_ptr_->GetNewMap(map_ptr, origin_ptr, map_cloud_ptr)) {
        if (!map_ptr->empty()) {
            *map_ptr = vins_handler::CreateShowMapMat(*map_ptr,
                                                      config_->do_obstacle_removal, 
                                                      config_->contour_length);
            interface_manager_ptr_->MapThresholding(map_ptr, origin_ptr);
        }
        return true;
    }
    return false;
}

bool Interface::GetNewLoop(common::LoopResult* loop_result_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        return vins_handler_ptr_->GetNewLoop(loop_result_ptr);
    }
    return false;
}

void Interface::GetNewKeyFrame(common::KeyFrame* keyframe_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetNewKeyFrame(keyframe_ptr);
    }
}

void Interface::GetGtStatus(std::deque<common::OdomData>* gt_status_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetGtStatus(gt_status_ptr);
    }
}

void Interface::GetLiveScan(common::EigenVector3dVec* live_scan_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetLiveScan(live_scan_ptr);
    }
}

void Interface::GetLiveCloud(common::EigenVector4dVec* live_cloud_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetLiveCloud(live_cloud_ptr);
    }
}

void Interface::GetRelocLandmarks(common::EigenVector3dVec* reloc_landmarks_ptr) {
     std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetRelocLandmarks(reloc_landmarks_ptr);
    }
}

void Interface::GetVizEdges(common::EdgeVec* viz_edge_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetVizEdges(viz_edge_ptr);
    }
}

void Interface::GetTGtoM(aslam::Transformation* T_GtoM_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) { 
        vins_handler_ptr_->GetTGtoM(T_GtoM_ptr);
    }
}

void Interface::GetTOtoC(aslam::Transformation* T_OtoC_ptr, size_t id) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetTOtoC(T_OtoC_ptr, id);
    }
}

void Interface::GetTOtoS(aslam::Transformation* T_OtoS_ptr) {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetTOtoS(T_OtoS_ptr);
    }
}

bool Interface::GetShowImage(common::CvMatConstPtrVec* imgs_ptr) const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetShowImage(imgs_ptr);
        if (imgs_ptr != nullptr && !(imgs_ptr->empty())) {
            return true;
        }
        return false;
    }
    return false;
}


bool Interface::GetLastOdom(common::OdomData& odom) {
    if (interface_manager_ptr_ != nullptr) {
        return interface_manager_ptr_->GetLastOdom(odom);
    }
    return false;
}


bool Interface::HasPoseGraphComplete() const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        return vins_handler_ptr_->HasOfflinePoseGraphComplete();
    }
    return false;
}

bool Interface::HasMappingComplete() const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        return vins_handler_ptr_->HasMappingComplete();
    }
    return false;
}

void Interface::GetPGPoses(common::EigenMatrix4dVec* poses_ptr) const {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->GetPGPoses(poses_ptr);
    }
}

void Interface::SetAllFinish() {
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->SetAllFinish();
    }
}

SLAM_STATUS Interface::GetSlamStatus() const {
    return slam_status_;
}
}
