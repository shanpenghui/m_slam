#include "interface/interface.h"

namespace mvins {
    
void Interface::ResetPose(ServiceResponse* res_ref_ptr) {
    ServiceResponse& res_ref = *CHECK_NOTNULL(res_ref_ptr);
    if (slam_status_ == SLAM_STATUS::IDLE) {
        LOG(WARNING) << "SLAM is in IDLE, ignored.";
        res_ref.feedback = 0;
        return;
    }

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr && !interface_manager_ptr_->IsDockerPosePtrNull()) {
        vins_handler_ptr_->SetResetPose(*interface_manager_ptr_->GetDockerPosePtr(),
                                        false/*force_reset*/);
        res_ref.feedback = 1;
    } else {
        res_ref.feedback = 0;
    }
}

void Interface::ShutDown() {
    AddEndSignal();
    SetAllFinish();
    slam_status_mutex_.lock();
    slam_status_ = SLAM_STATUS::STOP;
    slam_status_mutex_.unlock();
}

void Interface::SlamStart(ServiceResponse* res_ref_ptr) {
    ServiceResponse& res_ref = *CHECK_NOTNULL(res_ref_ptr);
    if (slam_status_ == SLAM_STATUS::RUNNING) {
        LOG(WARNING) << "SLAM is already running, ignored.";
        res_ref.feedback = 1;
        return;
    }

    const bool have_map = CheckMapExistence();
    if (have_map) {   
        ResetConfigAndVinsHandlerPtr(SLAM_MODE::RELOC);
    } else {
        ResetConfigAndVinsHandlerPtr(SLAM_MODE::MAPPING);
    }
    slam_status_mutex_.lock();
    slam_status_ = SLAM_STATUS::RUNNING;
    slam_status_mutex_.unlock();
    res_ref.feedback = 1;
}

void Interface::SlamIdle() {
    if (slam_status_ == SLAM_STATUS::IDLE) {
        LOG(WARNING) << "SLAM is already in IDLE, ignored.";
        return;
    } else if (config_->mapping) {
        LOG(WARNING) << "SLAM is MAPPING, ignored.";
        return;
    }

    if (interface_manager_ptr_ != nullptr) {
        interface_manager_ptr_->ResetMovedPosePtr();
        interface_manager_ptr_->SetLastKeyFrameBeforeIdlePtr();
        interface_manager_ptr_->SetLastTGtoMBeforeIdlePtr();
    }

    const bool have_map = CheckMapExistence();
    if (have_map) {
        if (vins_handler_ptr_->HasRelocInitSuccess()) {
            interface_manager_ptr_->SetLastKeyFrameBeforeIdlePtr(vins_handler_ptr_->GetLastKeyFrameCopy());
            interface_manager_ptr_->SetLastTGtoMBeforeIdlePtr(vins_handler_ptr_->GetTGtoMCopy());
            LOG(WARNING) << "Reloc init successed, record last T_OtoM: \n" << 
                vins_handler_ptr_->GetTGtoMCopy() * vins_handler_ptr_->GetLastKeyFrameCopy().state.T_OtoG;
        } else {
            LOG(WARNING) << "Reloc init has not init success, do not record last pose.";
        }
        ResetConfigAndVinsHandlerPtr(SLAM_MODE::IDLE);
    } else {
        ResetConfigAndVinsHandlerPtr(SLAM_MODE::IDLE);
    }

    slam_status_mutex_.lock();
    slam_status_ = SLAM_STATUS::IDLE;
    slam_status_mutex_.unlock();
}

void Interface::SaveMap() {
    if (slam_status_ == SLAM_STATUS::IDLE) {
        LOG(WARNING) << "SLAM is in IDLE, ignored.";
        return;
    }

    if (!config_->mapping) {
        LOG(WARNING) << "SLAM is in RELOC, ignored.";
        return;
    }

    AddEndSignal();
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    while (!vins_handler_ptr_->AllThreadFinished()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    vins_handler_ptr_->Release();
    vins_handler_ptr_.reset(nullptr);

    interface_manager_ptr_->CreateMaskTable();
    LOG(WARNING) << "Create mask database successfully";
    LOG(WARNING) << "Map saved successfully.";
    interface_manager_ptr_->TryLoadMap(config_->map_path);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ResetConfigAndVinsHandlerPtr(SLAM_MODE::RELOC);
}

void Interface::RemoveMap() {
    const bool have_map = CheckMapExistence();
    if (!have_map) {
        LOG(WARNING) << "WARNING: No map exists, ignored.";
        return;
    }
    interface_manager_ptr_mutex_.lock();
    interface_manager_ptr_.reset(nullptr);
    interface_manager_ptr_mutex_.unlock();

    std::vector<std::string> maps;
    common::getAllFilesInFolder(config_->map_path, &maps);
    for (const auto& map : maps) {
        if (common::deleteFile(map)) {
            LOG(INFO) << "Successfully delete " << map;
        } else {
            LOG(ERROR) << "Failed to delete Map: " << map;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (slam_status_ != SLAM_STATUS::IDLE) {
        SlamIdle();
    } else {
        LOG(WARNING) << "SLAM Status change to IDLE.";
    }
}

void Interface::LoadMap() {
    
}

void Interface::OperateMask(const ServiceRequest& req_ref, ServiceResponse* res_ref_ptr) {
    ServiceResponse& res_ref = *CHECK_NOTNULL(res_ref_ptr);
    if (interface_manager_ptr_ == nullptr) {
        LOG(ERROR) << "ERROR: Cannot operate mask without map.";
        res_ref.feedback = 0;
        return;
    } else if (config_->mapping) {
        LOG(ERROR) << "ERROR: Cannot operate mask when mapping.";
        res_ref.feedback = 0;
        return;
    }
    
    std::vector<float> params = req_ref.param;
    std::string params_str = std::accumulate(std::next(params.begin()), params.end(), std::to_string(params[0]),
    [](std::string a, float b) { return std::move(a) + ' ' + std::to_string(b); });
    LOG(INFO) << "Input mask operation params: " << params_str;

    const bool success = interface_manager_ptr_->SaveMaskOperationInMaskDatabase(params);
    if (success) {
        res_ref.feedback = 1;
    } else {
        res_ref.feedback = 0;
    }
}

void Interface::GetDockerPose(ServiceResponse* res_ref_ptr) {
    ServiceResponse& res_ref = *CHECK_NOTNULL(res_ref_ptr);
    if (interface_manager_ptr_->IsDockerPosePtrNull()) {
        LOG(WARNING) << "WARNING: Have not got docker pose";
        res_ref.feedback = 0;
        return;
    }

    const aslam::Transformation& docker_pose = *interface_manager_ptr_->GetDockerPosePtr();
    res_ref.param.push_back(std::to_string(docker_pose.getPosition().x()));
    res_ref.param.push_back(std::to_string(docker_pose.getPosition().y()));
    // Z of docker pose is yaw when save yaml
    res_ref.param.push_back(std::to_string(docker_pose.getPosition().z()));
    res_ref.feedback = 1;
}

bool Interface::CheckMapExistence() {
    std::vector<std::string> maps;
    std::string map_file_name = 
        common::ConcatenateFilePathFrom(config_->map_path, kOccupancyMapFileName);
    std::string map_yaml_name = 
        common::ConcatenateFilePathFrom(config_->map_path, kOccupancyMapYamlFileName);
    std::string mask_file_name = 
        common::ConcatenateFilePathFrom(config_->map_path, common::kMaskFileName);

    common::getAllFilesInFolder(config_->map_path, &maps);

    if (maps.size() == 0) {
        return false;
    } else if (maps.size() == 1 && 
        common::fileExists(mask_file_name)) {
        return false;
    }

    for (const auto& map : maps) {
        if (!common::fileExists(map)) {
            LOG(WARNING) << "File " << map << " does not exist!"; 
            if (map == map_file_name || map == map_yaml_name) {
                return false;
            }
        }
    }
    return true;
}

bool Interface::UseScanPointCloud() {
    return config_->use_scan_pointcloud;
}

void Interface::ResetConfigAndVinsHandlerPtr(const SLAM_MODE& mode) {
    if (mode == SLAM_MODE::MAPPING || mode == SLAM_MODE::RELOC) {
        if (vins_handler_ptr_ != nullptr) {
            LOG(WARNING) << "WARNING: vins core is not nullptr";
            return;
        }
    } else {
        if (vins_handler_ptr_ == nullptr) {
            LOG(WARNING) << "WARNING: In IDLE mode, vins core can not be nullptr!!!";
            return;
        }
    }
    
    switch (mode) {
        case SLAM_MODE::MAPPING: {
            const std::string complete_config_file_name_mapping = 
                common::ConcatenateFilePathFrom(slam_yamls_path_, kConfigMappingFileName);
            config_.reset(new common::SlamConfig(complete_config_file_name_mapping, true));
            interface_manager_ptr_.reset(new InterfaceManager(config_));
            vins_handler_ptr_.reset(new vins_handler::VinsHandler(config_,
                                                                  interface_manager_ptr_->scan_maps_ptr_,
                                                                  interface_manager_ptr_->summary_map_ptr_,
                                                                  interface_manager_ptr_->octree_map_ptr_));
            if (interface_manager_ptr_->IsDockerPosePtrNull()) {
                vins_handler_ptr_->SetDockerPoseCallBack([this](aslam::Transformation docker_pose) {
                    interface_manager_ptr_->ResetDockerPosePtr(docker_pose);
                });
            }
            vins_handler_ptr_->Start();
            LOG(WARNING) << "SLAM state changes to running: MAPPING";
            break;
        }
        case SLAM_MODE::RELOC: {
            const std::string complete_config_file_name_reloc = 
                common::ConcatenateFilePathFrom(slam_yamls_path_, kConfigRelocFileName);
            config_.reset(new common::SlamConfig(complete_config_file_name_reloc, true));
            CHECK_NOTNULL(interface_manager_ptr_);
            vins_handler_ptr_.reset(new vins_handler::VinsHandler(config_,
                                                                  interface_manager_ptr_->scan_maps_ptr_,
                                                                  interface_manager_ptr_->summary_map_ptr_,
                                                                  interface_manager_ptr_->octree_map_ptr_));
            if (interface_manager_ptr_->IsDockerPosePtrNull()) {
                vins_handler_ptr_->SetDockerPoseCallBack([this](aslam::Transformation docker_pose) {
                    interface_manager_ptr_->ResetDockerPosePtr(docker_pose);
                });
            }
            if (interface_manager_ptr_->GetLastKeyFrameBeforeIdlePtr() != nullptr) {
                SetMovedPose();
            } else {
                LOG(WARNING) << "Did not get last keyframe to set moved pose";
            }
            if (!interface_manager_ptr_->IsMovedPosePtrNull()) {
                vins_handler_ptr_->SetResetPose(*interface_manager_ptr_->GetMovedPosePtr(),
                                                true/*force_reset*/);
                LOG(INFO) << "Try init reloc with prior";
            }
            vins_handler_ptr_->Start();
            LOG(WARNING) << "SLAM state changes to running: RELOC";
            break;
        }
        case SLAM_MODE::IDLE: {
            AddEndSignal();
            std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
            vins_handler_ptr_->SetAllFinish();
            while (!vins_handler_ptr_->AllThreadFinished()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            vins_handler_ptr_->Release();
            vins_handler_ptr_.reset(nullptr);
            const std::string complete_config_file_name_idle = 
                common::ConcatenateFilePathFrom(slam_yamls_path_, kConfigIdleFileName);
            config_.reset(new common::SlamConfig(complete_config_file_name_idle, true));
            LOG(WARNING) << "SLAM state changes to IDLE.";
            break;
        }        
        default: {
            LOG(ERROR) << "Unknown SLAM Mode.";
            break;
        }  
    }
}

void Interface::SetMovedPose() {
    CHECK(interface_manager_ptr_->GetLastKeyFrameBeforeIdlePtr() != nullptr);
    CHECK(interface_manager_ptr_->GetLastTGtoMBeforeIdlePtr() != nullptr);

    common::OdomData odom_meas_kp1, last_keyframe_meas;
    
    if (!interface_manager_ptr_->GetLastOdom(odom_meas_kp1)) {
        LOG(ERROR) << "ERROR: No odom added in IDLE";
        return;
    }
    
    last_keyframe_meas = interface_manager_ptr_->GetLastKeyFrameBeforeIdlePtr()->sensor_meas.odom_datas.back();

    aslam::Transformation delta_T, T_OtoM_kp1;

    aslam::Transformation T_kp1_odom(odom_meas_kp1.q, odom_meas_kp1.p);
    aslam::Transformation T_k_odom(last_keyframe_meas.q, last_keyframe_meas.p);
    aslam::Transformation T_OtoG_k = interface_manager_ptr_->GetLastKeyFrameBeforeIdlePtr()->state.T_OtoG;
    aslam::Transformation T_GtoM_k = *interface_manager_ptr_->GetLastTGtoMBeforeIdlePtr();

    aslam::Transformation T_OtoM_k = T_GtoM_k * T_OtoG_k;

    delta_T = T_k_odom.inverse() * T_kp1_odom;
    T_OtoM_kp1 = T_OtoM_k * delta_T;
    interface_manager_ptr_->ResetMovedPosePtr(T_OtoM_kp1);
    LOG(WARNING) << "Set prior pose: \n" << T_OtoM_kp1;
}
}