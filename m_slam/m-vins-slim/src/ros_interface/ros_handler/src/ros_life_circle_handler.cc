#include "ros_handler/ros_life_circle_handler.h"

#include <thread>
#include <cv_bridge/cv_bridge.h>
#ifdef USE_ROS2
#include <rclcpp/rclcpp.hpp>
#else
#include <ros/ros.h>
#endif

#include "interface/interface.h"
#include "math_common/nano_ekf.h"
#include "ros_handler/rosbag_parsing_handler.h"

namespace mvins {

RosTime CvtToRosTime(const uint64_t time_ns) {
#ifdef USE_ROS2
    return RosTime(time_ns);
#else
    return RosTime(common::NanoSecondsToSeconds(time_ns));
#endif
}

void StartRosLifeCircle(const int sleep_time_ms,
    const common::SlamConfigPtr& config,
    const TopicMap& topic_names,
    const std::shared_ptr<NodeHandle>& node_handler_ptr,
    Interface* interface_ptr,
    TopicSubscriberOptions* subscriber_options_ptr,
    TopicSubscribers* topic_subscribers_ptr,
    TopicPublishers* topic_publishers_ptr,
    ServiceOptions* service_options_ptr,
    Services* services_ptr,
    bool* received_end_signal_ptr) {
    Interface& interface = *CHECK_NOTNULL(interface_ptr);
    TopicSubscriberOptions& subscriber_options = *CHECK_NOTNULL(subscriber_options_ptr);
    TopicSubscribers& topic_subscribers = *CHECK_NOTNULL(topic_subscribers_ptr);
    TopicPublishers& topic_publishers = *CHECK_NOTNULL(topic_publishers_ptr);
    ServiceOptions& service_options = *CHECK_NOTNULL(service_options_ptr);
    Services& services = *CHECK_NOTNULL(services_ptr);

    std::unique_ptr<mvins::RosbagParser> rosbag_parser;
    mvins::InitializePublishers(topic_names,
                                node_handler_ptr.get(),
                                &topic_publishers);
    if (config->online) {
    mvins::InitializeSubscriberOptions(node_handler_ptr.get(),
                                       &subscriber_options);
    mvins::InitializeServiceOptions(node_handler_ptr.get(),
                                    &service_options);
    mvins::InitializeServices(node_handler_ptr.get(),
                              &interface,
                              &service_options,
                              &services);
    } else {
        rosbag_parser.reset(new mvins::RosbagParser(config->data_path,
                                                    config->data_realtime_playback_rate,
                                                    topic_names,
                                                    &interface));
        rosbag_parser->Start();       
    }

#ifdef USE_ROS2
    using rclcpp::executors::MultiThreadedExecutor;
    MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 6/*number_of_threads*/);
    executor.add_node(node_handler_ptr);
    std::thread spinner(std::bind(&MultiThreadedExecutor::spin, &executor));
    spinner.detach();
    rclcpp::Rate loop_rate(1000 / sleep_time_ms);
#else
    ros::AsyncSpinner spinner(4);
    ros::Rate loop_rate(1000 / sleep_time_ms);
    spinner.start();
#endif

#ifdef PUB_DEBUG_INFO
    // The size of trajectory queue for visualization.
    constexpr size_t kTrajectoryVisualizationQueueSize = 100u;
#endif

    // Path of local tracking.
    PathMsg path, gt_path;
    int gt_count = 0;
    Eigen::Vector3d mean_error;

    constexpr int kMapPubFrequency = 1;
    const double map_pub_time_diff_s = 1.0 / static_cast<double>(kMapPubFrequency);

    common::OdomData last_keyframe_odom;
    aslam::Transformation last_keyframe_T_OtoG;
 
    size_t loop_count = 0u;
    double last_pub_map_time_s = 0.0;
    bool subscribers_inited = false;
    std::unique_ptr<aslam::Transformation> T_OtoG_newest = nullptr;

    uint64_t last_pose_time_ns = 0u;

    std::unique_ptr<common::NanoEKF> nano_ekf_ptr = nullptr;

    QuaternionMsg q_local_base_msg;
    PointMsg p_local_base_msg;
    Vector3Msg linear_velocity, angular_velocity;

    QuaternionMsg q_map_base_msg;
    PointMsg p_map_base_msg;

    const std::string odom_frame = "odom";
    const std::string map_frame = "map";
    const std::string base_link_frame = "base_link";

#ifdef USE_ROS2
    while (rclcpp::ok()) {
#else
    while (ros::ok()) {
#endif
        if (config->online) {
            mvins::SLAM_STATUS current_slam_status = interface.GetSlamStatus();
            if (current_slam_status == SLAM_STATUS::RUNNING && !subscribers_inited) {
                mvins::InitializeSubscribers(topic_names,
                                            node_handler_ptr.get(),
                                            &interface,
                                            &subscriber_options,
                                            &topic_subscribers);
                subscribers_inited = true;
            } else if (current_slam_status == SLAM_STATUS::IDLE && subscribers_inited) {
                mvins::ShutdownSubscribers(&topic_subscribers, ShutDownType::EXCEPT_ODOM);
                subscribers_inited = false;
            }
        }
        
        if (*received_end_signal_ptr == true && !interface.IsAllFinished()) {
            if (interface.GetSlamStatus() == SLAM_STATUS::IDLE) {
                break;
            } else {
                LOG(INFO) << "Received end signal. trying to exit...";
                interface.SetAllFinish();
            }
        }

        if (interface.IsAllFinished()) {
            VLOG(0) << "All processing done, exit.";
            break;
        }

        if (interface.GetSlamStatus() == SLAM_STATUS::STOP) {
            LOG(INFO) << "Slam status is stop, exit.";
            break;
        }

        if (interface.GetSlamStatus() == SLAM_STATUS::IDLE && config->online) {
            if (T_OtoG_newest != nullptr) {
                T_OtoG_newest.reset(nullptr);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
            continue;
        }
#ifdef USE_ROS2
        RosTime t_now = rclcpp::Clock().now();
        const double t_now_s = t_now.seconds();
#else
        RosTime t_now = ros::Time::now();
        const double t_now_s = ros::Time::now().toSec();
#endif
        const bool pub_map = (t_now_s - last_pub_map_time_s) > map_pub_time_diff_s;

        aslam::Transformation T_OtoC; 
        interface.GetTOtoC(&T_OtoC, 0);
        aslam::Transformation T_CtoO = T_OtoC.inverse();

        aslam::Transformation T_OtoS;
        interface.GetTOtoS(&T_OtoS);
        aslam::Transformation T_StoO = T_OtoS.inverse();

        aslam::Transformation T_GtoM;
        interface.GetTGtoM(&T_GtoM);

        // If there is a new local tracking pose.
        if (interface.IsNewPose()) {
            common::KeyFrame keyframe;
            interface.GetNewKeyFrame(&keyframe);
            const common::State& state = keyframe.state;
            const common::OdomDatas& odom_datas = keyframe.sensor_meas.odom_datas;
            last_keyframe_odom = odom_datas.back();
            last_keyframe_T_OtoG = state.T_OtoG;
            last_pose_time_ns = state.timestamp_ns;
            T_OtoG_newest = std::make_unique<aslam::Transformation>(last_keyframe_T_OtoG);
            if (config->online) {
                common::OdomData newest_odom;
                if (interface.GetLastOdom(newest_odom)) {
                    if (newest_odom.timestamp_ns > last_pose_time_ns) {
                        const aslam::Transformation T_k(last_keyframe_odom.q, last_keyframe_odom.p);
                        const aslam::Transformation T_kp1(newest_odom.q, newest_odom.p);
                        const aslam::Transformation delta_T = T_k.inverse() * T_kp1;
                        *T_OtoG_newest = last_keyframe_T_OtoG * delta_T;
                        last_pose_time_ns = newest_odom.timestamp_ns;
                    }
                }
            }

#ifdef PUB_DEBUG_INFO
            const aslam::Transformation T_OtoM =
                    T_GtoM * (*T_OtoG_newest);

            // Publish camera pose.
            const aslam::Transformation T_CtoM =
                    T_OtoM * T_OtoC.inverse();
            QuaternionMsg q_map_cam_msg;
            PointMsg p_map_cam_msg;
            const Eigen::Vector3d& p_cam = T_CtoM.getPosition();
            p_map_cam_msg.x = p_cam(0);
            p_map_cam_msg.y = p_cam(1);
            p_map_cam_msg.z = p_cam(2);
            const Eigen::Quaterniond& q_cam = T_CtoM.getEigenQuaternion();
            q_map_cam_msg.w = q_cam.w();
            q_map_cam_msg.x = q_cam.x();
            q_map_cam_msg.y = q_cam.y();
            q_map_cam_msg.z = q_cam.z();

            PoseStampedMsg this_pose_cam_stamped;
            this_pose_cam_stamped.header.stamp = t_now;
            this_pose_cam_stamped.header.frame_id = map_frame;
            this_pose_cam_stamped.pose.orientation = q_map_cam_msg;
            this_pose_cam_stamped.pose.position = p_map_cam_msg;

            // TODO: Set covariance.
#ifdef USE_ROS2
            if (topic_publishers.pose_cam_pub_->get_subscription_count() > 0u) {
#else
            if (topic_publishers.pose_cam_pub_->getNumSubscribers() > 0) {
#endif
                topic_publishers.pose_cam_pub_->publish(this_pose_cam_stamped);
                    }

            // Publish detected visual features.
            common::CvMatConstPtrVec show_imgs;
            if (interface.GetShowImage(&show_imgs)) {
                ImageMsgPtr feature_tracking_msg = cv_bridge::CvImage(
                            HeaderMsg(), "bgr8", *(show_imgs[0])).toImageMsg();
#ifdef USE_ROS2
                if (topic_publishers.feature_tracking_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.feature_tracking_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.feature_tracking_pub_->publish(*feature_tracking_msg);
                }
                ImageMsgPtr feature_depth_msg = cv_bridge::CvImage(
                            HeaderMsg(), "bgr8", *(show_imgs[1])).toImageMsg();
#ifdef USE_ROS2
                if (topic_publishers.feature_depth_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.feature_depth_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.feature_depth_pub_->publish(*feature_depth_msg);
                }
            }

            common::EigenVector4dVec live_cloud_vec;
            interface.GetLiveCloud(&live_cloud_vec);
            if (!live_cloud_vec.empty()) {
                PointCloudMsg live_cloud;
                HeaderMsg header;
                header.stamp = t_now;
                header.frame_id = map_frame;

                live_cloud.header = header;
                live_cloud.height = 1;
                live_cloud.width = live_cloud_vec.size();
                live_cloud.fields.resize(3);
                live_cloud.fields[0].name = "x";
                live_cloud.fields[0].offset = 0;
                live_cloud.fields[0].datatype = PointFieldMsg::FLOAT32;
                live_cloud.fields[0].count = 1;
                live_cloud.fields[1].name = "y";
                live_cloud.fields[1].offset = 4;
                live_cloud.fields[1].datatype = PointFieldMsg::FLOAT32;
                live_cloud.fields[1].count = 1;
                live_cloud.fields[2].name = "z";
                live_cloud.fields[2].offset = 8;
                live_cloud.fields[2].datatype = PointFieldMsg::FLOAT32;
                live_cloud.fields[2].count = 1;
                live_cloud.point_step = 12;
                live_cloud.row_step = live_cloud.point_step * live_cloud.width;
                live_cloud.is_dense = true;
                live_cloud.data.resize(live_cloud.row_step * live_cloud.height);
                for (size_t idx = 0u; idx < live_cloud_vec.size(); ++idx) {
                    float x = live_cloud_vec[idx](0);
                    float y = live_cloud_vec[idx](1);
                    float z = live_cloud_vec[idx](2);
                    memcpy(&live_cloud.data[idx * live_cloud.point_step], &x, sizeof(float));
                    memcpy(&live_cloud.data[idx * live_cloud.point_step + 4], &y, sizeof(float));
                    memcpy(&live_cloud.data[idx * live_cloud.point_step + 8], &z, sizeof(float));
                }
#ifdef USE_ROS2
                if (topic_publishers.live_cloud_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.live_cloud_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.live_cloud_pub_->publish(live_cloud);
                }

                MarkerMsg obs_msg;
                for (auto& p3d : live_cloud_vec) {
                    if (p3d(3) == 1) {
                        PointMsg p;
                        p.x = p3d(0);
                        p.y = p3d(1);
                        p.z = p3d(2);
                        obs_msg.points.emplace_back(p);
                        obs_msg.points.emplace_back(p_map_cam_msg);
                    }
                }
                obs_msg.header.frame_id = map_frame;
                obs_msg.header.stamp = t_now;
                obs_msg.ns = "observation";
                obs_msg.action = MarkerMsg::ADD;
                obs_msg.pose.orientation.w = 1.0;
                obs_msg.type = MarkerMsg::LINE_LIST;
                obs_msg.scale.x = obs_msg.scale.y = obs_msg.scale.z = 0.01;
                obs_msg.color.b = obs_msg.color.a = 1.0;
#ifdef USE_ROS2
                if (topic_publishers.obs_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.obs_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.obs_pub_->publish(obs_msg);
                }
            }

            common::EigenVector3dVec reloc_landmarks_vec;
            interface.GetRelocLandmarks(&reloc_landmarks_vec);
            if (!reloc_landmarks_vec.empty()) {
                MarkerMsg obs_msg;
                for (auto& p3d : reloc_landmarks_vec) {
                    PointMsg p;
                    p.x = p3d(0);
                    p.y = p3d(1);
                    p.z = p3d(2);
                    obs_msg.points.emplace_back(p);
                    obs_msg.points.emplace_back(p_map_cam_msg);
                }
                obs_msg.header.frame_id = map_frame;
                obs_msg.header.stamp = t_now;
                obs_msg.ns = "observation";
                obs_msg.action = MarkerMsg::ADD;
                obs_msg.pose.orientation.w = 1.0;
                obs_msg.type = MarkerMsg::LINE_LIST;
                obs_msg.scale.x = obs_msg.scale.y = obs_msg.scale.z = 0.01;
                obs_msg.color.r = obs_msg.color.g = obs_msg.color.a = 1.0;
#ifdef USE_ROS2
                if (topic_publishers.reloc_obs_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.reloc_obs_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.reloc_obs_pub_->publish(obs_msg);
                }
            }

            std::deque<common::OdomData> gt_status;
            interface.GetGtStatus(&gt_status);
            if (!gt_status.empty()) {
                const uint64_t curr_state_timestamp_ns = state.timestamp_ns;
                common::OdomData gt_state;
                const bool get_success = common::GetGtStateByTimeNs(curr_state_timestamp_ns,
                                                                    gt_status,
                                                                    &gt_state);
                if (get_success) {
                    gt_count++;
                    QuaternionMsg q_gt_msg;
                    PointMsg p_gt_msg;
                    q_gt_msg.x = gt_state.q.x();
                    q_gt_msg.y = gt_state.q.y();
                    q_gt_msg.z = gt_state.q.x();
                    q_gt_msg.w = gt_state.q.z();
                    p_gt_msg.x = gt_state.p(0);
                    p_gt_msg.y = gt_state.p(1);
                    p_gt_msg.z = gt_state.p(2);

                    PoseStampedMsg gt_pose_stamped;
                    gt_pose_stamped.header.stamp = t_now;
                    gt_pose_stamped.header.frame_id = map_frame;
                    gt_pose_stamped.pose.orientation = q_gt_msg;
                    gt_pose_stamped.pose.position = p_gt_msg;
                    gt_path.poses.push_back(gt_pose_stamped);

                    if (config->online && (gt_path.poses.size() >
                        kTrajectoryVisualizationQueueSize)) {
                        gt_path.poses.erase(gt_path.poses.begin());
                    }

                    gt_path.header.stamp = t_now;
                    gt_path.header.frame_id = map_frame;
#ifdef USE_ROS2
                    if (topic_publishers.gt_path_pub_->get_subscription_count() > 0u) {
#else
                    if (topic_publishers.gt_path_pub_->getNumSubscribers() > 0) {
#endif
                        topic_publishers.gt_path_pub_->publish(gt_path); 
                    }
                }
            }
#endif
        } else if (config->online && T_OtoG_newest != nullptr) {
            common::OdomData newest_odom;
            if (interface.GetLastOdom(newest_odom)) {
                if (newest_odom.timestamp_ns > last_pose_time_ns) {
                    const aslam::Transformation T_k(last_keyframe_odom.q, last_keyframe_odom.p);
                    const aslam::Transformation T_kp1(newest_odom.q, newest_odom.p);
                    const aslam::Transformation delta_T = T_k.inverse() * T_kp1;
                    *T_OtoG_newest = last_keyframe_T_OtoG * delta_T;
                    last_pose_time_ns = newest_odom.timestamp_ns;
                }
            }
        }

        if (T_OtoG_newest == nullptr) {
            loop_rate.sleep();
            continue;
        }
        
        p_local_base_msg.x = T_OtoG_newest->getPosition().x();
        p_local_base_msg.y = T_OtoG_newest->getPosition().y();
        p_local_base_msg.z = T_OtoG_newest->getPosition().z();
        q_local_base_msg.w = T_OtoG_newest->getEigenQuaternion().w();
        q_local_base_msg.x = T_OtoG_newest->getEigenQuaternion().x();
        q_local_base_msg.y = T_OtoG_newest->getEigenQuaternion().y();
        q_local_base_msg.z = T_OtoG_newest->getEigenQuaternion().z();

        OdometryMsg pose_local;
        pose_local.header.stamp = CvtToRosTime(last_pose_time_ns);
        pose_local.header.frame_id = base_link_frame;
        pose_local.pose.pose.orientation = q_local_base_msg;
        pose_local.pose.pose.position = p_local_base_msg;
#ifdef USE_ROS2
        if (topic_publishers.pose_local_pub_->get_subscription_count() > 0u) {
#else
        if (topic_publishers.pose_local_pub_->getNumSubscribers() > 0) {
#endif
            topic_publishers.pose_local_pub_->publish(pose_local);
        }

        // Publish base pose and path.
        const aslam::Transformation T_OtoM = T_GtoM * (*T_OtoG_newest);

#if 1
        aslam::Transformation T_OtoM_smooth;
        if (nano_ekf_ptr == nullptr) {
            nano_ekf_ptr.reset(new common::NanoEKF(T_OtoM));
            T_OtoM_smooth = T_OtoM;
        } else {
            nano_ekf_ptr->Predict();
            T_OtoM_smooth = nano_ekf_ptr->Update(T_OtoM,
                                                Eigen::Matrix<double, 6, 6>::Identity());
        }
#else
        const aslam::Transformation T_OtoM_smooth = T_OtoM;
#endif

        p_map_base_msg.x = T_OtoM_smooth.getPosition().x();
        p_map_base_msg.y = T_OtoM_smooth.getPosition().y();
        p_map_base_msg.z = T_OtoM_smooth.getPosition().z();
        q_map_base_msg.w = T_OtoM_smooth.getEigenQuaternion().w();
        q_map_base_msg.x = T_OtoM_smooth.getEigenQuaternion().x();
        q_map_base_msg.y = T_OtoM_smooth.getEigenQuaternion().y();
        q_map_base_msg.z = T_OtoM_smooth.getEigenQuaternion().z();

        PoseStampedMsg pose_map;
        pose_map.header.stamp = CvtToRosTime(last_pose_time_ns);
        pose_map.header.frame_id = map_frame;
        pose_map.pose.orientation = q_map_base_msg;
        pose_map.pose.position = p_map_base_msg;
#ifdef USE_ROS2
        if (topic_publishers.pose_pub_->get_subscription_count() > 0u) {
#else
        if (topic_publishers.pose_pub_->getNumSubscribers() > 0) {
#endif
            topic_publishers.pose_pub_->publish(pose_map);
        }

        // Pub TF.
        TransformStampedMsg odom_tf;
        odom_tf.header.stamp = CvtToRosTime(last_pose_time_ns);
        odom_tf.header.frame_id = odom_frame;
        odom_tf.child_frame_id = base_link_frame;
        odom_tf.transform.rotation.w = T_OtoG_newest->getEigenQuaternion().w();
        odom_tf.transform.rotation.x = T_OtoG_newest->getEigenQuaternion().x();
        odom_tf.transform.rotation.y = T_OtoG_newest->getEigenQuaternion().y();
        odom_tf.transform.rotation.z = T_OtoG_newest->getEigenQuaternion().z();
        odom_tf.transform.translation.x = T_OtoG_newest->getPosition().x();
        odom_tf.transform.translation.y = T_OtoG_newest->getPosition().y();
        odom_tf.transform.translation.z = T_OtoG_newest->getPosition().z();
        topic_publishers.tf_broadcaster_->sendTransform(odom_tf);

        TransformStampedMsg map_tf;
        map_tf.header.stamp = CvtToRosTime(last_pose_time_ns);
        map_tf.header.frame_id = map_frame;
        map_tf.child_frame_id = odom_frame;
        map_tf.transform.rotation.w = T_GtoM.getEigenQuaternion().w();
        map_tf.transform.rotation.x = T_GtoM.getEigenQuaternion().x();
        map_tf.transform.rotation.y = T_GtoM.getEigenQuaternion().y();
        map_tf.transform.rotation.z = T_GtoM.getEigenQuaternion().z();
        map_tf.transform.translation.x = T_GtoM.getPosition().x();
        map_tf.transform.translation.y = T_GtoM.getPosition().y();
        map_tf.transform.translation.z = T_GtoM.getPosition().z();
        topic_publishers.tf_broadcaster_->sendTransform(map_tf);

        #ifdef PUB_DEBUG_INFO
        path.poses.push_back(pose_map);
        if (config->online && (path.poses.size() >
            kTrajectoryVisualizationQueueSize)) {
            path.poses.erase(path.poses.begin());
        }

        path.header.stamp = t_now;
        path.header.frame_id = map_frame;
#ifdef USE_ROS2
        if (topic_publishers.path_pub_->get_subscription_count() > 0u) {
#else
        if (topic_publishers.path_pub_->getNumSubscribers() > 0) {
#endif
            topic_publishers.path_pub_->publish(path);
        }

        common::EdgeVec edge_vec;
        interface.GetVizEdges(&edge_vec);
        if (!edge_vec.empty()) {
            MarkerMsg edge_msg;
            for (auto& edge : edge_vec) {
                PointMsg a, b;
                a.x = edge.first(0);
                a.y = edge.first(1);
                a.z = edge.first(2);
                b.x = edge.second(0);
                b.y = edge.second(1);
                b.z = edge.second(2);
                edge_msg.points.emplace_back(a);
                edge_msg.points.emplace_back(b);
            }
            edge_msg.header.frame_id = map_frame;
            edge_msg.header.stamp = t_now;
            edge_msg.ns = "edge";
            edge_msg.action = MarkerMsg::ADD;
            edge_msg.pose.orientation.w = 1.0;
            edge_msg.type = MarkerMsg::LINE_LIST;
            edge_msg.scale.x = edge_msg.scale.y = edge_msg.scale.z = 0.01;
            edge_msg.color.r = edge_msg.color.g = edge_msg.color.a = 1.0;
#ifdef USE_ROS2
            if (topic_publishers.edge_pub_->get_subscription_count() > 0u) {
#else
            if (topic_publishers.edge_pub_->getNumSubscribers() > 0) {
#endif
                topic_publishers.edge_pub_->publish(edge_msg);
            }
        }

        common::EigenMatrix4dVec pg_poses;
        interface.GetPGPoses(&pg_poses);
        if (!pg_poses.empty()) {
            path.poses.clear();
            for (size_t i = 0u; i < pg_poses.size(); ++i) {
                const Eigen::Quaterniond q(pg_poses[i].block<3, 3>(0, 0));
                PoseStampedMsg this_pose;
                this_pose.pose.orientation.x = q.x();
                this_pose.pose.orientation.y = q.y();
                this_pose.pose.orientation.z = q.z();
                this_pose.pose.orientation.w = q.w();
                this_pose.pose.position.x = pg_poses[i](0, 3);
                this_pose.pose.position.y = pg_poses[i](1, 3);
                this_pose.pose.position.z = pg_poses[i](2, 3);
                path.poses.push_back(this_pose);
            }
#ifdef USE_ROS2
            if (topic_publishers.path_pub_->get_subscription_count() > 0u) {
#else
            if (topic_publishers.path_pub_->getNumSubscribers() > 0) {
#endif
                topic_publishers.path_pub_->publish(path);
            }
        }

        common::LoopResult loop_result;
        if (interface.GetNewLoop(&loop_result)) {
            QuaternionMsg q_map_base_msg;
            PointMsg p_map_base_msg;
            p_map_base_msg.x = loop_result.T_estimate.getPosition()(0);
            p_map_base_msg.y = loop_result.T_estimate.getPosition()(1);
            p_map_base_msg.z = loop_result.T_estimate.getPosition()(2);
            const Eigen::Quaterniond& q_base = loop_result.T_estimate.getEigenQuaternion();
            q_map_base_msg.w = q_base.w();
            q_map_base_msg.x = q_base.x();
            q_map_base_msg.y = q_base.y();
            q_map_base_msg.z = q_base.z();

            PoseStampedMsg this_pose_base_stamped;
            this_pose_base_stamped.header.stamp = t_now;
            this_pose_base_stamped.header.frame_id = map_frame;
            this_pose_base_stamped.pose.orientation = q_map_base_msg;
            this_pose_base_stamped.pose.position = p_map_base_msg;
            // TODO: Set covariance.
#ifdef USE_ROS2
            if (topic_publishers.pose_loop_pub_->get_subscription_count() > 0u) {
#else
            if (topic_publishers.pose_loop_pub_->getNumSubscribers() > 0) {
#endif
                topic_publishers.pose_loop_pub_->publish(this_pose_base_stamped);
            }
        }
#endif

        common::EigenVector3dVec live_scan_vec;
        interface.GetLiveScan(&live_scan_vec);
        if (!live_scan_vec.empty()) {
            PointCloudMsg live_scan;
            HeaderMsg header;
            header.stamp = CvtToRosTime(last_pose_time_ns);
            header.frame_id = map_frame;

            live_scan.header = header;
            live_scan.height = 1;
            live_scan.width = live_scan_vec.size();
            live_scan.fields.resize(3);
            live_scan.fields[0].name = "x";
            live_scan.fields[0].offset = 0;
            live_scan.fields[0].datatype = PointFieldMsg::FLOAT32;
            live_scan.fields[0].count = 1;
            live_scan.fields[1].name = "y";
            live_scan.fields[1].offset = 4;
            live_scan.fields[1].datatype = PointFieldMsg::FLOAT32;
            live_scan.fields[1].count = 1;
            live_scan.fields[2].name = "z";
            live_scan.fields[2].offset = 8;
            live_scan.fields[2].datatype = PointFieldMsg::FLOAT32;
            live_scan.fields[2].count = 1;
            live_scan.point_step = 12;
            live_scan.row_step = live_scan.point_step * live_scan.width;
            live_scan.is_dense = true;
            live_scan.data.resize(live_scan.row_step * live_scan.height);
            for (size_t idx = 0u; idx < live_scan_vec.size(); ++idx) {
                float x = live_scan_vec[idx](0);
                float y = live_scan_vec[idx](1);
                float z = live_scan_vec[idx](2);
                memcpy(&live_scan.data[idx * live_scan.point_step], &x, sizeof(float));
                memcpy(&live_scan.data[idx * live_scan.point_step + 4], &y, sizeof(float));
                memcpy(&live_scan.data[idx * live_scan.point_step + 8], &z, sizeof(float));
            }
#ifdef USE_ROS2
            if (topic_publishers.scan_cloud_pub_->get_subscription_count() > 0u) {
#else
            if (topic_publishers.scan_cloud_pub_->getNumSubscribers() > 0) {
#endif
                topic_publishers.scan_cloud_pub_->publish(live_scan);
            }
        }

        cv::Mat map;
        Eigen::Vector2d origin;
        common::EigenVector3dVec map_cloud_vec;
        if (pub_map && interface.GetNewMap(&map, &origin, &map_cloud_vec)) {
            // Pub occ map.
            if (!map.empty()) {
                CHECK_EQ(map.type(), CV_8UC1);
                cv::transpose(map, map);
                cv::flip(map, map, 0);
                cv::flip(map, map, 2);

                OccupancyGridMsg occ_msg;
                MapMetaDataMsg meta_msg;
                occ_msg.header.frame_id = map_frame;
                occ_msg.header.stamp = t_now;
                meta_msg.map_load_time = t_now;
                meta_msg.width = map.cols;
                meta_msg.height = map.rows;
                meta_msg.resolution = static_cast<float>(config->resolution);
                meta_msg.origin.position.x = origin(0);
                meta_msg.origin.position.y = origin(1);
                meta_msg.origin.position.z = 0.0;
                const Eigen::Quaterniond origin_q = Eigen::Quaterniond::Identity();
                meta_msg.origin.orientation.x = origin_q.x();
                meta_msg.origin.orientation.y = origin_q.y();
                meta_msg.origin.orientation.z = origin_q.z();
                meta_msg.origin.orientation.w = origin_q.w();
                occ_msg.info = meta_msg;

                const int map_size = meta_msg.width * meta_msg.height;
                occ_msg.data.resize(map_size);
                int idx = 0;
                for (int i = 0; i < map.rows; ++i) {
                    for (int j = 0; j < map.cols; ++j) {
                        occ_msg.data[idx++] = map.at<uchar>(i,j);
                    }
                }
#ifdef USE_ROS2
                if (topic_publishers.map_pub_->get_subscription_count() > 0u) {
#else
                if (topic_publishers.map_pub_->getNumSubscribers() > 0) {
#endif
                    topic_publishers.map_pub_->publish(occ_msg);
                }
            }
            // Pub visual map.
            if (!map_cloud_vec.empty()) {
                PointCloudMsg pc;
                HeaderMsg header;
                header.stamp = t_now;
                header.frame_id = map_frame;

                pc.header = header;
                pc.header = header;
                pc.height = 1;
                pc.width = map_cloud_vec.size();
                pc.fields.resize(3);
                pc.fields[0].name = "x";
                pc.fields[0].offset = 0;
                pc.fields[0].datatype = PointFieldMsg::FLOAT32;
                pc.fields[0].count = 1;
                pc.fields[1].name = "y";
                pc.fields[1].offset = 4;
                pc.fields[1].datatype = PointFieldMsg::FLOAT32;
                pc.fields[1].count = 1;
                pc.fields[2].name = "z";
                pc.fields[2].offset = 8;
                pc.fields[2].datatype = PointFieldMsg::FLOAT32;
                pc.fields[2].count = 1;
                pc.point_step = 12;
                pc.row_step = pc.point_step * pc.width;
                pc.is_dense = true;
                pc.data.resize(pc.row_step * pc.height);
                for (size_t idx = 0u; idx < map_cloud_vec.size(); ++idx) {
                    float x = map_cloud_vec[idx](0);
                    float y = map_cloud_vec[idx](1);
                    float z = map_cloud_vec[idx](2);
                    memcpy(&pc.data[idx * pc.point_step], &x, sizeof(float));
                    memcpy(&pc.data[idx * pc.point_step + 4], &y, sizeof(float));
                    memcpy(&pc.data[idx * pc.point_step + 8], &z, sizeof(float));
                }
    #ifdef USE_ROS2
                if (topic_publishers.map_cloud_pub_->get_subscription_count() > 0u) {
    #else
                if (topic_publishers.map_cloud_pub_->getNumSubscribers() > 0) {
    #endif
                    topic_publishers.map_cloud_pub_->publish(pc);   
                }
            }
            last_pub_map_time_s = t_now_s;
        }
        
        loop_rate.sleep();
        ++loop_count; 
    }

#ifdef USE_ROS2
    rclcpp::shutdown();
#else
    spinner.stop();
#endif

    mvins::ShutdownPublishers(&topic_publishers);
    if (config->online) {
        if (subscribers_inited) {
            mvins::ShutdownSubscribers(&topic_subscribers, ShutDownType::ALL);
        }
        mvins::ShutdownServices(&services);
    }
}
}