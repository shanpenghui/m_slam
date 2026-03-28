cmake_minimum_required(VERSION 3.5)
project(mirobot_msgs)
cmake_policy(SET CMP0057 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})


find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
    std_msgs
    nav_msgs
    geometry_msgs
    sensor_msgs
)
rosidl_generate_interfaces(${PROJECT_NAME}
	"srv/ArmAction.srv"
	"srv/NavAction.srv"
	"srv/ChassisCmd.srv"
	"srv/ExploreMapAction.srv"
	"srv/LineLidarAction.srv"
	"srv/MapManagerAction.srv"
	"srv/SlamService.srv"
	"srv/Xd1aAction.srv"
	"srv/YdlidarRosAction.srv"
	"srv/IniState.srv"
	"srv/ParaSetting.srv"
	"srv/ScoreTrajectory.srv"
	"srv/SemanticNavi.srv"
	"msg/android/NavStatus.msg"
	"msg/android/AndroidCmd.msg"
	"msg/android/AndroidResponse.msg"
	"msg/android/ErrorCode.msg"
	"msg/android/LogData.msg"
	"msg/android/SmachRequest.msg"
	"msg/android/SmachResponse.msg"
	"msg/behavior/BehaviorPara.msg"
	"msg/behavior/BehaviorState.msg"
	"msg/chassis/BatteryDetail.msg"
	"msg/chassis/ChassisDetail.msg"
	"msg/chassis/HeadDetail.msg"
	"msg/chassis/PowerBoardDetail.msg"
	"msg/chassis/HardWareDetail.msg"
	"msg/stellar/CameraStatus.msg"
	"msg/planning/ObstacleType.msg"
	"msg/planning/ObstacleTypeStamp.msg"
	"msg/planning/ParameterSetting.msg"
	"msg/planning/PoseState.msg"
	"msg/planning/TwistAccel.msg"
	"msg/planning/TrajectoryScore.msg"
	"msg/planning/Trajectory2D.msg"
	"msg/planning/CriticScore.msg"
	"msg/visual/CameraPeoplePose.msg"
	"action/GlobalPlanner.action"
	"action/LocalPlanner.action"
	"action/VelocityPlanner.action"
	DEPENDENCIES ${dependencies}
)

#rosidl_dependencies(${PROJECT_NAME} ${dependencies})

ament_auto_package ()
