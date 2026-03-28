set(INTERFACE_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        interface/include
        ros_handler/include
        ${THIRD_LIB_INCLUDE_DIR}
        ${VINS_COMMON_DIR}/include
        ${VINS_CORE_DIR}/include
        ${LOOP_CLOSURE_DIR}/include
        ${OCTOMAP_CORE_DIR}/include
        ${VINS_HANDLER_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)

add_library(ros_interface_${CMAKE_SYSTEM_PROCESSOR} SHARED
        interface/src/interface.cc
        interface/src/image_interface.cc
        interface/src/imu_interface.cc
        interface/src/laser_interface.cc
        interface/src/odom_interface.cc
        interface/src/service_interface.cc
        interface/src/interface_manager.cc
        interface/src/mask_manager.cc
        ros_handler/src/configurate_interface.cc
        ros_handler/src/ros_subscriber_handler.cc
        ros_handler/src/ros_publisher_handler.cc
        ros_handler/src/ros_service_handler.cc
        ros_handler/src/ros_life_circle_handler.cc
        ros_handler/src/rosbag_parsing_handler.cc
        ros_handler/src/sig_crash.cc
        )

add_library(ros_interface ALIAS ros_interface_${CMAKE_SYSTEM_PROCESSOR})

ament_target_dependencies(ros_interface_${CMAKE_SYSTEM_PROCESSOR}
        rcl
        rclcpp
        rclcpp_components
        sensor_msgs
        geometry_msgs
        nav_msgs
        std_msgs
        visualization_msgs
        cv_bridge
        rosbag2_cpp
        tf2_ros
        mirobot_msgs
        )

target_link_libraries(ros_interface_${CMAKE_SYSTEM_PROCESSOR}
        vins_common
        vins_handler
        )

add_executable(${PROJECT_NAME}_node
                m_vins_node.cc)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
	${THIRD_LIB_LIBRARIES}
	ros_interface
        )

install(TARGETS ros_interface_${CMAKE_SYSTEM_PROCESSOR}
  DESTINATION lib/m_vins
)
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/m_vins
)
