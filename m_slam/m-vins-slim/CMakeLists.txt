cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0048 NEW)

# Project name and version.
project(m_vins)
set(CMAKE_PROJECT_VERSION 0.1.0)
message(STATUS "Project ${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_VERSION}")

# Build CNN local feature.
# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Set build type.
set(CMAKE_BUILD_TYPE "Debug")
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -O2")
    add_definitions(-DPUB_DEBUG_INFO)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
message(STATUS "Build ${CMAKE_BUILD_TYPE} version.")

# Print build duration.
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
add_definitions(-fPIC -Wall -Wno-unknown-pragmas)

# Target platform, default is the host platform.
message(STATUS "PROCESSOR_TYPE: ${CMAKE_SYSTEM_PROCESSOR}")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ARM_NEON__ -flto -march=armv8-a")
    message(STATUS "ARM processor detected, will attempt to use NEON.")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
    message(STATUS "Assuming SSE instructions available.")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
endif()

# Eigen is header only.
find_package(Eigen3 REQUIRED)
message(STATUS EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

# Find boost.
find_package(Boost REQUIRED COMPONENTS date_time thread system)
message(STATUS BOOST_INCLUDE_DIRS ${BOOST_INCLUDE_DIRS})
message(STATUS BOOST_LIBRARIES ${BOOST_LIBRARIES})

# Find ceres solver
find_package(Ceres REQUIRED)
message(STATUS CERES_INCLUDE_DIRS ${CERES_INCLUDE_DIRS})
message(STATUS CERES_LIBRARIES ${CERES_LIBRARIES})

# Find OpenCV.
find_package(OpenCV 4.2.0 REQUIRED)
message(STATUS OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
message(STATUS OpenCV_VERSION ${OpenCV_VERSION})

find_package(octomap REQUIRED)
message(STATUS OCTOMAP_INCLUDE_DIRS ${OCTOMAP_INCLUDE_DIRS})
message(STATUS OCTOMAP_LIBRARIES ${OCTOMAP_LIBRARIES})


# Find ROS2 dependencies.
find_package(ament_cmake REQUIRED)
find_package(rcl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(mirobot_msgs REQUIRED)

# Third party libraries to link.
add_subdirectory(third_lib)

include_directories(
        ${BOOST_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OCTOMAP_INCLUDE_DIRS}
        ${THIRD_LIB_INCLUDE_DIR}
        )

add_subdirectory(src/thirdparty)

# Third party libraries to build.
include_directories(
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_calibration/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_cameras/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_common/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_detector/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_frames/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_geometric_vision/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_matcher/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_pipeline/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_tracker/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_triangulation/include
        ${THIRD_PARTY_DIR}/aslam_cv2/aslam_cv_visualization/include
        ${THIRD_PARTY_DIR}/minkindr/minkindr/include
        ${THIRD_PARTY_DIR}/libnabo
        ${THIRD_PARTY_DIR}/xfeatures2d/include
)

link_libraries(
        ${BOOST_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${OCTOMAP_LIBRARIES}
        ${THIRD_LIB_LIBRARIES}
)


add_subdirectory(src/vins_common)

add_subdirectory(src/loop_closure)

add_subdirectory(src/vins_core)

add_subdirectory(src/octomap_core)

add_subdirectory(src/vins_handler)

add_subdirectory(src/ros_interface)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slam_online_log_clear.sh
  DESTINATION scripts/m_vins/
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slam_online_machine.sh
  DESTINATION scripts/${PROJECT_NAME}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/slam_offline.sh
  DESTINATION scripts/${PROJECT_NAME}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kill_all.sh
  DESTINATION scripts/${PROJECT_NAME}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/rviz/m-vins.rviz
  DESTINATION share/${PROJECT_NAME}/rviz
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/assets/inverted_multi_index_quantizer_freak.dat
  DESTINATION assets/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(DIRECTORY cfg DESTINATION share/${PROJECT_NAME})

ament_package()







