cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0057 NEW)
project(ydlidar_ros)
set(CMAKE_PROJECT_VERSION 0.15)
message(STATUS "Project ${PROJECT_NAME} ${CMAKE_PROJECT_VERSION}")

# Set build type.
option(CMAKE_BUILD_TYPE "build type" "Release")
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -g -O2")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
message(STATUS "Build ${CMAKE_BUILD_TYPE} version.")

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")
    add_definitions(-DARM_SYS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ARM_NEON__ -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
    message(STATUS "ARM processor detected, will attempt to use NEON.")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
    message(STATUS "Assuming SSE instructions available.")
endif()

#############
## C++14 ##
# Compile as C++14, supported in ROS Kinetic and newer.
#############
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
endif()

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mirobot_msgs REQUIRED)

add_subdirectory(third_lib)
add_subdirectory(sdk)

file(GLOB SRC_LIST 
    "src/*.cpp"
    "src/*.cc"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${THIRD_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sdk/.
)

add_executable(ydlidar_node
    ${SRC_LIST}
)

ament_target_dependencies(ydlidar_node
    rclcpp
    sensor_msgs
    mirobot_msgs
)

target_link_libraries(ydlidar_node
    ydlidar_sdk
    ${THIRD_LIB_LIBRARIES}
)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/kill_all.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_online_machine.sh
    DESTINATION 
        scripts/${PROJECT_NAME}
)

install(FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/config.yaml
    DESTINATION 
        cfg/${PROJECT_NAME}
)

install(TARGETS
        ydlidar_node
    DESTINATION 
        bin/${PROJECT_NAME}
)

install(TARGETS
        ydlidar_sdk
    DESTINATION 
        lib/${PROJECT_NAME}
)
