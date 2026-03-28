cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0048 NEW)

project(third_lib)

# Include gflags.
set(GFLAGS_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/gflags")
set(GFLAGS_LIBRARIES ${PROJECT_SOURCE_DIR}/gflags/${CMAKE_SYSTEM_PROCESSOR}/libgflags.so)
include_directories(
    ${GFLAGS_INCLUDE_DIRS}
)

# Include glog.
set(GLOG_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/glog")
add_subdirectory(glog)

# Include Yaml-cpp.
set(YAML_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/yaml-cpp")
set(YAML_LIBRARIES ${PROJECT_SOURCE_DIR}/yaml-cpp/${CMAKE_SYSTEM_PROCESSOR}/libyaml-cpp.so)

set(THIRD_LIB_LIBRARIES
    glog_ydlidar
    ${GFLAGS_LIBRARIES}
    ${YAML_LIBRARIES}
    PARENT_SCOPE
)

set(THIRD_LIB_INCLUDE_DIR
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${YAML_INCLUDE_DIR}
    PARENT_SCOPE
)

install(TARGETS
        glog_ydlidar
    DESTINATION 
        lib/ydlidar_ros
)

install(FILES 
        ${GFLAGS_LIBRARIES}
        ${YAML_LIBRARIES}
    DESTINATION 
        lib/ydlidar_ros
)
