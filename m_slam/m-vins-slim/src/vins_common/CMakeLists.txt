cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)

project (vins_common)
set(VINS_COMMON_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${THIRD_LIB_INCLUDE_DIR}
        ${CMAKE_BINARY_DIR}/generated
)

set(SOURCE_FILES
        src/cfg_common/slam_config.cc
        src/data_common/data_structures_impl.cc
        src/file_common/file_system_tools.cc
        src/file_common/binary_serialization.cc
        src/log_common/logging_tools.cc
        src/time_common/time.cc
        src/time_common/time_table.cc
        src/occ_common/castrayer.cc
	src/occ_common/fast_correlative_scan_matcher.cc
	src/occ_common/realtime_correlative_scan_matcher.cc
        src/occ_common/grid_2d.cc
        src/occ_common/laser_line_detector.cc
        src/occ_common/live_submaps.cc
        src/occ_common/pointcloud_kdtree.cc
        src/occ_common/probability_grid.cc
        src/occ_common/probability_values.cc
        src/occ_common/submap_2d.cc
        src/occ_common/voxel_filter.cc
        )

add_library(vins_common_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${SOURCE_FILES}
        )

add_library(vins_common ALIAS vins_common_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(vins_common_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

target_link_libraries(vins_common_${CMAKE_SYSTEM_PROCESSOR}
    ${OpenCV_LIBS}
    )

install(TARGETS vins_common_${CMAKE_SYSTEM_PROCESSOR}
  DESTINATION lib/m_vins
)
