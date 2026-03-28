cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)

project (loop_closure)
set(LOOP_CLOSURE_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        include
        ${THIRD_LIB_INCLUDE_DIR}
        ${VINS_COMMON_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)

set(SOURCE_FILES
        src/descriptor_projection/descriptor_projection.cc
        src/loop_detector/loop_detector.cc
        src/loop_detector/pnp.cc
        src/loop_interface/loop_setting.cc
        src/loop_interface/scan_loop_interface.cc
        src/loop_interface/visual_loop_interface.cc
        src/loop_index/index_interface.cc
        src/loop_index/inverted_multi_index_interface.cc
        src/loop_index/inverted_multi_index.cc
        src/loop_index/kd_tree_index_interface.cc
        src/loop_index/kd_tree_index.cc
        src/summary_map/map_loader.cc
        src/summary_map/map_saver.cc
        src/summary_map/summary_map.cc
        )

add_library(loop_closure_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${SOURCE_FILES}
        )

add_library(loop_closure ALIAS loop_closure_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(loop_closure_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

target_link_libraries(loop_closure_${CMAKE_SYSTEM_PROCESSOR}
        aslam_cv
        nabo
        sqlite3
        vins_common
        )

if(USE_ROS2_API)
        install(TARGETS loop_closure_${CMAKE_SYSTEM_PROCESSOR}
          DESTINATION lib/m_vins
        )
endif()