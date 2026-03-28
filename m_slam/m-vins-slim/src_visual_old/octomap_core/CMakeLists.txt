cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)

project (octomap_core)
set(OCTOMAP_CORE_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        include
        ${THIRD_LIB_INCLUDE_DIR}
        ${VINS_COMMON_DIR}/include
        ${VINS_CORE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)

set(SOURCE_FILES
          src/octo_interface.cc
          src/octo_mapper.cc
          src/pointcloud_filter.cc)

add_library(octomap_core_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${SOURCE_FILES}
        )

add_library(octomap_core ALIAS octomap_core_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(octomap_core_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

target_link_libraries(octomap_core_${CMAKE_SYSTEM_PROCESSOR}
    aslam_cv
    ${OpenCV_LIBS}
    )

if(USE_ROS2_API)
    install(TARGETS octomap_core_${CMAKE_SYSTEM_PROCESSOR}
      DESTINATION lib/m_vins
    )
endif()
