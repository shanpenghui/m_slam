cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project (vins_handler)
set(VINS_HANDLER_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        include
        ${THIRD_LIB_INCLUDE_DIR}
        ${VINS_COMMON_DIR}/include
        ${VINS_CORE_DIR}/include
        ${OCTOMAP_CORE_DIR}/include
        ${LOOP_CLOSURE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)

set(SOURCE_FILES
          src/map_tool.cc
          src/vins_thread.cc
          src/vins_handler.cc
          src/vins_monitor.cc)

add_library(vins_handler_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${SOURCE_FILES}
        )

add_library(vins_handler ALIAS vins_handler_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(vins_handler_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

target_link_libraries(vins_handler_${CMAKE_SYSTEM_PROCESSOR}
    aslam_cv
    loop_closure
    vins_core
    octomap_core
    ${OpenCV_LIBS}
    )

if(USE_ROS2_API)
    install(TARGETS vins_handler_${CMAKE_SYSTEM_PROCESSOR}
      DESTINATION lib/m_vins
    )
endif()
