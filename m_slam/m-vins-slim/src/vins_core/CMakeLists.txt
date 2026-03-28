cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)

project (vins_core)
set(VINS_CORE_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

include_directories(
        include
        ${THIRD_LIB_INCLUDE_DIR}
        ${VINS_COMMON_DIR}/include
        ${LOOP_CLOSURE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)

set(SOURCE_FILES
        src/cost_function/camera_reprojection_cost.cc
        src/cost_function/imu_propagation_cost.cc
        src/cost_function/line_rectional_cost.cc
        src/cost_function/marginalization_cost_function.cc
        src/cost_function/occupied_space_cost_function.cc
        src/cost_function/odom_propagation_cost.cc
        src/cost_function/pose_in_plane_cost.cc
        src/cost_function/pose_local_parameterization.cc
        src/cost_function/relative_pose_cost.cc
        src/cost_function/schur_complement_problem.cc
        src/cost_function/switch_prior_cost.cc
        src/cost_function/zero_velocity_cost.cc
                                                                        src/hybrid_optimizer/hybrid_optimizer.cc
        src/sensor_propagator/imu_propagator.cc
        src/sensor_propagator/motion_checker.cc
        src/sensor_propagator/odom_propagator.cc
        src/sensor_propagator/sensor_propagator.cc
        )

if(USE_CNN_FEATURE)
        set(SOURCE_FILES
            ${SOURCE_FILES}
            
endif()

add_library(vins_core_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${SOURCE_FILES}
        )

add_library(vins_core ALIAS vins_core_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(vins_core_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

target_link_libraries(vins_core_${CMAKE_SYSTEM_PROCESSOR}
    nabo
    ${THIRD_LIB_LIBRARIES}
    ${OpenCV_LIBS}
    )

if(USE_ROS2_API)
    install(TARGETS vins_core_${CMAKE_SYSTEM_PROCESSOR}
      DESTINATION lib/m_vins
    )
endif()