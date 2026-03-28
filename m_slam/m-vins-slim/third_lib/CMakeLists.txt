cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project(third_lib)

# Include Yaml-cpp.
set(YAML_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/yaml-cpp")
set(YAML_LIBRARIES ${PROJECT_SOURCE_DIR}/yaml-cpp/${CMAKE_SYSTEM_PROCESSOR}/libyaml-cpp.so)
# Set parent scope variable.
set(YAML_INCLUDE_DIR ${YAML_INCLUDE_DIR} PARENT_SCOPE)
set(YAML_LIBRARIES ${YAML_LIBRARIES} PARENT_SCOPE)
message(STATUS YAML_INCLUDE_DIR ${YAML_INCLUDE_DIR})
message(STATUS YAML_LIBRARIES ${YAML_LIBRARIES})

# Include RKNN API.
set(RKNN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/rknn")
set(RKNN_LIBRARIES ${PROJECT_SOURCE_DIR}/rknn/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
# Set parent scope variable.
set(RKNN_INCLUDE_DIR ${RKNN_INCLUDE_DIR} PARENT_SCOPE)
set(RKNN_LIBRARIES ${RKNN_LIBRARIES} PARENT_SCOPE)
message(STATUS RKNN_INCLUDE_DIR ${RKNN_INCLUDE_DIR})
message(STATUS RKNN_LIBRARIES ${RKNN_LIBRARIES})

if (USE_CNN_FEATURE)
        set (THIRD_LIB_INCLUDE_DIR
                ${YAML_INCLUDE_DIR}
                ${RKNN_INCLUDE_DIR}
                PARENT_SCOPE
        )
        set (THIRD_LIB_LIBRARIES
                ${YAML_LIBRARIES}
                ${RKNN_LIBRARIES}
                PARENT_SCOPE
        )
else()
        set (THIRD_LIB_INCLUDE_DIR
                ${YAML_INCLUDE_DIR}
                PARENT_SCOPE
        )
        set (THIRD_LIB_LIBRARIES
                ${YAML_LIBRARIES}
                PARENT_SCOPE
        )
endif()

if (USE_ROS2_API)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/${CMAKE_SYSTEM_PROCESSOR}
                DESTINATION lib/m_vins
        )
        if (USE_CNN_FEATURE)
                install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rknn/${CMAKE_SYSTEM_PROCESSOR}
                        DESTINATION lib/m_vins
                )
        endif()
endif()