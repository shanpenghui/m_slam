cmake_minimum_required(VERSION 3.5)
project(interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(dependencies
  builtin_interfaces
  std_msgs
  nav_msgs
  sensor_msgs
  shape_msgs
  geometry_msgs
)

# Collect all message, service, and action files using globbing
file(GLOB MSG_INTERFACE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "msg/*.msg")
file(GLOB SRV_INTERFACE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "srv/*.srv")
file(GLOB ACTION_INTERFACE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "action/*.action")
rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_INTERFACE_FILES}
  ${SRV_INTERFACE_FILES}
  ${ACTION_INTERFACE_FILES}
  DEPENDENCIES ${dependencies}
)

ament_auto_package ()
