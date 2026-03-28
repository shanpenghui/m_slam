cmake_minimum_required(VERSION 3.8)
project(m_vins)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(m_vins_node src/m_vins_node.cc)
ament_target_dependencies(m_vins_node rclcpp)

install(TARGETS m_vins_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
