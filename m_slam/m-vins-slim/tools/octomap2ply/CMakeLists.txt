cmake_minimum_required(VERSION 3.5)
project(convert_octomap_to_ply)

set(CMAKE_CXX_STANDARD 11)

find_package(octomap REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(convert_octomap_to_ply convert_octomap_to_ply.cc)
target_include_directories(convert_octomap_to_ply PUBLIC ${OCTOMAP_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(convert_octomap_to_ply ${OCTOMAP_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})