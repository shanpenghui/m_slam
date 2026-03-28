cmake_minimum_required (VERSION 2.8.12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

ADD_DEFINITIONS(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
SET(CMAKE_BUILD_TYPE Release)
set (CMAKE_DEBUG_POSTFIX d)
set (CMAKE_THREAD_PREFER_PTHREAD 1)

find_package(Threads)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set (GLOG_PUBLIC_H
  ${CMAKE_CURRENT_SOURCE_DIR}/glog/logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/glog/raw_logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/glog/stl_logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/glog/vlog_is_on.h
  ${CMAKE_CURRENT_SOURCE_DIR}/glog/log_severity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h
)

set (GLOG_SRCS
  ${GLOG_PUBLIC_H}
  ${CMAKE_CURRENT_SOURCE_DIR}/base/commandlineflags.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/googleinit.h
  ${CMAKE_CURRENT_SOURCE_DIR}/base/mutex.h
  ${CMAKE_CURRENT_SOURCE_DIR}/demangle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/demangle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/raw_logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/symbolize.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/symbolize.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vlog_is_on.cc
)

add_library (glog_ydlidar SHARED
  ${GLOG_SRCS}
)

set_target_properties (glog_ydlidar PROPERTIES POSITION_INDEPENDENT_CODE ON)
