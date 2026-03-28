cmake_minimum_required(VERSION 2.8.12)
project(ydlidar_sdk C CXX)

#########################################################
# version
set(YDLIDAR_SDK_VERSION_MAJOR 1)
set(YDLIDAR_SDK_VERSION_MINOR 1)
set(YDLIDAR_SDK_VERSION_PATCH 6)
set(YDLIDAR_SDK_VERSION ${YDLIDAR_SDK_VERSION_MAJOR}.${YDLIDAR_SDK_VERSION_MINOR}.${YDLIDAR_SDK_VERSION_PATCH})

##########################################################
# Detect wordsize:
IF(CMAKE_SIZEOF_VOID_P EQUAL 8)  # Size in bytes!
	SET(CMAKE_MRPT_WORD_SIZE 64)
ELSE()
	SET(CMAKE_MRPT_WORD_SIZE 32)
ENDIF()
##################################################
#c++ 11
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

##########################################################
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "Current platform: Linux")
	#Linux add -fPIC
	add_compile_options(-fPIC)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE(STATUS "Current platform: Windows")
	add_compile_options("/utf-8")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
#or
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#####################################################
# add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(SDK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#############################################################################
# include cmake file
include(common/ydlidar_base)

#############################################################################
# Policy CMP0023 allows to mix old and new interfaces of target_link_libraries
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW) # We don't want to mix relative and absolute paths in linker lib lists.
	cmake_policy(SET CMP0005 NEW) # Escape definitions (-D) strings
	# if(POLICY CMP0053)
	# 	cmake_policy(SET CMP0053 OLD) # Simplify variable reference and escape sequence evaluation.
	# endif()
	# if(POLICY CMP0037)
	# 	cmake_policy(SET CMP0037 OLD)  # Allow defining target "test"
	# endif()
	# if(POLICY CMP0043)
	# 	cmake_policy(SET CMP0043 OLD) #  Ignore COMPILE_DEFINITIONS_<Config> properties.
	# endif()
endif()

#############################################################################
# option
option( YDLIDAR_SDK_BUILD_SHARED_LIBS "Build shared libraries." ON)

############################################################################
# include headers
include_directories(.)
include_directories(core)
include_directories(src)

#############################################################################
# addd subdirectory
add_subdirectory(core)
add_subdirectory(src)

#############################################################################
# PARSE libraries
include(common/ydlidar_parse)
include_directories(${SDK_INCS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

########################################################
## Create configure file for inclusion in library
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ydlidar_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ydlidar_config.h" )
set( GENERATED_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/ydlidar_config.h )


#############################################################################
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

#############################################################################################
#shared library
if(YDLIDAR_SDK_BUILD_SHARED_LIBS)
ydlidar_add_library(${PROJECT_NAME} SHARED ${SDK_SOURCES} ${SDK_HEADERS} ${GENERATED_HEADERS})
else()
ydlidar_add_library(${PROJECT_NAME} STATIC ${SDK_SOURCES} ${SDK_HEADERS} ${GENERATED_HEADERS})
endif()

target_link_libraries(${PROJECT_NAME} ${SDK_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
