set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(xfeatures2d_${CMAKE_SYSTEM_PROCESSOR} SHARED
        src/freak.cpp
        src/msd.cpp
        src/stardetector.cpp
        )

add_library(xfeatures2d ALIAS xfeatures2d_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(xfeatures2d_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

if(USE_ROS2_API)
        install(TARGETS xfeatures2d_${CMAKE_SYSTEM_PROCESSOR}
          DESTINATION lib/m_vins
        )
endif()
