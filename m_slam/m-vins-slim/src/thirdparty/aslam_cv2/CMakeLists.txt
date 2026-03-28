add_library(aslam_cv_${CMAKE_SYSTEM_PROCESSOR} SHARED
        #
        aslam_cv_cameras/src/camera.cc
        aslam_cv_cameras/src/camera-equirect.cc
        aslam_cv_cameras/src/camera-factory.cc
        aslam_cv_cameras/src/camera-pinhole.cc
        aslam_cv_cameras/src/camera-unified-projection.cc
        aslam_cv_cameras/src/camera-yaml-serialization.cc
        aslam_cv_cameras/src/distortion.cc
        aslam_cv_cameras/src/distortion-equidistant.cc
        aslam_cv_cameras/src/distortion-fisheye.cc
        aslam_cv_cameras/src/distortion-plumb-bob.cc
        aslam_cv_cameras/src/distortion-radtan.cc
        aslam_cv_cameras/src/ncamera.cc
        aslam_cv_cameras/src/ncamera-yaml-serialization.cc
        #
        aslam_cv_common/src/channel.cc
        aslam_cv_common/src/channel-serialization.cc
        aslam_cv_common/src/hash-id.cc
        aslam_cv_common/src/reader-writer-lock.cc
        aslam_cv_common/src/thread-pool.cc
        aslam_cv_common/src/timer.cc
        )

add_library(aslam_cv ALIAS aslam_cv_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(aslam_cv_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

if(USE_ROS2_API)
        install(TARGETS aslam_cv_${CMAKE_SYSTEM_PROCESSOR}
          DESTINATION lib/m_vins
        )
endif()