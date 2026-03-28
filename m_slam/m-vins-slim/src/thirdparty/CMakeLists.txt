cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project(thirdparty)
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

link_libraries(
    ${OpenCV_LIBRARIES}
    ${THIRD_LIB_LIBRARIES}
    )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${THIRD_LIB_INCLUDE_DIR}
    )

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_calibration/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_cameras/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_common/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_detector/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_frames/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_geometric_vision/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_matcher/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_pipeline/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_tracker/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_triangulation/include
    ${PROJECT_SOURCE_DIR}/aslam_cv2/aslam_cv_visualization/include
    ${PROJECT_SOURCE_DIR}/libnabo
    ${PROJECT_SOURCE_DIR}/minkindr/minkindr/include
    ${PROJECT_SOURCE_DIR}/xfeatures2d/include
    )

add_subdirectory(aslam_cv2)
add_subdirectory(libnabo)
add_subdirectory(xfeatures2d)
