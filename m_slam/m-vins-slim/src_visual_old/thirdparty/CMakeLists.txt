cmake_minimum_required(VERSION 3.5)
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
            ${PROJECT_SOURCE_DIR}/libnabo
    ${PROJECT_SOURCE_DIR}/minkindr/minkindr/include
    )

add_subdirectory(libnabo)
