# Switch on warnings.
add_definitions(-Wall -Wextra)

# main nabo lib
set(NABO_SRC
    nabo/nabo.cpp
    nabo/brute_force_cpu.cpp
    nabo/kdtree_cpu.cpp
    )

add_library(nabo_${CMAKE_SYSTEM_PROCESSOR} SHARED
        ${NABO_SRC}
        )

add_library(nabo ALIAS nabo_${CMAKE_SYSTEM_PROCESSOR})

# Set VERSION
set_target_properties(nabo_${CMAKE_SYSTEM_PROCESSOR} PROPERTIES
        VERSION ${CMAKE_PROJECT_VERSION})

install(TARGETS nabo_${CMAKE_SYSTEM_PROCESSOR}
  DESTINATION lib/m_vins
)
