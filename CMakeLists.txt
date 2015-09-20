cmake_minimum_required(VERSION 2.8)
project(Seg_by_Comp)

# set up flags here
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# find needed packages
find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
${SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
)

set(LINK_LIBS
${OpenCV_LIBS}
)

set(SOURCEFILES
${SOURCE_DIR}/main.cpp
${SOURCE_DIR}/array_to_matrix.cpp
${SOURCE_DIR}/distancemap.cpp
${SOURCE_DIR}/multi_object_tracking.cpp
${SOURCE_DIR}/narrowband.cpp
${SOURCE_DIR}/region_solver.cpp)

add_executable(Seg_by_Comp ${SOURCEFILES} )
target_link_libraries(Seg_by_Comp ${LINK_LIBS} )

