cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
project(Particle_Filter)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( include ${Eigen3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})
add_executable(pf src/particle_filter.cpp)
target_link_libraries(pf ${OpenCV_LIBRARIES})

