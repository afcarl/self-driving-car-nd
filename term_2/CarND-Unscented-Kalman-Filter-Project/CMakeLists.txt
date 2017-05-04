project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
   src/lidar.h
   src/lidar.cpp
   src/radar.h
   src/radar.cpp
   src/ctrv.h
   src/ctrv.cpp
   src/ukf.h
   src/ukf.cpp
   src/fusion.h
   src/fusion.cpp
   src/main.cpp
   src/tools.cpp)

add_executable(UnscentedKF ${sources})
