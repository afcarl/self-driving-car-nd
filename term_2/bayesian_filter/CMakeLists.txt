project(BayesianFilter)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
   src/main.cpp
   src/bayesianFilter.h
   src/bayesianFilter.cpp
   src/help_functions.h
   src/map.h
   src/measurament_package.h)

add_executable(BayesianFilter ${sources})
