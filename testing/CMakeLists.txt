#----------------------------------------------------------------------------------------------------
# File: CMakeLists.txt
# Desc: Top level CMake configuration file for the symforce_tool project
# Usage: 
#----------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.13)
project(symforce_tool CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#----------------------------------------------------------------------------------------------------
# Third party libraries
#----------------------------------------------------------------------------------------------------
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

#----------------------------------------------------------------------------------------------------
# Testing (comparison with functions generated from a sympy tool)
#----------------------------------------------------------------------------------------------------
# include_directories("./include/")
# add_library(exoLeftDynamics "./src/exoLeftDynamics.cpp")
# add_library(exoRightDynamics "./src/exoRightDynamics.cpp")
# add_executable(firstTest firstTest.cpp )
# target_link_libraries(firstTest exoLeftDynamics exoRightDynamics Eigen3::Eigen)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)