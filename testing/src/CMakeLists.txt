add_library(exoLeftDynamics exoLeftDynamics.cpp)
add_library(exoRightDynamics exoRightDynamics.cpp)

# IF you don't include this, when building cmake, the header files won't be found
target_include_directories(exoLeftDynamics PUBLIC include)
target_include_directories(exoRightDynamics PUBLIC include)