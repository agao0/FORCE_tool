# add_executable(main_test firstTest.cpp)
add_executable(main_test secondTest.cpp)
target_link_libraries(main_test exoLeftDynamics exoRightDynamics Eigen3::Eigen)

# We want to tell cmake to add the test executable (above) to its list of tests it can run with ctest. To do this we do:
# first arg is the name we want for the test, the second arg is the executable name
add_test(please_test main_test)
# need to enable_testing() to root CMakeLists.txt
# we can also use ctest to run test scripts "script_test.sh", just have to add and run

# How can the test file find out where the include is? I didn't even link it...
# Also, in firstTest.cpp, I only need to use #include "fileName", not even a path at all... interesting
# How to rebuild something (namely the firstTest changes aren't appearing): NEED TO MAKE

