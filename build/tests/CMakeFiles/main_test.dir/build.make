# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.7/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.7/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/andrewgao/Documents/Bionics/symforce

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/andrewgao/Documents/Bionics/symforce/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/main_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/main_test.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/main_test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/main_test.dir/flags.make

tests/CMakeFiles/main_test.dir/secondTest.cpp.o: tests/CMakeFiles/main_test.dir/flags.make
tests/CMakeFiles/main_test.dir/secondTest.cpp.o: /Users/andrewgao/Documents/Bionics/symforce/tests/secondTest.cpp
tests/CMakeFiles/main_test.dir/secondTest.cpp.o: tests/CMakeFiles/main_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/andrewgao/Documents/Bionics/symforce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/main_test.dir/secondTest.cpp.o"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/main_test.dir/secondTest.cpp.o -MF CMakeFiles/main_test.dir/secondTest.cpp.o.d -o CMakeFiles/main_test.dir/secondTest.cpp.o -c /Users/andrewgao/Documents/Bionics/symforce/tests/secondTest.cpp

tests/CMakeFiles/main_test.dir/secondTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main_test.dir/secondTest.cpp.i"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/andrewgao/Documents/Bionics/symforce/tests/secondTest.cpp > CMakeFiles/main_test.dir/secondTest.cpp.i

tests/CMakeFiles/main_test.dir/secondTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main_test.dir/secondTest.cpp.s"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/tests && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/andrewgao/Documents/Bionics/symforce/tests/secondTest.cpp -o CMakeFiles/main_test.dir/secondTest.cpp.s

# Object files for target main_test
main_test_OBJECTS = \
"CMakeFiles/main_test.dir/secondTest.cpp.o"

# External object files for target main_test
main_test_EXTERNAL_OBJECTS =

tests/main_test: tests/CMakeFiles/main_test.dir/secondTest.cpp.o
tests/main_test: tests/CMakeFiles/main_test.dir/build.make
tests/main_test: src/libexoLeftDynamics.a
tests/main_test: src/libexoRightDynamics.a
tests/main_test: tests/CMakeFiles/main_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/andrewgao/Documents/Bionics/symforce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main_test"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/main_test.dir/build: tests/main_test
.PHONY : tests/CMakeFiles/main_test.dir/build

tests/CMakeFiles/main_test.dir/clean:
	cd /Users/andrewgao/Documents/Bionics/symforce/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/main_test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/main_test.dir/clean

tests/CMakeFiles/main_test.dir/depend:
	cd /Users/andrewgao/Documents/Bionics/symforce/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/andrewgao/Documents/Bionics/symforce /Users/andrewgao/Documents/Bionics/symforce/tests /Users/andrewgao/Documents/Bionics/symforce/build /Users/andrewgao/Documents/Bionics/symforce/build/tests /Users/andrewgao/Documents/Bionics/symforce/build/tests/CMakeFiles/main_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/main_test.dir/depend
