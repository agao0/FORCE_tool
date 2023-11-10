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
include src/CMakeFiles/exoRightDynamics.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/exoRightDynamics.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/exoRightDynamics.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/exoRightDynamics.dir/flags.make

src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o: src/CMakeFiles/exoRightDynamics.dir/flags.make
src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o: /Users/andrewgao/Documents/Bionics/symforce/src/exoRightDynamics.cpp
src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o: src/CMakeFiles/exoRightDynamics.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/andrewgao/Documents/Bionics/symforce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o -MF CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o.d -o CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o -c /Users/andrewgao/Documents/Bionics/symforce/src/exoRightDynamics.cpp

src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.i"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/andrewgao/Documents/Bionics/symforce/src/exoRightDynamics.cpp > CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.i

src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.s"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/andrewgao/Documents/Bionics/symforce/src/exoRightDynamics.cpp -o CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.s

# Object files for target exoRightDynamics
exoRightDynamics_OBJECTS = \
"CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o"

# External object files for target exoRightDynamics
exoRightDynamics_EXTERNAL_OBJECTS =

src/libexoRightDynamics.a: src/CMakeFiles/exoRightDynamics.dir/exoRightDynamics.cpp.o
src/libexoRightDynamics.a: src/CMakeFiles/exoRightDynamics.dir/build.make
src/libexoRightDynamics.a: src/CMakeFiles/exoRightDynamics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/andrewgao/Documents/Bionics/symforce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libexoRightDynamics.a"
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && $(CMAKE_COMMAND) -P CMakeFiles/exoRightDynamics.dir/cmake_clean_target.cmake
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exoRightDynamics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/exoRightDynamics.dir/build: src/libexoRightDynamics.a
.PHONY : src/CMakeFiles/exoRightDynamics.dir/build

src/CMakeFiles/exoRightDynamics.dir/clean:
	cd /Users/andrewgao/Documents/Bionics/symforce/build/src && $(CMAKE_COMMAND) -P CMakeFiles/exoRightDynamics.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/exoRightDynamics.dir/clean

src/CMakeFiles/exoRightDynamics.dir/depend:
	cd /Users/andrewgao/Documents/Bionics/symforce/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/andrewgao/Documents/Bionics/symforce /Users/andrewgao/Documents/Bionics/symforce/src /Users/andrewgao/Documents/Bionics/symforce/build /Users/andrewgao/Documents/Bionics/symforce/build/src /Users/andrewgao/Documents/Bionics/symforce/build/src/CMakeFiles/exoRightDynamics.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/exoRightDynamics.dir/depend

