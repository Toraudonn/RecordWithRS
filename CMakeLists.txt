#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

# Define the project
project(RecordWithRS)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# View the makefile commands during build
#set(CMAKE_VERBOSE_MAKEFILE on)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

include(GNUInstallDirs)

# Initial check for packages:

find_package(OpenGL)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
endif()

set(DEPENDENCIES realsense2 ${OPENGL_LIBRARIES})

# Find glfw header
find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
    PATHS /usr/X11R6/include
            /usr/include/X11
            /opt/graphics/OpenGL/include
            /opt/graphics/OpenGL/contrib/libglfw
            /usr/local/include
            /usr/include/GL
            /usr/include
)
# Find glfw library
find_library(GLFW_LIBRARIES NAMES glfw glfw3
        PATHS /usr/lib64
                /usr/lib
                /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                /usr/local/lib64
                /usr/local/lib
                /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                /usr/X11R6/lib
)
list(APPEND DEPENDENCIES m ${GLFW_LIBRARIES} ${LIBUSB1_LIBRARIES})
include_directories(${GLFW_INCLUDE_DIR})

# Sub directories:

add_subdirectory(src)
# add_subdirectory(third-party)

