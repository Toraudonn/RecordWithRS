#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(rs-capture)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# OPENCV
find_package( OpenCV REQUIRED )

# REALSENSE
find_library(RS_LIB realsense)

set(SOURCES rs-capture.cpp)

add_executable(rs-capture ${SOURCES})
target_link_libraries(rs-capture ${RS_LIBS} ${OpenCV_LIBS} ${DEPENDENCIES})