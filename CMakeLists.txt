#  minimum required cmake version: 3.1.0
cmake_minimum_required( VERSION 3.1.0 )

# Define the project
project( rs-capture )

# Save the command line compile commands in the build output
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )
if(COMPILER_SUPPORTS_CXX11)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif(COMPILER_SUPPORTS_CXX0X)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
endif()

#====================================================================================================================#
# Options:

# include( GNUInstallDirs )

include( CMakeToolsHelpers OPTIONAL )

#====================================================================================================================#
# Dependencies:

# Boost
set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_RUNTIME OFF )
find_package( Boost COMPONENTS system filesystem date_time )
if( NOT Boost_FOUND )
    message( SEND_ERROR "Failed to find Boost" )
    return()
else()
    include_directories( ${Boost_INCLUDE_DIRS} )
endif()

# OPENCV
find_package( OpenCV REQUIRED )

# REALSENSE
find_library( RS_LIB realsense2 )

set( DEPENDENCIES ${RS_LIB} ${OpenCV_LIBS} ${Boost_LIBRARIES} )

#====================================================================================================================#

# Add SOURCES:
file( GLOB SOURCES "src/rs-capture.cpp" )

# Add HEADERS:
file( GLOB HEADERS "include/*.hpp" )

#====================================================================================================================#

#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

# don't really need this now
#link_directories( ${CMAKE_BINARY_DIR} )
set( EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR} )

# executables
add_executable( rs-capture ${SOURCES} ${HEADERS} )
target_link_libraries( rs-capture ${DEPENDENCIES} )
