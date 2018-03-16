cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")

project( DifferentialDrive )

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DifferentialDrive_SOURCE_DIR}/build/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DifferentialDrive_SOURCE_DIR}/build/bin)

add_subdirectory(./sandbox)

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories( ${Boost_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${DifferentialDrive_SOURCE_DIR}/include)
include_directories(${DifferentialDrive_SOURCE_DIR}/AprilTags/build/include/)
include_directories(/usr/local/include/eigen3)
include_directories(/usr/include/eigen3)


link_directories(${DifferentialDrive_SOURCE_DIR}/AprilTags/build/lib/)
#link_directories(${CMAKE_SOURCE_DIR}/build/)#for our own library objects, not required as cmake already knows

add_library(serial SHARED ${DifferentialDrive_SOURCE_DIR}/src/Serial.cpp)
add_library(planner SHARED ${DifferentialDrive_SOURCE_DIR}/src/pathplanners.cpp)
add_library(controller SHARED ${DifferentialDrive_SOURCE_DIR}/src/controllers.cpp)
add_library(aprilvideointerface SHARED ${DifferentialDrive_SOURCE_DIR}/src/aprilvideointerface.cpp)

add_executable( differentialDrive differentialDrive.cpp )
target_link_libraries( differentialDrive ${Boost_LIBRARIES} boost_system ${OpenCV_LIBS} aprilvideointerface libapriltags.a libv4l2.so serial planner controller )#order matters, the one that comes earlier depends on the one that comes later
