find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${DifferentialDrive_SOURCE_DIR}/include)
include_directories(${DifferentialDrive_SOURCE_DIR}/AprilTags/build/include/)
include_directories(/usr/local/include/eigen3)
include_directories(/usr/include/eigen3)

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories( ${Boost_INCLUDE_DIR} )

link_directories(${DifferentialDrive_SOURCE_DIR}/AprilTags/build/lib/)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DifferentialDrive_SOURCE_DIR}/sandbox/)

add_executable( sandbox sandbox.cpp )

target_link_libraries( sandbox ${Boost_LIBRARIES} ${OpenCV_LIBS} aprilvideointerface libapriltags.a libv4l2.so serial planner controller )#order matters, the one that comes earlier depends on the one that comes later
