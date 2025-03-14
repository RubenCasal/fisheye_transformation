cmake_minimum_required(VERSION 3.10)
project(FisheyeExample)

find_package(OpenCV REQUIRED)

add_executable(fisheye main.cpp)
target_link_libraries(fisheye ${OpenCV_LIBS})
