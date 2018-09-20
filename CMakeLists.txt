cmake_minimum_required(VERSION 3.5)
project(PicoPCL)
add_executable(PicoPCL PicoPCL.cpp)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O0 -std=c++11")

find_package(PkgConfig REQUIRED)

pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(PicoPCL ${OpenCV_LDFLAGS})

pkg_check_modules(Royale royale REQUIRED)
include_directories(${Royale_CFLAGS})
target_link_libraries(PicoPCL ${Royale_LDFLAGS})


find_package(PkgConfig REQUIRED)
pkg_check_modules(eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

pkg_check_modules(eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)
include_directories(${eigen3_INCLUDE_DIRS})
target_link_libraries(PicoPCL ${eigen3_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(PicoPCL ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(PicoPCL ${VTK_LIBRARIES})
