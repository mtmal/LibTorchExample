cmake_minimum_required(VERSION 3.12)

# set the project name
project(TorchInference)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-fPIC -g -pedantic -Wall -Wextra")

# get OpenCV
find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# get Torch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIR})

# source code
include_directories(src)

add_library(TorchInference SHARED src/TorchInference.h src/TorchInference.cpp)
target_link_libraries(TorchInference ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

