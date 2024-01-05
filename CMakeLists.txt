cmake_minimum_required(VERSION 3.12)

# set the project name
project(TorchInference)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-fPIC -g -pedantic -Wall -Wextra")

# specify older CUDA standard for CUDA 10.2. Needs to be done for Jetson Nano with Ubuntu 20.04
set(CMAKE_CUDA_STANDARD 14)

# get OpenCV
find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# get Torch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIR})

# source code
include_directories(src)

# build the actual library
add_library(TorchInference SHARED src/TorchInference.cpp)
target_link_libraries(TorchInference PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

# get Torch_TensorRT
find_package(TorchTensorRT COMPONENTS)
if(${TorchTensorRT_FOUND})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed")
    target_link_libraries(TorchInference PRIVATE ${TORCHTENSORRT_LIBRARIES})
else()
    message(STATUS "TorchTensorRT not found. Please write a custom FindTorchTensorRT.cmake and place it in your torch_tensorrt folder within Python dist-packages for this project to support torch_tensorrt-optimised models.")
endif()

# add the test application
add_executable(test_Inference tests/test_inference.cpp)
target_link_libraries(test_Inference TorchInference)

file(COPY tests/0.453627_-1.000000_c8155f0a-fd0e-11ec-b27a-3413e86352f4.jpg DESTINATION ./)
file(COPY tests/1.000000_-1.000000_988_left.jpg DESTINATION ./)
file(COPY tests/1.000000_-1.000000_988_right.jpg DESTINATION ./)
