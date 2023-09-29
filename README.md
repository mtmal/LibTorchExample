# TorchInference
An example of inference using LibTorch.

## Requirements
* OpenCV
* cmake 3.12 (older versions do not resolve torch dependencies correctly)
* PyTorch 1.10 (LibTorch comes installed with it) that comes with JetPack 4.6.4
* [Torch-TensorRT](https://github.com/pytorch/TensorRT) 

When building this library, CMakeList.txt will look for TorchTensorRTConfig.cmake that is not provided by default. Example can be found [here](https://github.com/mtmal/jetcard/blob/jetpack_4.6.4/TorchTensorRTConfig.cmake)

## Building
```
$ mkdir build
$ cd build
$ cmake .. -DTorch_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch
$ make
$ export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/torch/lib:$LD_LIBRARY_PATH
```
Note that you either need to pass a path to Torch folder as in the above example, or you need to export your own environment variable, or add it to your path.

## Converting PyTorch Models
PyTorch model needs to be converted to JIT script. A Python scirpt is provided for convenience that does the job (converts to half precision). Example usage:
```
$ python3 utils/convertModel.py --input your_model.pth --height 224 --width 224 --channels 1 --batch 2 --type trt --output your_model.ts
```

