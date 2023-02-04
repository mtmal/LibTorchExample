# LibTorchInference
An example of inference using LibTorch.


## Requirements
* OpenCV
* PyTorch (LibTorch comes installed with it)
* cmake 3.12 (lower versions do not resolve torch dependencies correctly)

## Building
```
$ mkdir build
$ cd build
$ cmake .. -DTorch_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch
$ make
```
Note that you either need to pass a path to Torch folder as in the above example, or you need to export your own environment variable, or add it to your path.
