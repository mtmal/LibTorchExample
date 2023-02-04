# TorchInference
An example of inference using LibTorch.


## Requirements
* OpenCV
* PyTorch (LibTorch comes installed with it)
* cmake 3.12 (older versions do not resolve torch dependencies correctly)

## Building
```
$ mkdir build
$ cd build
$ cmake .. -DTorch_DIR=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch
$ make
```
Note that you either need to pass a path to Torch folder as in the above example, or you need to export your own environment variable, or add it to your path.

## Converting PyTorch Models
PyTorch model needs to be converted to JIT script. A Python scirpt is provided for convenience that does the job (converts to half precision). Example usage:
```
$ python3 utils/convertModel.py --input drive_road_following_model.pth --height 224 --width 224 --channels 3 --output drive_road_following_model_cpp.pt
```

