'''
////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2023 Mateusz Malinowski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
'''

import argparse
import pathlib
import torch
import torchvision


def convertModels(inputPath: str, height: int, width: int, channels: int, outputPath: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # this needs to match the trained model from jetracer_road_following repository
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(512, 2)
    model = model.cuda().eval().half()

    model.load_state_dict(torch.load(inputPath))

    example = torch.rand(1, channels, height, width).cuda().half()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(outputPath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts PyTorch model to torch script file for LibTorch')
    parser.add_argument('--input', type=pathlib.Path, help='Path to PyTorch model')
    parser.add_argument('--height', type=int, help='Height of an input images')
    parser.add_argument('--width', type=int, help='Width of an input images')
    parser.add_argument('--channels', type=int, help='The number of channels of an input images')
    parser.add_argument('--output', help='Path where converted model should be saved')
    args = parser.parse_args()

    convertModels(args.input, args.height, args.width, args.channels, args.output)
