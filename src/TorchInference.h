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

#ifndef TORCHINFERENCE_H_
#define TORCHINFERENCE_H_

#include <opencv2/core/mat.hpp>
#include <torch/script.h>

/**
 * This class represents a basic example of using Torch to perform inference using half precision.
 */
class TorchInference
{
public:
    /**
     * Basic constructor, initialises variables.
     *  @param width the width of images in pixels.
     *  @param height the height of images in pixels.
     */
    TorchInference(const int width, const int height);

    /**
     * Class destructor.
     */
    virtual ~TorchInference();

    /**
     * Initialises the inference by loading the model.
     *  @param pathToModel the file path to the model.
     */
    void initialise(const char* pathToModel);

    /**
     * Processes input image through the module to provide output tensor.
     *  @param inputImage an input image to process. It is expected to be a coloured image of configured size.
     *  @param[out] output the result of applying @p inputImage through the model. Tensor is detached and moved to CPU.
     *  @return the same @p output reference to allow chained operations on the output tensor.
     */
    torch::Tensor& processImage(const cv::Mat& inputImage, torch::Tensor& output);

private:
    /** The width of images in pixels. */
    int mWidth;
    /** The height of images in pixels. */
    int mHeight;
    /** JIT module that converts input image into trained output. */
    torch::jit::script::Module mModule;
    /** Input data as a tensor. */
    torch::Tensor mInputTensor;
    /** Floating-point representation of input image. */
    cv::Mat mFloatImage;
};

#endif /* TORCHINFERENCE_H_ */
