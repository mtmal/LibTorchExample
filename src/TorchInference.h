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

#pragma once

#include <opencv2/core/mat.hpp>
#include <torch/csrc/jit/api/module.h>

namespace at
{
    class Tensor;
}

/**
 * This class represents a basic example of using Torch to perform inference using half precision.
 */
class TorchInference
{
public:
    /**
     * Basic constructor.
     */
    TorchInference();

    /**
     * Class destructor.
     */
    virtual ~TorchInference();

    /**
     * Initialises the inference by loading the model. Also calls processImage with empty image to fully initialise
     * the module. Obviously loading the model is not enough to have the module fully initialised.
     *  @param pathToModel the file path to the model.
     *  @param width the width of images in pixels.
     *  @param height the height of images in pixels.
     *  @param channels the number of channels, 1 or 3.
     */
    void initialise(const std::string& pathToModel, const int width, const int height, const int channels);

    /**
     * Processes input BGR image through the module to provide output tensor.
     *  @param tta flag to indicate if test time augmentation should be applied. If true, @p inputImage is also flipped and the batch size is doubled.
     *  @param inputImage an input image to process. It is expected to be a BGR image of configured size.
     *  @param[out] output the result of applying @p inputImage through the model. Tensor is detached and moved to CPU.
     *  @return the same @p output reference to allow chained operations on the output tensor.
     */
    at::Tensor& processImage(const bool tta, const cv::Mat& inputImage, at::Tensor& output);

    /**
     * Processes stereo greyscale images through the module to provide output tensor.
     *  @param tta flag to indicate if test time augmentation should be applied. If true, @p leftImage and @p rightImage are also flipped and the batch size is doubled.
     *  @param leftImage an image from a left camera to process. It is expected to be a greyscale image of configured size.
     *  @param rightImage an image from a right camera to process. It is expected to be a greyscale image of configured size.
     *  @param[out] output the result of applying @p leftImage and @p rightImage through the model (batch size 2). Tensor is detached and moved to CPU.
     *  @return the same @p output reference to allow chained operations on the output tensor.
     */
    at::Tensor& processGreyImage(const bool tta, const cv::Mat& leftImage, const cv::Mat& rightImage, at::Tensor& output);

private:
    /**
     * Converts an image by scaling it to be from zero to one and applying standarisation based on ImageNet values.
     *  @param image an image to process.
     *  @param[out] tensor a resulting tensor.
     */
    void convertImage(const cv::Mat& image, at::Tensor& tensor);

    /** The number of channels in images. */
    int mChannels;
    /** Temporary buffer for converting images to floating point representation. */
    cv::Mat mFloatImage;
    /** Input tensor sizes. */
    std::array<int64_t, 4> mSizes;
    /** JIT module that converts input image into trained output. */
    torch::jit::script::Module mModule;
};
