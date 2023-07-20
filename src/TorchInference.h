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
    void initialise(const char* pathToModel, const int width, const int height, const int channels);

    /**
     * Processes input BGR image through the module to provide output tensor.
     *  @param inputImage an input image to process. It is expected to be a BGR image of configured size.
     *  @param[out] output the result of applying @p inputImage through the model. Tensor is detached and moved to CPU.
     *  @return the same @p output reference to allow chained operations on the output tensor.
     */
    torch::Tensor& processImage(const cv::Mat& inputImage, torch::Tensor& output);

    /**
     * Processes input image through the module to provide output tensor.
     *  @param inputImage an input image to process. It is expected to be a greyscale image of configured size.
     *  @param[out] output the result of applying @p inputImage through the model. Tensor is detached and moved to CPU.
     *  @return the same @p output reference to allow chained operations on the output tensor.
     */
    torch::Tensor& processGreyImage(const cv::Mat& inputImage, torch::Tensor& output);

private:
    /** Temporary buffer for converting images to floating point representation. */
    cv::Mat mFloatImage;
    /** Input tensor sizes. */
    std::array<int64_t, 4> mSizes;
    /** JIT module that converts input image into trained output. */
    torch::jit::script::Module mModule;
    /** Input data as a tensor. */
    torch::Tensor mInputTensor;
};

#endif /* TORCHINFERENCE_H_ */
