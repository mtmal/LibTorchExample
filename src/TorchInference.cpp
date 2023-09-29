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

#include <torch/script.h>
#include "TorchInference.h"


namespace
{
/** Desired order of image dimentions. */
const constexpr std::array<int64_t, 4> PERMUTE_DIM = {0, 3, 1, 2};

/**
 * Permutes and adust pixel valus of an inout tensor.
 *  @param[in/out] inputTensor 3-channel input tensor to be adjusted.
 */
void adjustInputRGBTensor(at::Tensor& inputTensor)
{
    inputTensor = inputTensor.permute(PERMUTE_DIM);
    /* do the shenanigans: Imagenet's mean and std pixel values */
    inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
    inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
    inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);
}

/**
 * Permutes and adust pixel valus of an inout tensor.
 *  @param[in/out] inputTensor 1-channel greyscale input tensor to be adjusted.
 */
void adjustInputGreyTensor(at::Tensor& inputTensor)
{
    inputTensor = inputTensor.permute(PERMUTE_DIM);
    /* do the shenanigans: Imagenet's mean and std pixel values */
    inputTensor[0][0] = inputTensor[0][0].sub_(0.445).div_(0.269);
}

} // end of anonymouse namespace

TorchInference::TorchInference() : mChannels(1)
{
}

TorchInference::~TorchInference()
{
}

void TorchInference::initialise(const std::string& pathToModel, const int width, const int height, const int channels)
{
    at::Tensor output;
    mChannels = channels;

    /* Initialise sizes array */
    mSizes = {1, height, width, channels};
    /* Initialise temporary buffer */
    mFloatImage = cv::Mat(height, width, CV_MAKETYPE(CV_16F, channels));
    /* Load the model */
    mModule = torch::jit::load(pathToModel);
    /* Upload model to GPU */
    mModule.to(torch::kCUDA, torch::kFloat16);
    /* Invoke processImage once to fully initialise the module. Reuse mInputTensor */
    if (channels == 1)
    {
        processGreyImage(false, cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0)),
                         cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0)), output);
    }
    else
    {
        processImage(false, cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0)), output);
    }
}

at::Tensor& TorchInference::processImage(const bool tta, const cv::Mat& inputImage, at::Tensor& output)
{
    at::Tensor inputTensor;
    convertImage(inputImage, inputTensor);

    if (tta)
    {
        output = mModule.forward({torch::cat({inputTensor, inputTensor.fliplr()}, 0)}).toTensor().detach().to(torch::kCPU);
    }
    else
    {
        output = mModule.forward({inputTensor}).toTensor().detach().to(torch::kCPU);
    }

    return output;
}

at::Tensor& TorchInference::processGreyImage(const bool tta, const cv::Mat& leftImage, const cv::Mat& rightImage, at::Tensor& output)
{
    at::Tensor batch;
    at::Tensor leftInputTensor, rightInputTensor;

    convertImage(leftImage, leftInputTensor);
    convertImage(rightImage, rightInputTensor);

    if (tta)
    {
        batch = torch::cat({leftInputTensor, rightInputTensor, leftInputTensor.flip({3}), rightInputTensor.flip({3})}, 0);
    }
    else
    {
        batch = torch::cat({leftInputTensor, rightInputTensor}, 0);
    }
    output = mModule.forward({batch}).toTensor().detach().to(torch::kCPU);
    return output;
}

void TorchInference::convertImage(const cv::Mat& image, at::Tensor& tensor)
{
    image.convertTo(mFloatImage, CV_MAKETYPE(CV_16F, mChannels), 1.0f / 255.0f);
    tensor = torch::from_blob(mFloatImage.data, mSizes, at::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU)).to(torch::kCUDA);
    if (mChannels == 1)
    {
        adjustInputGreyTensor(tensor);
    }
    else
    {
        adjustInputRGBTensor(tensor);
    }
}