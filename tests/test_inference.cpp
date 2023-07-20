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

#include <opencv2/imgcodecs.hpp>
#include <TorchInference.h>

void loadModel(const cv::Mat& image, TorchInference& torchInference)
{
    int64 time2;
    int64 time1 = cv::getTickCount();
    torchInference.initialise("./drive_road_following_model_cpp.pt", image.cols, image.rows, 3);
    time2 = cv::getTickCount();
    printf("Model loaded in %f \n", static_cast<double>(time2 - time1) / cv::getTickFrequency());
}

void processCPU(const cv::Mat& image, TorchInference& torchInference, torch::Tensor& output)
{
    int64 time2;
    int64 time1 = cv::getTickCount();
    output = torchInference.processImage(image, output).flatten();
    time2 = cv::getTickCount();
    printf("Estimated: %f, processed in: %f \n", 
            output[0].item().toFloat(), static_cast<double>(time2 - time1) / cv::getTickFrequency());
}

int main()
{
    /** Loop iterator */
    int i;
    /** Input image */
    const cv::Mat image = cv::imread("./0.453627_-1.000000_c8155f0a-fd0e-11ec-b27a-3413e86352f4.jpg", cv::IMREAD_UNCHANGED);
    /** Wrapper class to perform Torch inference */
    TorchInference torchInference;
    /** Testing output */
    torch::Tensor output;

    loadModel(image, torchInference);

    for (i = 0; i < 10; ++i)
    {
        processCPU(image, torchInference, output);
    }

    return 0;
}