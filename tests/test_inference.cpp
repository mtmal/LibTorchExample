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

void loadModel(const std::string& modelPath, const cv::Mat& image, TorchInference& torchInference)
{
    int64 time2;
    int64 time1 = cv::getTickCount();
    torchInference.initialise(modelPath, image.cols, image.rows, image.channels());
    time2 = cv::getTickCount();
    printf("Model loaded in %f \n", static_cast<double>(time2 - time1) / cv::getTickFrequency());
}

void processCPU(const cv::Mat& image, TorchInference& torchInference, torch::Tensor& output)
{
    int64 time2;
    int64 time1 = cv::getTickCount();
    switch (image.channels())
    {
        case 1:
            output = torchInference.processGreyImage(image, output).flatten();
            break;
        case 3:
            output = torchInference.processImage(image, output).flatten();
            break;
        default:
            printf("Unsupported number of channels %d \n", image.channels());
            break;
    }
    time2 = cv::getTickCount();
    printf("Estimated steering: %f, estimated throttle: %f, processed in: %f \n", 
            output[0].item().toFloat(), output[1].item().toFloat(), static_cast<double>(time2 - time1) / cv::getTickFrequency());
}

int main(int argc, char** argv)
{
    /** Input image */
    cv::Mat image;
    /** Wrapper class to perform Torch inference */
    TorchInference torchInference;
    /** Testing output */
    torch::Tensor output;

    if (argc > 1)
    {
        if (argc > 2 && (0 == strcmp(argv[2], "grey") || 0 == strcmp(argv[2], "gray")))
        {
            image = cv::imread("./0.000000_-1.000000_599.jpg", cv::IMREAD_GRAYSCALE);
        }
        else
        {
            image = cv::imread("./0.453627_-1.000000_c8155f0a-fd0e-11ec-b27a-3413e86352f4.jpg", cv::IMREAD_UNCHANGED);
        }
        
        loadModel(argv[1], image, torchInference);

        for (int i = 0; i < 10; ++i)
        {
            processCPU(image, torchInference, output);
        }
    }
    else
    {
        puts("Please provide a path to .pt model as the first argument.");
    }

    return 0;
}