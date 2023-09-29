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
    int64 time1 = cv::getTickCount();
    torchInference.initialise(modelPath, image.cols, image.rows, image.channels());
    printf("Model loaded in %f \n", static_cast<double>(cv::getTickCount() - time1) / cv::getTickFrequency());
}

void getResults(const bool tta, const at::Tensor& tensor, float& meanSteering, float& meanThrottle)
{
    meanSteering = 0.0f;
    meanThrottle = 0.0f;
    if (static_cast<int>(tensor.size(0)) > 0)
    {
        for (int i = 0; i < static_cast<int>(tensor.size(0)); ++i)
        {
            if (tta && (i >= (static_cast<int>(tensor.size(0)) / 2)))
            {
                meanSteering -= tensor[i][0].item().toFloat();
            }
            else
            {
                meanSteering += tensor[i][0].item().toFloat();
            }
            meanThrottle += tensor[i][1].item().toFloat();
        }
        meanSteering /= static_cast<int>(tensor.size(0));
        meanThrottle /= static_cast<int>(tensor.size(0));
    }
}

void processMonoRGB(const bool tta, const cv::Mat& image, TorchInference& torchInference, at::Tensor& output)
{
    float steering = 0;
    float throttle = 0;
    int64 time1 = cv::getTickCount();
    torchInference.processImage(tta, image, output);
    getResults(tta, output, steering, throttle);
    printf("Estimated steering: %f, estimated throttle: %f, processed in: %f \n", 
            steering, throttle, static_cast<double>(cv::getTickCount() - time1) / cv::getTickFrequency());
}

void processStereoGrey(const bool tta, const cv::Mat& leftImage, const cv::Mat& rightImage, TorchInference& torchInference, at::Tensor& output)
{
    float steering = 0;
    float throttle = 0;
    int64 time1 = cv::getTickCount();
    torchInference.processGreyImage(tta, leftImage, rightImage, output);
    getResults(tta, output, steering, throttle);
    printf("Estimated steering: %f, estimated throttle: %f, processed in: %f \n", 
            steering, throttle, static_cast<double>(cv::getTickCount() - time1) / cv::getTickFrequency());
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        /** Wrapper class to perform Torch inference */
        TorchInference torchInference;
        /** Testing output */
        at::Tensor output;

        if (argc > 2 && (0 == strcmp(argv[2], "grey") || 0 == strcmp(argv[2], "gray")))
        {
            bool tta = ((argc > 3) && (0 == strcmp(argv[3], "tta")));
            cv::Mat leftImage  = cv::imread("./1.000000_-1.000000_988_left.jpg",  cv::IMREAD_GRAYSCALE);
            cv::Mat rightImage = cv::imread("./1.000000_-1.000000_988_right.jpg", cv::IMREAD_GRAYSCALE);
            loadModel(argv[1], leftImage, torchInference);
            for (int i = 0; i < 10; ++i)
            {
                processStereoGrey(tta, leftImage, rightImage, torchInference, output);
            }
        }
        else
        {
            bool tta = ((argc > 2) && (0 == strcmp(argv[2], "tta")));
            cv::Mat image = cv::imread("./0.453627_-1.000000_c8155f0a-fd0e-11ec-b27a-3413e86352f4.jpg", cv::IMREAD_UNCHANGED);
            loadModel(argv[1], image, torchInference);
            for (int i = 0; i < 10; ++i)
            {
                processMonoRGB(tta, image, torchInference, output);
            }
        }
    }
    else
    {
        puts("Please provide a path to .pt model as the first argument.");
    }

    return 0;
}