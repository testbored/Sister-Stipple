#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem> 

cv::Mat getImageGrayscale(const std::string& path);

double getDensity(uchar input);

cv::Mat assignDensity(cv::Mat input);

std::string resolveInputpath();

bool validateInputPath();







