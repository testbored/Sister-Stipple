#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem> 

cv::Mat getImageGrayscale(const std::string& path);

float getDensity(uchar input);

cv::Mat assignDensity(const cv::Mat& input);

std::string resolveInputPath();

bool validateInputPath(const std::string& path);







