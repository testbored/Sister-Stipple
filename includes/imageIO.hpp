#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem> 

cv::Mat getImageGrayscale(const std::string& path);

cv::Mat assignDensity(const cv::Mat& input, float gamma, float edgeWeight);

std::string resolveInputPath();

bool validateInputPath(const std::string& path);







