#include "../includes/imageIO.hpp"


cv::Mat getImageGrayscale(const std::string& path){
    cv::Mat gray;
    cv::cvtColor(cv::imread(path), gray, cv::COLOR_BGR2GRAY);
    return gray;
}

double getDensity(uchar input){
    return 255.0f - static_cast<float>(input);
}

cv::Mat assignDensity(cv::Mat input){
    cv::Mat density(input.rows, input.cols, CV_32F);

    for(int y = 0; y < input.rows; ++y){
        for(int x = 0; x < input.cols; ++x){
            density.at<float>(y, x) = getDensity(input.at<uchar>(y, x));
        }
    }

    return density;
}

std::string resolveInputpath(){

}

bool validateInputPath(){

}