#include "../includes/imageIO.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>


cv::Mat getImageGrayscale(const std::string& path){
    if (!validateInputPath(path)) {
        throw std::invalid_argument("Input path is not a readable image: " + path);
    }

    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat assignDensity(const cv::Mat& input, float gamma, float edgeWeight){
    if (input.empty() || input.type() != CV_8UC1 || gamma <= 0.0f || edgeWeight < 0.0f) {
        throw std::invalid_argument("Parameter density map tidak valid.");
    }

    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Mat gradientMagnitude;
    cv::Sobel(input, gradientX, CV_32F, 1, 0, 3);
    cv::Sobel(input, gradientY, CV_32F, 0, 1, 3);
    cv::magnitude(gradientX, gradientY, gradientMagnitude);

    double maximumGradient = 0.0;
    cv::minMaxLoc(gradientMagnitude, nullptr, &maximumGradient);
    const float gradientScale = maximumGradient > 0.0
        ? 1.0f / static_cast<float>(maximumGradient)
        : 0.0f;
    cv::Mat density(input.rows, input.cols, CV_32F);

    for(int y = 0; y < input.rows; ++y){
        for(int x = 0; x < input.cols; ++x){
            const float darkness = (255.0f - static_cast<float>(input.at<uchar>(y, x))) / 255.0f;
            const float toneDensity = std::pow(darkness, gamma);
            const float edgeDensity = gradientMagnitude.at<float>(y, x) * gradientScale;
            density.at<float>(y, x) = 255.0f * (toneDensity + edgeWeight * edgeDensity);
        }
    }

    return density;
}

std::string resolveInputPath(){
    std::string path;

    while (true) {
        std::cout << "Masukkan path gambar input: ";
        if (!std::getline(std::cin >> std::ws, path)) {
            throw std::runtime_error("Gagal membaca input path.");
        }

        if (validateInputPath(path)) {
            return path;
        }

        std::cerr << "Path tidak valid atau file bukan gambar yang dapat dibaca. Coba lagi.\n";
    }
}

bool validateInputPath(const std::string& path){
    if (path.empty()) {
        return false;
    }

    const std::filesystem::path inputPath(path);
    std::error_code error;
    if (!std::filesystem::is_regular_file(inputPath, error) || error) {
        return false;
    }

    return !cv::imread(path, cv::IMREAD_GRAYSCALE).empty();
}
