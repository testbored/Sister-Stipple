#include "../includes/imageIO.hpp"

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

float getDensity(uchar input){
    return 255.0f - static_cast<float>(input);
}

cv::Mat assignDensity(const cv::Mat& input){
    cv::Mat density(input.rows, input.cols, CV_32F);

    for(int y = 0; y < input.rows; ++y){
        for(int x = 0; x < input.cols; ++x){
            density.at<float>(y, x) = getDensity(input.at<uchar>(y, x));
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
