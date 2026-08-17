#include "../includes/LloydStippler.hpp"
#include "../includes/CudaLloydBackend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace {

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\\"'\\\"'";
        } else {
            quoted += character;
        }
    }
    return quoted + "'";
}

void makeGif(const std::filesystem::path& frameDirectory,
             const std::string& gifPath,
             int delayCentiseconds) {
    const std::string frames = shellQuote(frameDirectory.string()) + "/frame_*.png";
    const std::string command = "convert -delay " + std::to_string(delayCentiseconds) +
                                " -loop 0 " + frames + " " + shellQuote(gifPath);
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("Gagal membuat GIF. Pastikan ImageMagick ('convert') terpasang.");
    }
}

} // namespace

Stippler::Stippler(int count, int iter, float eps){
    if (count <= 0 || iter < 0 || eps < 0) {
        throw std::invalid_argument("Jumlah titik harus positif; iterasi dan epsilon tidak boleh negatif.");
    }

    this->pointCount = count;
    this->iterations = iter;
    this->epsilon = eps;
}

void Stippler::createMapDensity(const std::string& path){
    this->density = assignDensity(getImageGrayscale(path));
}

void Stippler::initializePoints(std::uint32_t seed) {
    pointX.clear();
    pointY.clear();
    pointX.reserve(pointCount);
    pointY.reserve(pointCount);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> randX(0.0f, static_cast<float>(density.cols - 1));
    std::uniform_real_distribution<float> randY(0.0f, static_cast<float>(density.rows - 1));
    for (int i = 0; i < pointCount; ++i) {
        pointX.emplace_back(randX(rng));
        pointY.emplace_back(randY(rng));
    }
}

int Stippler::findNearestPoint(int x, int y){

    int near = 0;
    double curr = pow((pointX[0] - x), 2) + pow((pointY[0] - y), 2);
    for(int i = 1; i < pointCount; ++i){
        double temp = pow((pointX[i] - x), 2) + pow((pointY[i] - y), 2);
        if(curr > temp){
            near = i;
            curr = temp;
        }
    }

    return near;
}

float Stippler::calculateNewCentroid(){
    if (pointX.size() != static_cast<std::size_t>(pointCount) ||
        pointY.size() != static_cast<std::size_t>(pointCount)) {
        throw std::logic_error("Titik belum diinisialisasi dengan benar.");
    }

    std::vector<double> sumX(pointCount, 0.0);
    std::vector<double> sumY(pointCount, 0.0);
    std::vector<double> sumWeight(pointCount, 0.0);
    
    for(int y = 0; y < density.rows; ++y){
        for(int x = 0; x < density.cols; ++x){
            int point = findNearestPoint(x, y);
            
            const float weight = density.at<float>(y, x);
            sumX[point] += x * weight;
            sumY[point] += y * weight;
            sumWeight[point] += weight;
        }
    }

    float maximumShift = 0.0f;
    for(int i = 0; i < pointCount; i++){
        if(sumWeight[i] > 0.0){
            const float newX = static_cast<float>(sumX[i] / sumWeight[i]);
            const float newY = static_cast<float>(sumY[i] / sumWeight[i]);
            const float dx = newX - pointX[i];
            const float dy = newY - pointY[i];
            maximumShift = std::max(maximumShift, std::sqrt(dx * dx + dy * dy));
            pointX[i] = newX;
            pointY[i] = newY;
        }
    }
    return maximumShift;
}

float Stippler::calculateNewOMP(){
    if (pointX.size() != static_cast<std::size_t>(pointCount) ||
        pointY.size() != static_cast<std::size_t>(pointCount)) {
        throw std::logic_error("Titik belum diinisialisasi dengan benar.");
    }

    const int threadCount = omp_get_max_threads();
    const std::size_t partialSize =
        static_cast<std::size_t>(threadCount) * static_cast<std::size_t>(pointCount);
    std::vector<double> partialSumX(partialSize, 0.0);
    std::vector<double> partialSumY(partialSize, 0.0);
    std::vector<double> partialSumWeight(partialSize, 0.0);

    #pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        const std::size_t offset =
            static_cast<std::size_t>(threadId) * static_cast<std::size_t>(pointCount);

        #pragma omp for schedule(static)
        for(int y = 0; y < density.rows; ++y){
            for(int x = 0; x < density.cols; ++x){
                const int point = findNearestPoint(x, y);
                const std::size_t index = offset + static_cast<std::size_t>(point);

                const float weight = density.at<float>(y, x);
                partialSumX[index] += x * weight;
                partialSumY[index] += y * weight;
                partialSumWeight[index] += weight;
            }
        }
    }

    float maximumShift = 0.0f;
    #pragma omp parallel for reduction(max:maximumShift) schedule(static)
    for(int i = 0; i < pointCount; i++){
        double sumX = 0.0;
        double sumY = 0.0;
        double sumWeight = 0.0;

        for (int thread = 0; thread < threadCount; ++thread) {
            const std::size_t index =
                static_cast<std::size_t>(thread) * static_cast<std::size_t>(pointCount) +
                static_cast<std::size_t>(i);
            sumX += partialSumX[index];
            sumY += partialSumY[index];
            sumWeight += partialSumWeight[index];
        }

        if(sumWeight > 0.0){
            const float newX = static_cast<float>(sumX / sumWeight);
            const float newY = static_cast<float>(sumY / sumWeight);
            const float dx = newX - pointX[i];
            const float dy = newY - pointY[i];
            maximumShift = std::max(maximumShift, std::sqrt(dx * dx + dy * dy));
            pointX[i] = newX;
            pointY[i] = newY;
        }
    }
    return maximumShift;
}

RunStatistics Stippler::runLloyd(const std::string& path, std::uint32_t seed){
    createMapDensity(path);
    initializePoints(seed);

    RunStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i){
        const float shift = calculateNewCentroid();
        ++statistics.iterationsExecuted;
        if (shift <= epsilon) {
            statistics.converged = true;
            break;
        }
    }
    statistics.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return statistics;
}

RunStatistics Stippler::runLloydOMP(const std::string& path, std::uint32_t seed){
    createMapDensity(path);
    initializePoints(seed);

    RunStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i){
        const float shift = calculateNewOMP();
        ++statistics.iterationsExecuted;
        if (shift <= epsilon) {
            statistics.converged = true;
            break;
        }
    }
    statistics.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return statistics;
}

RunStatistics Stippler::runLloydCUDA(const std::string& path, std::uint32_t seed){
    createMapDensity(path);
    initializePoints(seed);

    RunStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    const cuda_backend::RunStatistics cudaStatistics =
        cuda_backend::runLloyd(density, pointX, pointY, iterations, epsilon);
    statistics.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    statistics.iterationsExecuted = cudaStatistics.iterationsExecuted;
    statistics.converged = cudaStatistics.converged;
    return statistics;
}

void Stippler::saveStippleImage(const std::string& outputPath, int pointRadius) const {
    if (density.empty() || pointX.size() != pointY.size()) {
        throw std::logic_error("Tidak ada hasil stippling untuk disimpan.");
    }
    if (pointRadius <= 0) {
        throw std::invalid_argument("Radius titik harus positif.");
    }

    const std::filesystem::path path(outputPath);
    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            throw std::runtime_error("Tidak dapat membuat direktori output: " + error.message());
        }
    }

    const cv::Mat output = renderStippleImage(pointRadius);
    if (!cv::imwrite(outputPath, output)) {
        throw std::runtime_error("Gagal menyimpan gambar output: " + outputPath);
    }
}

cv::Mat Stippler::renderStippleImage(int pointRadius) const {
    if (density.empty() || pointX.size() != pointY.size()) {
        throw std::logic_error("Tidak ada hasil stippling untuk dirender.");
    }
    if (pointRadius <= 0) {
        throw std::invalid_argument("Radius titik harus positif.");
    }

    cv::Mat output(density.rows, density.cols, CV_8UC1, cv::Scalar(255));
    for (std::size_t i = 0; i < pointX.size(); ++i) {
        const int x = std::clamp(static_cast<int>(std::lround(pointX[i])), 0, density.cols - 1);
        const int y = std::clamp(static_cast<int>(std::lround(pointY[i])), 0, density.rows - 1);
        cv::circle(output, cv::Point(x, y), pointRadius, cv::Scalar(0), cv::FILLED, cv::LINE_AA);
    }
    return output;
}

RunStatistics Stippler::createProgressGif(const std::string& path,
                                           const std::string& gifPath,
                                           int delayCentiseconds,
                                           std::uint32_t seed) {
    if (delayCentiseconds <= 0) {
        throw std::invalid_argument("Delay GIF harus positif.");
    }

    createMapDensity(path);
    initializePoints(seed);

    const std::filesystem::path gifFilesystemPath(gifPath);
    const std::filesystem::path frameDirectory = gifFilesystemPath.string() + ".frames";
    std::error_code error;
    std::filesystem::remove_all(frameDirectory, error);
    if (error || !std::filesystem::create_directories(frameDirectory, error) || error) {
        throw std::runtime_error("Tidak dapat membuat direktori frame GIF.");
    }

    const auto saveFrame = [this, &frameDirectory](int frameNumber) {
        std::ostringstream name;
        name << "frame_" << std::setw(6) << std::setfill('0') << frameNumber << ".png";
        if (!cv::imwrite((frameDirectory / name.str()).string(), renderStippleImage())) {
            throw std::runtime_error("Gagal menyimpan frame GIF.");
        }
    };

    RunStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    try {
        saveFrame(0);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const float shift = calculateNewOMP();
            ++statistics.iterationsExecuted;
            saveFrame(statistics.iterationsExecuted);
            if (shift <= epsilon) {
                statistics.converged = true;
                break;
            }
        }
        makeGif(frameDirectory, gifPath, delayCentiseconds);
    } catch (...) {
        std::filesystem::remove_all(frameDirectory, error);
        throw;
    }
    std::filesystem::remove_all(frameDirectory, error);
    statistics.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return statistics;
}
