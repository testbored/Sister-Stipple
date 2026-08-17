#pragma once
#include "imageIO.hpp"

#include <cstdint>
#include <vector>

struct RunStatistics {
    double milliseconds = 0.0;
    int iterationsExecuted = 0;
    bool converged = false;
};

class Stippler{
    private:
        int pointCount;
        int iterations;
        float epsilon;
        std::vector<float> pointX;
        std::vector<float> pointY;
        cv::Mat density;

        void initializePoints(std::uint32_t seed);
    
    public:
        Stippler(int count, int iter, float eps);
        void createMapDensity(const std::string& path);
        int findNearestPoint(int x, int y);
        float calculateNewCentroid();
        float calculateNewOMP();
        RunStatistics runLloyd(const std::string& path, std::uint32_t seed = 42);
        RunStatistics runLloydOMP(const std::string& path, std::uint32_t seed = 42);
        RunStatistics runLloydCUDA(const std::string& path, std::uint32_t seed = 42);
        RunStatistics createProgressGif(const std::string& path,
                                        const std::string& gifPath,
                                        int delayCentiseconds = 8,
                                        std::uint32_t seed = 42);
        cv::Mat renderStippleImage(int pointRadius = 1) const;
        void saveStippleImage(const std::string& outputPath, int pointRadius = 1) const;
};
