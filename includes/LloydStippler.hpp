#pragma once
#include "imageIO.hpp"


class Stippler{
    private:
        int pointCount;
        int iterations;
        int epsilon;
        std::vector<float> pointX;
        std::vector<float> pointY;
        cv::Mat density;
    
    public:
        Stippler(int count, int iter, int eps);
        void createMapDensity(const std::string& path);
        int findNearestPoint(int x, int y);
        void calculateNewCentroid();
        void Stippler::runLloyd(const std::string& path);
};