#pragma once
#include "imageIO.hpp"


class Stippler{
    private:
        int pointCount;
        int iterations;
        int epsilon;
        cv::Mat density;
    
    public:
        Stippler(int count, int iter, int eps);
        void CreateMapDensity(const std::string& path);
        
};