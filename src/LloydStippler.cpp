#include "../includes/LloydStippler.hpp"
#include <cmath>
#include <random>

Stippler::Stippler(int count, int iter, int eps){
    this->pointCount = count;
    this->iterations = iter;
    this->epsilon = eps;
}

void Stippler::createMapDensity(const std::string& path){
    this->density = assignDensity(getImageGrayscale(path));
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

void Stippler::calculateNewCentroid(){

    float sumX[pointCount];
    float sumY[pointCount];
    float sumWeight[pointCount];
    
    for(int y = 0; y < density.rows; ++y){
        for(int x = 0; x < density.cols; ++x){
            int point = findNearestPoint(x, y);
            
            float weight = density.at<float>(y, x);
            sumX[point] += x * weight;
            sumY[point] += y * weight;
            sumWeight[point] += weight;
        }
    }

    for(int i = 0; i < pointCount; i++){
        if(sumWeight[i] > 0.0f){
            pointX[i] = sumX[i] / sumWeight[i];
            pointY[i] = sumY[i] / sumWeight[i];
        }
    }
}

void Stippler::runLloyd(const std::string& path){
    createMapDensity(path);

    std::mt19937 rng(std::random_device{}());
    
    std::uniform_real_distribution<float> randX(0, density.cols);
    std::uniform_real_distribution<float> randY(0, density.rows);

    for(int i = 0; i < pointCount; i++){
        pointX.emplace_back(randX(rng));
        pointY.emplace_back(randY(rng));
    }

    for(int i = 0; i < iterations; i++){
        calculateNewCentroid();
    }
}
