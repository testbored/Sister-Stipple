#include "../includes/LloydStippler.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

Stippler::Stippler(int count, int iter, int eps){
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

    for(int i = 0; i < pointCount; i++){
        if(sumWeight[i] > 0.0){
            pointX[i] = static_cast<float>(sumX[i] / sumWeight[i]);
            pointY[i] = static_cast<float>(sumY[i] / sumWeight[i]);
        }
    }
}

void Stippler::calculateNewOMP(){
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

    #pragma omp parallel for schedule(static)
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
            pointX[i] = static_cast<float>(sumX / sumWeight);
            pointY[i] = static_cast<float>(sumY / sumWeight);
        }
    }
}

void Stippler::runLloyd(const std::string& path){
    createMapDensity(path);

    pointX.clear();
    pointY.clear();
    pointX.reserve(pointCount);
    pointY.reserve(pointCount);

    std::mt19937 rng(std::random_device{}());
    
    std::uniform_real_distribution<float> randX(0.0f, static_cast<float>(density.cols - 1));
    std::uniform_real_distribution<float> randY(0.0f, static_cast<float>(density.rows - 1));

    for(int i = 0; i < pointCount; i++){
        pointX.emplace_back(randX(rng));
        pointY.emplace_back(randY(rng));
    }

    for(int i = 0; i < iterations; i++){
        calculateNewCentroid();
    }
}

void Stippler::runLloydOMP(const std::string& path){
    createMapDensity(path);

    pointX.clear();
    pointY.clear();
    pointX.reserve(pointCount);
    pointY.reserve(pointCount);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> randX(0.0f, static_cast<float>(density.cols - 1));
    std::uniform_real_distribution<float> randY(0.0f, static_cast<float>(density.rows - 1));

    for(int i = 0; i < pointCount; i++){
        pointX.emplace_back(randX(rng));
        pointY.emplace_back(randY(rng));
    }

    for(int i = 0; i < iterations; i++){
        calculateNewOMP();
    }
}
