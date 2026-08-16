#include "../includes/LloydStippler.hpp"

Stippler::Stippler(int count, int iter, int eps){
    this->pointCount = count;
    this->iterations = iter;
    this->epsilon = eps;
}

void Stippler::CreateMapDensity(const std::string& path){
    this->density = assignDensity(getImageGrayscale(path));
}

