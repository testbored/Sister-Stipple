#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace cuda_backend {

struct RunStatistics {
    int iterationsExecuted = 0;
    bool converged = false;
};

// Menjalankan seluruh iterasi Lloyd di GPU. Density diunggah sekali dan
// koordinat titik tetap berada di GPU sampai semua iterasi selesai.
RunStatistics runLloyd(const cv::Mat& density,
                       std::vector<float>& pointX,
                       std::vector<float>& pointY,
                       int iterations,
                       float epsilon);

} // namespace cuda_backend
