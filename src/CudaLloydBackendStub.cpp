#include "../includes/CudaLloydBackend.hpp"

#include <stdexcept>

namespace cuda_backend {

RunStatistics runLloyd(const cv::Mat&, std::vector<float>&, std::vector<float>&, int, float) {
    throw std::runtime_error(
        "CUDA backend tidak tersedia. Instal CUDA Toolkit dan konfigurasi ulang CMake.");
}

} // namespace cuda_backend
