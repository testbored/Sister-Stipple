#include "../includes/CudaLloydBackend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        checkCuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

__global__ void assignPixelsKernel(const float* density,
                                   int width,
                                   int height,
                                   const float* pointX,
                                   const float* pointY,
                                   int pointCount,
                                   float* sumX,
                                   float* sumY,
                                   float* sumWeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float weight = density[y * width + x];
    if (weight <= 0.0f) {
        return;
    }

    int nearest = 0;
    float nearestDistance = (pointX[0] - x) * (pointX[0] - x) +
                            (pointY[0] - y) * (pointY[0] - y);
    for (int point = 1; point < pointCount; ++point) {
        const float dx = pointX[point] - x;
        const float dy = pointY[point] - y;
        const float distance = dx * dx + dy * dy;
        if (distance < nearestDistance) {
            nearest = point;
            nearestDistance = distance;
        }
    }

    // atomicAdd diperlukan karena banyak pixel dapat menjadi anggota satu
    // Voronoi region. Semua akumulasi tetap berada di memori GPU.
    atomicAdd(&sumX[nearest], static_cast<float>(x) * weight);
    atomicAdd(&sumY[nearest], static_cast<float>(y) * weight);
    atomicAdd(&sumWeight[nearest], weight);
}

__global__ void updateCentroidsKernel(float* pointX,
                                      float* pointY,
                                      const float* sumX,
                                      const float* sumY,
                                      const float* sumWeight,
                                      unsigned int* maximumShiftSquaredBits,
                                      int pointCount) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= pointCount || sumWeight[point] <= 0.0f) {
        return;
    }

    const float newX = sumX[point] / sumWeight[point];
    const float newY = sumY[point] / sumWeight[point];
    const float dx = newX - pointX[point];
    const float dy = newY - pointY[point];
    atomicMax(maximumShiftSquaredBits, __float_as_uint(dx * dx + dy * dy));
    pointX[point] = newX;
    pointY[point] = newY;
}

} // namespace

namespace cuda_backend {

RunStatistics runLloyd(const cv::Mat& density,
                       std::vector<float>& pointX,
                       std::vector<float>& pointY,
                       int iterations,
                       float epsilon) {
    if (density.empty() || density.type() != CV_32FC1) {
        throw std::invalid_argument("CUDA backend memerlukan density map CV_32FC1 yang tidak kosong.");
    }
    if (pointX.empty() || pointX.size() != pointY.size() || iterations < 0 || epsilon < 0.0f) {
        throw std::invalid_argument("Data titik atau jumlah iterasi CUDA tidak valid.");
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        throw std::runtime_error("CUDA backend dipilih, tetapi tidak ada GPU CUDA yang tersedia.");
    }

    const int width = density.cols;
    const int height = density.rows;
    const int pointCount = static_cast<int>(pointX.size());
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;

    DeviceBuffer<float> deviceDensity(pixelCount);
    DeviceBuffer<float> devicePointX(pointCount);
    DeviceBuffer<float> devicePointY(pointCount);
    DeviceBuffer<float> deviceSumX(pointCount);
    DeviceBuffer<float> deviceSumY(pointCount);
    DeviceBuffer<float> deviceSumWeight(pointCount);
    DeviceBuffer<unsigned int> deviceMaximumShiftSquared(1);

    checkCuda(cudaMemcpy(deviceDensity.get(), density.ptr<float>(), pixelCount * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy density to GPU");
    checkCuda(cudaMemcpy(devicePointX.get(), pointX.data(), pointCount * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy pointX to GPU");
    checkCuda(cudaMemcpy(devicePointY.get(), pointY.data(), pointCount * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy pointY to GPU");

    const dim3 pixelBlock(16, 16);
    const dim3 pixelGrid((width + pixelBlock.x - 1) / pixelBlock.x,
                         (height + pixelBlock.y - 1) / pixelBlock.y);
    constexpr int pointBlockSize = 256;
    const int pointGridSize = (pointCount + pointBlockSize - 1) / pointBlockSize;

    RunStatistics statistics;
    const float epsilonSquared = epsilon * epsilon;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        checkCuda(cudaMemset(deviceSumX.get(), 0, pointCount * sizeof(float)), "reset sumX");
        checkCuda(cudaMemset(deviceSumY.get(), 0, pointCount * sizeof(float)), "reset sumY");
        checkCuda(cudaMemset(deviceSumWeight.get(), 0, pointCount * sizeof(float)), "reset sumWeight");
        checkCuda(cudaMemset(deviceMaximumShiftSquared.get(), 0, sizeof(unsigned int)),
                  "reset maximum shift");

        assignPixelsKernel<<<pixelGrid, pixelBlock>>>(deviceDensity.get(), width, height,
                                                       devicePointX.get(), devicePointY.get(), pointCount,
                                                       deviceSumX.get(), deviceSumY.get(),
                                                       deviceSumWeight.get());
        checkCuda(cudaGetLastError(), "launch assignPixelsKernel");

        updateCentroidsKernel<<<pointGridSize, pointBlockSize>>>(devicePointX.get(), devicePointY.get(),
                                                                   deviceSumX.get(), deviceSumY.get(),
                                                                   deviceSumWeight.get(),
                                                                   deviceMaximumShiftSquared.get(), pointCount);
        checkCuda(cudaGetLastError(), "launch updateCentroidsKernel");

        unsigned int maximumShiftSquaredBits = 0;
        checkCuda(cudaMemcpy(&maximumShiftSquaredBits, deviceMaximumShiftSquared.get(),
                             sizeof(unsigned int), cudaMemcpyDeviceToHost),
                  "copy maximum shift from GPU");
        ++statistics.iterationsExecuted;
        float maximumShiftSquared = 0.0f;
        static_assert(sizeof(maximumShiftSquared) == sizeof(maximumShiftSquaredBits));
        std::memcpy(&maximumShiftSquared, &maximumShiftSquaredBits, sizeof(float));
        if (maximumShiftSquared <= epsilonSquared) {
            statistics.converged = true;
            break;
        }
    }

    checkCuda(cudaMemcpy(pointX.data(), devicePointX.get(), pointCount * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy pointX from GPU");
    checkCuda(cudaMemcpy(pointY.data(), devicePointY.get(), pointCount * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "copy pointY from GPU");
    return statistics;
}

} // namespace cuda_backend
