cmake_minimum_required(VERSION 3.16)

project(SisterStipple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Finds the system-installed OpenCV package and supplies its headers and libraries.
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

option(SISTER_STIPPLE_ENABLE_CUDA "Build the CUDA Lloyd backend when CUDA is available" ON)
include(CheckLanguage)
if(SISTER_STIPPLE_ENABLE_CUDA)
    # CUDA di WSL sering terpasang di /usr/local/cuda tetapi belum ada di PATH.
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(SISTER_STIPPLE_NVCC nvcc HINTS /usr/local/cuda/bin)
        if(SISTER_STIPPLE_NVCC)
            set(CMAKE_CUDA_COMPILER "${SISTER_STIPPLE_NVCC}" CACHE FILEPATH
                "CUDA compiler used by Sister Stipple")
        endif()
    endif()
    check_language(CUDA)
endif()

add_library(sister_stipple
    src/imageIO.cpp
    src/LloydStippler.cpp
)
target_include_directories(sister_stipple PUBLIC includes)
target_link_libraries(sister_stipple PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(sister-stipple src/main.cpp src/StippleGui.cpp)
target_link_libraries(sister-stipple PRIVATE sister_stipple)

if(SISTER_STIPPLE_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(sister_stipple PRIVATE src/CudaLloydBackend.cu)
    target_link_libraries(sister_stipple PRIVATE CUDA::cudart)
    set_target_properties(sister_stipple PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(sister_stipple PUBLIC SISTER_STIPPLE_HAS_CUDA=1)
else()
    target_sources(sister_stipple PRIVATE src/CudaLloydBackendStub.cpp)
    target_compile_definitions(sister_stipple PUBLIC SISTER_STIPPLE_HAS_CUDA=0)
    message(STATUS "CUDA Toolkit tidak ditemukan; CUDA backend akan memakai stub runtime.")
endif()
