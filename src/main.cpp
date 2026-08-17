#include "../includes/LloydStippler.hpp"
#include "../includes/StippleGui.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string inputPath;
    std::string outputPath;
    int pointCount = 0;
    int iterations = 0;
    float epsilon = 0.0f;
    std::uint32_t seed = 42;
    std::string gifPath;
};

void printUsage(const char* executable) {
    std::cout << "Usage:\n  " << executable
              << " --input <image> --points <count> --iterations <count> --epsilon <value> --output <image>"
                 " [--gif <animation.gif>] [--seed <number>]\n"
                 "  " << executable << " --gui\n";
}

Options parseArguments(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        const std::string flag = argv[index];
        if (flag == "--help" || flag == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Nilai tidak ditemukan untuk " + flag);
        }

        const std::string value = argv[index + 1];
        if (flag == "--input") {
            options.inputPath = value;
        } else if (flag == "--output") {
            options.outputPath = value;
        } else if (flag == "--points") {
            options.pointCount = std::stoi(value);
        } else if (flag == "--iterations") {
            options.iterations = std::stoi(value);
        } else if (flag == "--epsilon") {
            options.epsilon = std::stof(value);
        } else if (flag == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value));
        } else if (flag == "--gif") {
            options.gifPath = value;
        } else {
            throw std::invalid_argument("Argumen tidak dikenal: " + flag);
        }
    }

    if (options.inputPath.empty() || options.outputPath.empty() || options.pointCount <= 0 ||
        options.iterations < 0 || options.epsilon < 0.0f) {
        throw std::invalid_argument("--input, --output, --points, --iterations, dan --epsilon harus valid.");
    }
    return options;
}

void printResult(const char* backend, const RunStatistics& statistics, double speedup) {
    std::cout << std::left << std::setw(10) << backend
              << std::right << std::setw(14) << std::fixed << std::setprecision(3)
              << statistics.milliseconds
              << std::setw(14) << statistics.iterationsExecuted
              << std::setw(14) << std::setprecision(2) << speedup << "x"
              << std::setw(14) << (statistics.converged ? "ya" : "tidak") << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--gui") {
            return runStippleGui();
        }
        const Options options = parseArguments(argc, argv);

        Stippler serial(options.pointCount, options.iterations, options.epsilon);
        const RunStatistics serialStatistics = serial.runLloyd(options.inputPath, options.seed);

        Stippler omp(options.pointCount, options.iterations, options.epsilon);
        const RunStatistics ompStatistics = omp.runLloydOMP(options.inputPath, options.seed);

        Stippler cuda(options.pointCount, options.iterations, options.epsilon);
        bool cudaAvailable = true;
        RunStatistics cudaStatistics;
        std::string cudaError;
        try {
            cudaStatistics = cuda.runLloydCUDA(options.inputPath, options.seed);
        } catch (const std::exception& error) {
            cudaAvailable = false;
            cudaError = error.what();
        }

        if (cudaAvailable) {
            cuda.saveStippleImage(options.outputPath);
        } else {
            omp.saveStippleImage(options.outputPath);
        }

        if (!options.gifPath.empty()) {
            Stippler animation(options.pointCount, options.iterations, options.epsilon);
            const RunStatistics animationStatistics = animation.createProgressGif(
                options.inputPath, options.gifPath, 8, options.seed);
            std::cout << "GIF: " << options.gifPath << " (" << animationStatistics.iterationsExecuted
                      << " iterasi, " << animationStatistics.milliseconds << " ms)\n";
        }

        std::cout << "\nBenchmark Lloyd Stippling\n"
                  << "Seed titik awal: " << options.seed << "\n"
                  << "Output: " << options.outputPath
                  << (cudaAvailable ? " (CUDA)\n" : " (OpenMP fallback)\n")
                  << std::left << std::setw(10) << "Backend"
                  << std::right << std::setw(14) << "Waktu (ms)"
                  << std::setw(14) << "Iterasi"
                  << std::setw(14) << "Speedup"
                  << std::setw(14) << "Konvergen" << '\n';

        printResult("Serial", serialStatistics, 1.0);
        printResult("OpenMP", ompStatistics, serialStatistics.milliseconds / ompStatistics.milliseconds);
        if (cudaAvailable) {
            printResult("CUDA", cudaStatistics,
                        serialStatistics.milliseconds / cudaStatistics.milliseconds);
        } else {
            std::cout << "CUDA       tidak tersedia: " << cudaError << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
