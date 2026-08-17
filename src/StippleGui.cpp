#include "../includes/StippleGui.hpp"

#include "../includes/LloydStippler.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int windowWidth = 900;
constexpr int windowHeight = 640;
constexpr int firstFieldY = 85;
constexpr int fieldHeight = 38;
constexpr int fieldGap = 52;
constexpr int fieldX = 230;
constexpr int fieldWidth = 620;

struct GuiState {
    std::vector<std::string> labels = {
        "Path input", "Path output PNG", "Path output GIF (opsional)",
        "Jumlah titik", "Maksimal iterasi", "Epsilon", "Gamma density", "Bobot edge"
    };
    std::vector<std::string> values = {
        "", "stipple.png", "", "1000", "50", "0.1", "1.6", "0.25"
    };
    int activeField = 0;
    bool runRequested = false;
    std::string status = "Klik field, ketik nilai, lalu tekan Enter atau tombol Jalankan.";
};

int fieldY(int index) {
    return firstFieldY + index * fieldGap;
}

void drawGui(const GuiState& state) {
    cv::Mat canvas(windowHeight, windowWidth, CV_8UC3, cv::Scalar(38, 38, 38));
    cv::putText(canvas, "Sister Stipple - Lloyd Algorithm", cv::Point(30, 42),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(255, 255, 255), 2);

    for (int index = 0; index < static_cast<int>(state.labels.size()); ++index) {
        const int y = fieldY(index);
        cv::putText(canvas, state.labels[index], cv::Point(30, y + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1);
        const cv::Scalar border = index == state.activeField ? cv::Scalar(70, 200, 255)
                                                              : cv::Scalar(140, 140, 140);
        cv::rectangle(canvas, cv::Rect(fieldX, y, fieldWidth, fieldHeight), border, 2);
        cv::putText(canvas, state.values[index], cv::Point(fieldX + 10, y + 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(245, 245, 245), 1);
    }

    const cv::Rect runButton(650, 540, 200, 48);
    cv::rectangle(canvas, runButton, cv::Scalar(70, 160, 80), cv::FILLED);
    cv::putText(canvas, "Jalankan", cv::Point(700, 571), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(canvas, state.status, cv::Point(30, 620), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(190, 220, 190), 1);
    cv::imshow("Sister Stipple GUI", canvas);
}

void mouseCallback(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    auto& state = *static_cast<GuiState*>(userdata);
    for (int index = 0; index < static_cast<int>(state.labels.size()); ++index) {
        if (cv::Rect(fieldX, fieldY(index), fieldWidth, fieldHeight).contains(cv::Point(x, y))) {
            state.activeField = index;
            return;
        }
    }
    if (cv::Rect(650, 540, 200, 48).contains(cv::Point(x, y))) {
        state.runRequested = true;
    }
}

void execute(GuiState& state) {
    try {
        const int points = std::stoi(state.values[3]);
        const int iterations = std::stoi(state.values[4]);
        const float epsilon = std::stof(state.values[5]);
        const float gamma = std::stof(state.values[6]);
        const float edgeWeight = std::stof(state.values[7]);
        if (state.values[0].empty() || state.values[1].empty() || points <= 0 || iterations < 0 ||
            epsilon < 0.0f || gamma <= 0.0f || edgeWeight < 0.0f) {
            throw std::invalid_argument("Isi semua parameter wajib dengan nilai valid.");
        }

        state.status = "Memproses OpenMP...";
        drawGui(state);
        Stippler stippler(points, iterations, epsilon, gamma, edgeWeight);
        const RunStatistics result = stippler.runLloydOMP(state.values[0]);
        stippler.saveStippleImage(state.values[1]);

        if (!state.values[2].empty()) {
            Stippler animation(points, iterations, epsilon, gamma, edgeWeight);
            animation.createProgressGif(state.values[0], state.values[2]);
        }

        cv::Mat output = stippler.renderStippleImage();
        cv::imshow("Hasil Stippling", output);
        std::ostringstream status;
        status << "Selesai: " << result.iterationsExecuted << " iterasi, " << result.milliseconds << " ms.";
        state.status = status.str();
    } catch (const std::exception& error) {
        state.status = std::string("Error: ") + error.what();
    }
}

} // namespace

int runStippleGui() {
    GuiState state;
    cv::namedWindow("Sister Stipple GUI", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Sister Stipple GUI", mouseCallback, &state);

    while (true) {
        drawGui(state);
        const int key = cv::waitKey(25);
        if (key == 27) {
            break;
        }
        if (state.runRequested || key == '\r' || key == '\n') {
            state.runRequested = false;
            execute(state);
            continue;
        }
        if (key == '\t') {
            state.activeField = (state.activeField + 1) % static_cast<int>(state.values.size());
        } else if (key == 8 || key == 127) {
            if (!state.values[state.activeField].empty()) {
                state.values[state.activeField].pop_back();
            }
        } else if (key >= 32 && key <= 126) {
            state.values[state.activeField].push_back(static_cast<char>(key));
        }
    }

    cv::destroyAllWindows();
    return 0;
}
