#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

constexpr int esc_key = 27;

int main() {
    //cv::dnn::Net net = cv::dnn::readNetFromONNX("__internal/weights/weights.onnx");
    auto net = cv::dnn::readNet("__internal/weights/last.onnx");
    cv::VideoCapture cap;
    if(!cap.open(0)) {
        std::cout << "Failed to open camera!\n";
        return 0;
    }

    while(true) {
        cv::Mat frame;
        cap.read(frame);

        cv::imshow("vid", frame);

        if(cv::waitKey(30) == esc_key) {
            break;
        }
    }

    cv::destroyAllWindows();
    cap.release();

    return 0;
}
