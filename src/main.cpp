#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

constexpr int esc_key = 27;

int main() {
    auto net = cv::dnn::readNetFromONNX("__internal/weights/last.onnx");
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap;
    if(!cap.open(0)) {
        std::cout << "Failed to open camera!\n";
        return 0;
    }

    std::string class_names[2] = {"face", "mask"};

    while(true) {
        auto start = std::chrono::high_resolution_clock::now();

        // get frame
        cv::Mat frame;
        cap.read(frame);

        // resize to 640x640
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // predict                
        // std::vector<cv::Mat> outputs;
        // net.forward(outputs, net.getUnconnectedOutLayersNames());

        // for(auto& mat: outputs) {
        //     std::cout << "mat: " << mat.size[0][0] << "\n";
        // }

        /*// draw predictions
        cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        for (int i = 0; i < detectionMat.rows; i++){
			int class_id = detectionMat.at<float>(i, 1);
			float confidence = detectionMat.at<float>(i, 2);  

            std::cout << "Class id: " << class_id << "\n";
            std::cout << "Class: " << class_names[class_id - 1] << "\n";
            std::cout << "Conf : " << confidence << "\n\n";   
			
			// // Check if the detection is of good quality
			// if (confidence > 0){
			// 	int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			// 	int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			// 	int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
			// 	int box_height = 640;
            //     // static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
            //     // std::cout << static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y) << "\n";
			// 	cv::rectangle(frame, cv::Point(box_x, box_y), cv::Point(box_x + box_width, box_y + box_height), cv::Scalar(255, 255, 255), 2);
			// 	cv::putText(frame, class_names[class_id - 1].c_str(), cv::Point(box_x, box_y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
			// }
		} */

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << "Time spent: " << duration.count() << "\n";

        // show
        cv::imshow("vid", frame);
        if(cv::waitKey(300) == esc_key) {
            break;
        }
    }

    cv::destroyAllWindows();
    cap.release();

    return 0;
}
