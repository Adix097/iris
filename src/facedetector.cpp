#include "iris/facedetector.hpp"
#include <stdexcept>

namespace iris {

    FaceDetector::FaceDetector(const std::string& prototxt, const std::string& weights) {
        net_ = cv::dnn::readNetFromCaffe(prototxt, weights);
        if (net_.empty()) throw std::runtime_error("failed to load face detector model");

        // CPU-only
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<Detection> FaceDetector::detect(const cv::Mat& bgr, float conf_threshold) {
        // the network was trained on 300x300 inputs with these exact mean-subtraction values
        // changing them silently breaks accuracy
        cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
        net_.setInput(blob);
        cv::Mat out = net_.forward(); // shape: [1, 1, N, 7]

        std::vector<Detection> results;
        cv::Mat det(out.size[2], out.size[3], CV_32F, out.ptr<float>());

        for (int i = 0; i < det.rows; ++i) {
            float confidence = det.at<float>(i, 2);
            if (confidence < conf_threshold) continue;

            // output coords are normalized [0,1]; scale to actual pixel size
            int x1 = static_cast<int>(det.at<float>(i, 3) * bgr.cols);
            int y1 = static_cast<int>(det.at<float>(i, 4) * bgr.rows);
            int x2 = static_cast<int>(det.at<float>(i, 5) * bgr.cols);
            int y2 = static_cast<int>(det.at<float>(i, 6) * bgr.rows);

            cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
            box &= cv::Rect(0, 0, bgr.cols, bgr.rows); // clamp to frame bounds
            if (box.width > 0 && box.height > 0) results.push_back({box, confidence});
        }
        
        return results;
    }
}  // namespace iris
