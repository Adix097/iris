#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace iris {

    struct Detection {
        cv::Rect box;
        float confidence;
    };

    class FaceDetector {
        private:
            cv::dnn::Net net_;

        public:
            FaceDetector(const std::string& prototxt, const std::string& weights);
            std::vector<Detection> detect(const cv::Mat& bgr, float conf_threshold = 0.6f); // Runs the detector on a BGR frame
    };

}  // namespace iris
