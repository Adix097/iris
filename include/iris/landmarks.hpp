#pragma once
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>

namespace iris {
    // lens ROI approximated as a rotated ellipse: axis-aligned to the eye line
    // hopefully it stays correct when the head tilts
    struct LensRegion {
        cv::Point2f center;
        cv::Size2f axes; // half-width, half-height
        float angle_deg; // angle of eye line vs horizontal
    };

    struct FaceLandmarks {
        std::vector<cv::Point2f> points; // 68 points, dlib layout
        LensRegion left_lens;
        LensRegion right_lens;
    };

    class LandmarkDetector {
        private:
            cv::Ptr<cv::face::Facemark> facemark_;
    
        public:
            explicit LandmarkDetector(const std::string& model_path);
            bool detect(const cv::Mat& bgr, const cv::Rect& face_box, FaceLandmarks& out); // one call per tracked face box returns false if fitting failed
    };
} // namespace iris
