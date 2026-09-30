#include "iris/landmarks.hpp"
#include <stdexcept>
#include <array>
#include <iterator>

namespace iris {
    LandmarkDetector::LandmarkDetector(const std::string& model_path) {
        facemark_ = cv::face::FacemarkLBF::create();
        facemark_->loadModel(model_path);
    }

    // dlib-style 68-point indices for the eyes
    // Layout reference: https://ibug.doc.ic.ac.uk/resources/300-W/
    static constexpr std::array<int, 6> kRightEye = {36, 37, 38, 39, 40, 41};
    static constexpr std::array<int, 6> kLeftEye  = {42, 43, 44, 45, 46, 47};

    // builds a lens ellipse from 6 eye-contour points: center = mean of the points
    // axes scaled from the eye's own width/height because of glasses
    // angle = outer-to-inner corner line
    static LensRegion eyeToLens(const std::vector<cv::Point2f>& pts, const std::array<int, 6>& idx) {
        cv::Point2f sum(0, 0);
        for (int i : idx) sum += pts[i];
        cv::Point2f eye_center = sum * (1.0f / 6.0f);

        cv::Point2f corner_vec = pts[idx[3]] - pts[idx[0]];
        float eye_width = static_cast<float>(cv::norm(corner_vec));
        float angle = std::atan2(corner_vec.y, corner_vec.x) * 180.0f / static_cast<float>(CV_PI);

        // lens center sits below the eye-opening center roughly by half the eye's
        cv::Point2f down(-std::sin(angle * static_cast<float>(CV_PI) / 180.0f), std::cos(angle * static_cast<float>(CV_PI) / 180.0f));
        cv::Point2f lens_center = eye_center + down * (eye_width * 0.18f);

        LensRegion lens;
        lens.center = lens_center;
        //* revisit if frames vary a lot in style
        lens.axes = cv::Size2f(eye_width * 0.95f, eye_width * 0.75f);
        lens.angle_deg = angle;
        return lens;
    }

    bool LandmarkDetector::detect(const cv::Mat& bgr, const cv::Rect& face_box, FaceLandmarks& out) {
        std::vector<cv::Rect> faces{face_box};
        std::vector<std::vector<cv::Point2f>> shapes;
        if (!facemark_->fit(bgr, faces, shapes) || shapes.empty() || shapes[0].size() != 68) {
            return false;
        }

        out.points = std::move(shapes[0]);
        out.right_lens = eyeToLens(out.points, kRightEye);
        out.left_lens = eyeToLens(out.points, kLeftEye);
        return true;
    }
} // namespace iris