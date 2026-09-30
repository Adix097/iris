#include "iris/glassesdetector.hpp"
#include <opencv2/imgproc.hpp>

namespace iris {
    bool GlassesDetector::present(const cv::Mat& bgr, const LensRegion& lens) {
        int pad = static_cast<int>(std::max(lens.axes.width, lens.axes.height) * 0.3f);
        cv::Rect roi(cv::Point(static_cast<int>(lens.center.x - lens.axes.width - pad),
                               static_cast<int>(lens.center.y - lens.axes.height - pad)),
                     cv::Point(static_cast<int>(lens.center.x + lens.axes.width + pad),
                               static_cast<int>(lens.center.y + lens.axes.height + pad)));
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 4 || roi.height < 4) return false;

        cv::Point2f local_center(lens.center.x - roi.x, lens.center.y - roi.y);

        cv::Mat filled = cv::Mat::zeros(roi.size(), CV_8UC1);
        cv::ellipse(filled, local_center, lens.axes, lens.angle_deg, 0, 360, 255, -1);

        int band = std::max(2, static_cast<int>(std::min(lens.axes.width, lens.axes.height) * 0.12f));
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {band * 2 + 1, band * 2 + 1});
        cv::Mat outer, inner;
        cv::dilate(filled, outer, kernel);
        cv::erode(filled, inner, kernel);
        cv::Mat ring = outer - inner;

        cv::Mat gray, edges;
        cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 40, 120);

        cv::Mat ring_edges;
        cv::bitwise_and(edges, ring, ring_edges);

        int ring_pixels = cv::countNonZero(ring);
        if (ring_pixels == 0) return false;
        float coverage = static_cast<float>(cv::countNonZero(ring_edges)) / ring_pixels;

        return coverage >= rim_coverage_threshold_;
    }
} // namespace iris
