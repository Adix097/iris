#include "iris/glaredetector.hpp"
#include <opencv2/imgproc.hpp>

namespace iris {
    cv::Mat GlareDetector::detect(const cv::Mat& bgr, const LensRegion& lens) {
        cv::Mat full_mask = cv::Mat::zeros(bgr.size(), CV_8UC1);

        int pad = static_cast<int>(std::max(lens.axes.width, lens.axes.height) * 0.3f);
        cv::Rect roi(cv::Point(static_cast<int>(lens.center.x - lens.axes.width - pad),
                               static_cast<int>(lens.center.y - lens.axes.height - pad)),
                     cv::Point(static_cast<int>(lens.center.x + lens.axes.width + pad),
                               static_cast<int>(lens.center.y + lens.axes.height + pad)));

        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 4 || roi.height < 4) return full_mask;

        cv::Mat lens_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
        cv::Point2f local_center(lens.center.x - roi.x, lens.center.y - roi.y);
        cv::ellipse(lens_mask, local_center, lens.axes, lens.angle_deg, 0, 360, 255, -1);

        cv::Mat gray;
        cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);

        cv::Mat local_mean;
        int k = std::max(5, static_cast<int>(std::min(roi.width, roi.height) * 0.15f) | 1);
        cv::boxFilter(gray, local_mean, CV_32F, {k, k});

        cv::Mat brightness_diff = gray - local_mean;
        cv::Mat bright_enough;
        cv::compare(brightness_diff, brightness_delta_, bright_enough, cv::CMP_GT);

        // close small gaps to merge nearby bright pixels into one solid region
        int close_k = std::max(3, static_cast<int>(std::min(roi.width, roi.height) * close_kernel_frac_) | 1);
        cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {close_k, close_k});
        cv::morphologyEx(bright_enough, bright_enough, cv::MORPH_CLOSE, close_kernel);

        cv::Mat roi_mask;
        cv::bitwise_and(bright_enough, lens_mask, roi_mask); // stay inside the lens ellipse only

        roi_mask.copyTo(full_mask(roi));
        return full_mask;
    }
} // namespace iris
