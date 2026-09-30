#pragma once
#include "iris/landmarks.hpp"
#include <opencv2/core.hpp>

namespace iris {
    class GlareDetector {
        private:
            float brightness_delta_; // how much brighter than local surroundings counts as suspicious (in 0-255 grayscale units)
            float close_kernel_frac_; // size of the morphological close, as a fraction of the lens size - merges nearby bright/speckled pixels into solid regions

        public:
            GlareDetector(float brightness_delta = 12.0f, float close_kernel_frac = 0.08f) : brightness_delta_(brightness_delta), close_kernel_frac_(close_kernel_frac) {}
            cv::Mat detect(const cv::Mat& bgr, const LensRegion& lens);
    };
} // namespace iris
