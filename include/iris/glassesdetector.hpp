#pragma once
#include "iris/landmarks.hpp"
#include <opencv2/core.hpp>

namespace iris {
    class GlassesDetector {
        private:
            float rim_coverage_threshold_; // fraction of the ring band that must contain an edge to consider a lens frame present (0-1)

        public:
            explicit GlassesDetector(float rim_coverage_threshold = 0.08f) : rim_coverage_threshold_(rim_coverage_threshold) {}
            bool present(const cv::Mat& bgr, const LensRegion& lens);
    };
} // namespace iris
