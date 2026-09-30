#pragma once
#include "iris/facedetector.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace iris {

    struct TrackedFace {
        int id;
        cv::Rect box;
        int missed_frames = 0;  // consecutive frames this face wasn't matched
    };

    // assigns a stable ID to each detected face across frames by nearest-centroid
    // good enough for 2-3 people sitting roughly still in front of a camera
    class CentroidTracker {
        private:
            std::vector<TrackedFace> tracks_;
            int next_id_ = 0;
            int max_missed_;
            double max_dist_;

        public:
            explicit CentroidTracker(int max_missed_frames = 10, double max_match_dist = 80.0) : max_missed_(max_missed_frames), max_dist_(max_match_dist) {}
            std::vector<TrackedFace> update(const std::vector<Detection>& detections);  
    };

}  // namespace iris
