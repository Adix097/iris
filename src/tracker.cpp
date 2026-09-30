#include "iris/tracker.hpp"
#include <limits>
#include <algorithm>

namespace iris {

    static cv::Point2f center(const cv::Rect& r) {
        return {r.x + r.width / 2.0f, r.y + r.height / 2.0f};
    }
    
    std::vector<TrackedFace> CentroidTracker::update(const std::vector<Detection>& detections) {
        std::vector<bool> used(detections.size(), false);
    
        // try to match each existing track to the nearest unclaimed detection
        for (auto& track : tracks_) {
            double best_dist = std::numeric_limits<double>::max();
            int best_idx = -1;
            cv::Point2f track_c = center(track.box);
            for (size_t i = 0; i < detections.size(); ++i) {
                if (used[i]) continue;
                double d = cv::norm(track_c - center(detections[i].box));
                if (d < best_dist) { best_dist = d; best_idx = static_cast<int>(i); }
            }
            if (best_idx >= 0 && best_dist <= max_dist_) {
                track.box = detections[best_idx].box;
                track.missed_frames = 0;
                used[best_idx] = true;
            } else {
                ++track.missed_frames;  // no matching detection this frame
            }
        }
    
        // any unclaimed detection is a new face
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!used[i]) tracks_.push_back({next_id_++, detections[i].box, 0});
        }
    
        // drop tracks that have been missing too long (person left the frame)
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackedFace& t) { return t.missed_frames > max_missed_; }), tracks_.end());
        return tracks_;
    }

}  // namespace iris
