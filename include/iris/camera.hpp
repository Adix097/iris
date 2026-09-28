#pragma once
#include <opencv2/core.hpp>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace iris {
    struct CameraConfig {
        std::string device = "/dev/video0";
        int width = 1280;
        int height = 720;
        int fps = 30;
    };

    class Camera {
        private:
            struct Buffer { void* start = nullptr; std::size_t length = 0; };

            CameraConfig cfg_;
            int fd_ = -1;
            std::vector<Buffer> buffers_;
            
        public:
            explicit Camera(CameraConfig cfg) : cfg_(std::move(cfg)) {}
            ~Camera();
            Camera(const Camera&) = delete;
            Camera& operator=(const Camera&) = delete;

            bool open();   
            bool read(cv::Mat& frame);

            // values the driver actually accepted
            const CameraConfig& actual() const { return cfg_; }
    };
}  // namespace iris
