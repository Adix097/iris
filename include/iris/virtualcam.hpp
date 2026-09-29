#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace iris {
    class VirtualCam {
        private:
            std::string device_;
            int fd_ = -1;
            int width_ = 0, height_ = 0;
            cv::Mat yuyv_;
        
        public:
            explicit VirtualCam(std::string device = "/dev/video10") : device_(std::move(device)) {}
            ~VirtualCam();
            VirtualCam(const VirtualCam&) = delete;
            VirtualCam& operator = (const VirtualCam&) = delete;

             bool open(int width, int height);
             bool write(const cv::Mat& bgr);  // must match the size given to open()
    };
} // namespace iris
