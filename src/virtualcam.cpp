#include "iris/virtualcam.hpp"
#include <opencv2/imgproc.hpp>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace iris {

    VirtualCam::~VirtualCam() { 
        if (fd_ >= 0) ::close(fd_); 
    }

    bool VirtualCam::open(int width, int height) {
        fd_ = ::open(device_.c_str(), O_WRONLY);
        
        if (fd_ < 0) { 
            std::perror("open virtual cam"); 
            return false; 
        }

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        fmt.fmt.pix.bytesperline = width * 2; // YUYV = 2 bytes per pixel
        fmt.fmt.pix.sizeimage = width * height * 2;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) { 
            std::perror("VIDIOC_S_FMT (vcam)"); 
            return false;
        }

        width_ = width;
        height_ = height;
        return true;
    }

    bool VirtualCam::write(const cv::Mat& bgr) {
        if (bgr.cols != width_ || bgr.rows != height_) return false;
        cv::cvtColor(bgr, yuyv_, cv::COLOR_BGR2YUV_YUY2); // BGR -> packed YUYV (CV_8UC2)
        const size_t bytes = yuyv_.total() * yuyv_.elemSize();
        return ::write(fd_, yuyv_.data, bytes) == static_cast<ssize_t>(bytes);
    }

}  // namespace iris
