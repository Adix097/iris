#include "iris/camera.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace iris {
    // ioctl can be interrupted by a signal; retry instead of failing
    static int xioctl(int fd, unsigned long request, void* arg) {
        int r;

        do { 
            r = ioctl(fd, request, arg); 
        } while (r == -1 && errno == EINTR);

        return r;
    }

    Camera::~Camera() {
        if (fd_ < 0) return;

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);

        for (auto& b : buffers_) {
            if (b.start) {
                munmap(b.start, b.length);
            }
        }

        ::close(fd_);
    }

    bool Camera::open() {
        fd_ = ::open(cfg_.device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) { 
            std::perror("open"); 
            return false; 
        }

        v4l2_capability cap{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0 ||
            !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
            !(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::fprintf(stderr, "%s is not a streaming capture device\n", cfg_.device.c_str());
            return false;
        }

        // pixel format + size
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = cfg_.width;
        fmt.fmt.pix.height = cfg_.height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) { 
            std::perror("VIDIOC_S_FMT"); 
            return false; 
        }

        if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
            std::fprintf(stderr, "camera refused MJPG\n");
            return false;
        }

        cfg_.width = static_cast<int>(fmt.fmt.pix.width);
        cfg_.height = static_cast<int>(fmt.fmt.pix.height);

        // frame rate
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe = {1, static_cast<uint32_t>(cfg_.fps)};

        if (xioctl(fd_, VIDIOC_S_PARM, &parm) == 0 && parm.parm.capture.timeperframe.numerator) {
            cfg_.fps = static_cast<int>(parm.parm.capture.timeperframe.denominator / parm.parm.capture.timeperframe.numerator);
        }

        // ask the kernel for 4 buffers and map them into address space
        // the camera writes JPEG data straight into this memory: no copy
        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
            std::perror("VIDIOC_REQBUFS");
            return false;
        }

        buffers_.resize(req.count);
        for (unsigned i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                std::perror("VIDIOC_QUERYBUF");
                return false;
            }

            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            
            if (buffers_[i].start == MAP_FAILED) {
                buffers_[i].start = nullptr;
                std::perror("mmap");
                return false;
            }

            if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                std::perror("VIDIOC_QBUF");
                return false;
            }
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            std::perror("VIDIOC_STREAMON");
            return false;
        }

        return true;
    }

    bool Camera::read(cv::Mat& frame) {
        for (;;) {
            pollfd p{fd_, POLLIN, 0};
            int r = poll(&p, 1, 2000);  // wait up to 2 s for a frame

            if (r <= 0) {
                std::fprintf(stderr, "capture timeout\n");
                return false;
            }

            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
                if (errno == EAGAIN) continue;
                std::perror("VIDIOC_DQBUF");
                return false;
            }

            // wrap the mapped JPEG bytes without copying then decode into frame
            cv::Mat jpeg(1, static_cast<int>(buf.bytesused), CV_8UC1, buffers_[buf.index].start);
            cv::imdecode(jpeg, cv::IMREAD_COLOR, &frame);

            // hand the buffer back to the kernel
            xioctl(fd_, VIDIOC_QBUF, &buf);

            if (!frame.empty()) return true;  // skip broken JPEG
        }
    }
}  // namespace iris
