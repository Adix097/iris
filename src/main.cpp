#include "iris/camera.hpp"
#include "iris/virtualcam.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    iris::CameraConfig cfg;
    std::string out_dev; // empty -> no virtual camera
    bool mirror = true; // preview only
    bool view = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) cfg.device = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) out_dev = argv[++i];
        else if (std::strcmp(argv[i], "--no-mirror") == 0) mirror = false;
        else if (std::strcmp(argv[i], "--no-view") == 0) view = false;
    }

    iris::Camera cam(cfg);
    if (!cam.open()) return 1;

    const auto& a = cam.actual();
    std::printf("opened %s at %dx%d @ %d fps\n", a.device.c_str(), a.width, a.height, a.fps);

    iris::VirtualCam vcam(out_dev);
    if (!out_dev.empty()) {
        if (!vcam.open(a.width, a.height)) return 1;
        std::printf("streaming to %s\n", out_dev.c_str());
    }

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point from, clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };

    auto window_start = clock::now();
    int frames = 0;
    double read_sum = 0.0, out_sum = 0.0, view_sum = 0.0;

    cv::Mat frame, flipped;
    while (true) {
        auto t0 = clock::now();
        if (!cam.read(frame)) break; // blocks until the camera has a frame
        auto t1 = clock::now();

        if (!out_dev.empty() && !vcam.write(frame)) std::fprintf(stderr, "vcam write failed\n");
        auto t2 = clock::now();

        if (view) {
            if (mirror) {
                cv::flip(frame, flipped, 1); // 1 -> around the vertical axis
                cv::imshow("iris", flipped);
            } else {
                cv::imshow("iris", frame);
            }
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }
        auto t3 = clock::now();

        read_sum += ms(t0, t1);
        out_sum  += ms(t1, t2);
        view_sum += ms(t2, t3);
        ++frames;

        if (ms(window_start, t2) >= 1000.0) {
            std::printf("fps: %2d  read(): %5.1f ms  out: %4.1f ms  view: %4.1f ms\n", frames, read_sum / frames, out_sum / frames, view_sum / frames);
            frames = 0;
            read_sum = out_sum = view_sum = 0.0;
            window_start = t3;
        }
    }
    return 0;
}
