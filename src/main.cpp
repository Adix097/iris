#include "iris/camera.hpp"
#include "iris/virtualcam.hpp"
#include "iris/facedetector.hpp"
#include "iris/tracker.hpp"
#include "iris/landmarks.hpp"
#include "iris/glassesdetector.hpp"
#include "iris/glaredetector.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    iris::CameraConfig cfg;
    std::string out_dev;
    bool mirror = true;
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

    iris::FaceDetector detector("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    iris::CentroidTracker tracker(30);
    iris::LandmarkDetector landmarker("models/lbfmodel.yaml");
    iris::GlassesDetector glasses_detector;
    iris::GlareDetector glare_detector;

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point from, clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };

    auto window_start = clock::now();
    int frames = 0;
    double read_sum = 0.0, detect_sum = 0.0, out_sum = 0.0, view_sum = 0.0;

    cv::Mat frame, display;
    while (true) {
        auto t0 = clock::now();
        if (!cam.read(frame)) break;
        auto t1 = clock::now();

        auto detections = detector.detect(frame);
        auto tracked = tracker.update(detections);
        auto t2 = clock::now();

        std::vector<iris::FaceLandmarks> faces_landmarks;
        for (const auto& f : tracked) {
            iris::FaceLandmarks lm;
            if (landmarker.detect(frame, f.box, lm)) faces_landmarks.push_back(std::move(lm));
        }

        cv::Mat glare_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        for (const auto& lm : faces_landmarks) {
            if (glasses_detector.present(frame, lm.left_lens)) {
                cv::bitwise_or(glare_mask, glare_detector.detect(frame, lm.left_lens), glare_mask);
            }
            
            if (glasses_detector.present(frame, lm.right_lens)) {
                cv::bitwise_or(glare_mask, glare_detector.detect(frame, lm.right_lens), glare_mask);
            }
        }

        if (!out_dev.empty() && !vcam.write(frame)) std::fprintf(stderr, "vcam write failed\n");
        auto t3 = clock::now();

        if (view) {
            if (mirror) cv::flip(frame, display, 1);
            else frame.copyTo(display);
            
            // draw the box
            for (const auto& f : tracked) {
                cv::Rect box = f.box;
                if (mirror) box.x = display.cols - box.x - box.width;  // mirror the box position only
                cv::rectangle(display, box, {0, 255, 0}, 2);
                cv::putText(
                    display, "id " + std::to_string(f.id),
                    {box.x, box.y - 8}, 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, {0, 255, 0}, 2
                );
            }
            
            // draw the ellipse
            for (const auto& lm : faces_landmarks) {
                for (const auto* lens : {&lm.left_lens, &lm.right_lens}) {
                    cv::Point2f c = lens->center;
                    float angle = lens->angle_deg;

                    if (mirror) { 
                        c.x = display.cols - c.x; 
                        angle = 180.0f - angle; 
                    }

                    cv::ellipse(display, c, lens->axes, angle, 0, 360, {0, 200, 255}, 2);
                }
            }

            cv::Mat glare_display = glare_mask;
            if (mirror) cv::flip(glare_mask, glare_display, 1);
            display.setTo(cv::Scalar(0, 0, 255), glare_display); // paint red where glare detected

            cv::imshow("iris", display);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }
        auto t4 = clock::now();

        read_sum += ms(t0, t1);
        detect_sum += ms(t1, t2);
        out_sum += ms(t2, t3);
        view_sum += ms(t3, t4);
        ++frames;

        if (ms(window_start, t4) >= 1000.0) {
            std::printf("fps: %2d  read: %5.1f ms  detect: %5.1f ms  out: %4.1f ms  view: %4.1f ms\n",
                        frames, read_sum / frames, detect_sum / frames, out_sum / frames, view_sum / frames);
            frames = 0;
            read_sum = detect_sum = out_sum = view_sum = 0.0;
            window_start = t4;
        }
    }
    return 0;
}
