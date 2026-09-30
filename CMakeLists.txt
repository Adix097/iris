cmake_minimum_required(VERSION 3.25)
project(iris LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn face)

# Everything except main() lives in a library so tests/tools can reuse it later
add_library(iris_core src/camera.cpp src/virtualcam.cpp src/facedetector.cpp src/tracker.cpp src/landmarks.cpp src/glaredetector.cpp src/glassesdetector.cpp)
target_include_directories(iris_core PUBLIC include)
target_link_libraries(iris_core PUBLIC ${OpenCV_LIBS})
target_compile_options(iris_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(iris src/main.cpp)
target_link_libraries(iris PRIVATE iris_core)
target_compile_options(iris PRIVATE -Wall -Wextra -Wpedantic)
