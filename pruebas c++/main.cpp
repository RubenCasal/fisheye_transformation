#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

cv::Mat applyFisheyeEffect(const cv::Mat& src, const cv::Mat& K, const cv::Mat& D) {
    int width = src.cols;
    int height = src.rows;
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // Get focal lengths and principal point from K (Intrinsic Matrix)
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    // Get distortion coefficients
    double k1 = D.at<double>(0, 0);
    double k2 = D.at<double>(1, 0);
    double k3 = D.at<double>(2, 0);
    double k4 = D.at<double>(3, 0);

    // Loop over each pixel
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            // Convert pixel coordinates to normalized camera coordinates
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r2 = x * x + y * y; // Squared radius

            // Apply radial distortion model
            double radial_distortion = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;

            // Compute new distorted coordinates
            double x_distorted = x * radial_distortion;
            double y_distorted = y * radial_distortion;

            // Convert back to pixel coordinates
            int u_new = static_cast<int>(fx * x_distorted + cx);
            int v_new = static_cast<int>(fy * y_distorted + cy);

            // Check if new coordinates are within image bounds
            if (u_new >= 0 && u_new < width && v_new >= 0 && v_new < height) {
                dst.at<cv::Vec3b>(v, u) = src.at<cv::Vec3b>(v_new, u_new);
            }
        }
    }

    return dst;
}

int main() {
    // Load input image
    cv::Mat src = cv::imread("../img.webp");
    if (src.empty()) {
        std::cerr << "Error: Could not load the image!" << std::endl;
        return -1;
    }

    // Camera intrinsic matrix (K)
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        284.509100, 0.000000, 421.896335,
        0.000000, 282.941856, 398.100316,
        0.000000, 0.000000, 1.000000);

    // Distortion coefficients (D)
    cv::Mat D = (cv::Mat_<double>(4, 1) <<
        -0.014216, 0.060412, -0.054711, 0.011151);

    // Apply fisheye distortion
    cv::Mat fisheye_img = applyFisheyeEffect(src, K, D);

    // Save and display results
    cv::imwrite("fisheye_manual.webp", fisheye_img);
    cv::imshow("Fisheye Effect", fisheye_img);
    cv::waitKey(0);

    return 0;
}
