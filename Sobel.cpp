/** Implementation file for image filtering
 *
 *  \file ipcv/spatial_filtering/Filter2D.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 20 Sep 2018
 */

#include "Sobel.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace ipcv {

bool Sobel(const cv::Mat& src, cv::Mat& dst, const int delta,
              const BorderMode border_mode, const uint8_t border_value) {

    dst = cv::Mat::zeros(src.size(), CV_32FC2);
    cv::Mat dst_x(src.size(), CV_8UC3);
    cv::Mat dst_y(src.size(), CV_8UC3);
/*
    if (border_mode == BorderMode::CONSTANT) {
        cv::copyMakeBorder(src, new, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, border_value);
    } else {
        cv::copyMakeBorder(src, new, pad_y, pad_y, pad_x, pad_x, cv::BORDER_REPLICATE);
    }
*/

    cv::Mat sobely = cv::Mat_<float>(3, 3, CV_32F);
    sobely.at<float>(0, 0) = -1;
    sobely.at<float>(0, 1) = -2;
    sobely.at<float>(0, 2) = -1;
    sobely.at<float>(1, 0) = 0;
    sobely.at<float>(1, 1) = 0;
    sobely.at<float>(1, 2) = 0;
    sobely.at<float>(2, 0) = -1;
    sobely.at<float>(2, 1) = 0;
    sobely.at<float>(2, 2) = 1;

    //(-1, -2, -1, 0, 0, 0, 1, 2, 1);

    cv::Mat sobelx = cv::Mat_<float>(3, 3, CV_32F);
    sobelx.at<float>(0, 0) = -1;
    sobelx.at<float>(0, 1) = 0;
    sobelx.at<float>(0, 2) = 1;
    sobelx.at<float>(1, 0) = -2;
    sobelx.at<float>(1, 1) = 0;
    sobelx.at<float>(1, 2) = 2;
    sobelx.at<float>(2, 0) = -1;
    sobelx.at<float>(2, 1) = 0;
    sobelx.at<float>(2, 2) = 1;


    float Gy;
    float Gx;
    float pixel;
    float src_x;
    float src_y;

    int sum_x;
    int sum_y;
    int half_r = sobelx.rows / 2;
    int half_c = sobelx.cols / 2;

    for (int y = half_r; y <= src.rows - half_r - 1; y++) {
        for (int x = half_c; x <= src.cols - half_c - 1; x++) {
            sum_x = delta;
            sum_y = delta;

            for (int t = -half_r; t <= half_r; t++) {
                for (int s = -half_c; s <= half_c; s++) {

                    sum_x += src.at<float>(y + t, x + s) * sobelx.at<float>(t + half_r, s + half_c);
                    sum_y += src.at<float>(y + t, x + s) * sobely.at<float>(t + half_r, s + half_c);
                }
            }
            dst_x.at<float>(y, x) = sum_x;
            dst_y.at<float>(y, x) = sum_y;

        }
    }

    cv::Mat dst_xy[2] = {dst_x, dst_y};
    cv::merge(dst_xy, 2, dst);


    return true;
}
}
