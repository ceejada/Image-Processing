/** Implementation file for image filtering
 *
 *  \file ipcv/spatial_filtering/Filter2D.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 20 Sep 2018
 */

#include "Filter2D.h"

#include <iostream>

using namespace std;

namespace ipcv {

/** Correlates an image with the provided kernel
 *
 *  \param[in] src          source cv::Mat of CV_8UC3
 *  \param[out] dst         destination cv::Mat of ddepth type
 *  \param[in] ddepth       desired depth of the destination image
 *  \param[in] kernel       convolution kernel (or rather a correlation
 *                          kernel), a single-channel floating point matrix
 *  \param[in] anchor       anchor of the kernel that indicates the relative
 *                          position of a filtered point within the kernel;
 *                          the anchor should lie within the kernel; default
 *                          value (-1,-1) means that the anchor is at the
 *                          kernel center
 *  \param[in] delta        optional value added to the filtered pixels
 *                          before storing them in dst
 *  \param[in] border_mode  pixel extrapolation method
 *  \param[in] border_value value to use for constant border mode
 */
bool Filter2D(const cv::Mat& src, cv::Mat& dst, const int ddepth,
              const cv::Mat& kernel, const cv::Point anchor, const int delta,
              const BorderMode border_mode, const uint8_t border_value) {

  dst = cv::Mat::zeros(src.size(), ddepth);

  int pad_y = 1;
  int pad_x = 1;
  cv::Mat padded(src.size(), ddepth);
/*
    if (border_mode == BorderMode::CONSTANT) {
        cv::copyMakeBorder(src, padded, pad_y, pad_y, pad_x, pad_x, cv::BORDER_CONSTANT, border_value);
    } else {
        cv::copyMakeBorder(src, padded, pad_y, pad_y, pad_x, pad_x, cv::BORDER_REPLICATE);
    }
*/
  float src_y;
  float src_x;

  float pixel;
  double sum;

  float kernel_value;
  // iterate through channels
  for (int ch = 0; ch < src.channels(); ch++) {

    // iterate through pixels
    for (int y = ((kernel.cols - 1) / 2); y < src.rows - ((kernel.cols - 1) / 2); y++) {
      for (int x = ((kernel.rows - 1) / 2); x < src.cols - ((kernel.rows - 1) / 2); x++) {
        sum = 0;

        for (int t = 0; t < kernel.rows; t++) {
          for (int s = 0; s < kernel.cols; s++) {
            src_x = x + s - anchor.x;
            src_y = y + t - anchor.y;

            pixel = src.at<cv::Vec3b>(src_y, src_x)[ch];
            kernel_value = kernel.at<float>(t, s);

            sum += kernel_value * pixel;
          }
        }
        sum = std::min(std::max(sum, 0.0), 255.0);
        //sum = std::clamp(sum, 0.0, 255.0);
        dst.at<cv::Vec3b>(y, x)[ch] = sum + delta;
      }
    }
  }

  return true;
}
}
