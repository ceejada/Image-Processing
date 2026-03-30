/** Implementation file for image quantization
 *
 *  \file ipcv/quantize/quantize.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu) and Carly Adams (cja5549@rit.edu)
 *  \date 4 September 2024
 */

#include "Quantize.h"
#include <cmath>
#include <iostream>

using namespace std;

/** Perform uniform grey-level quantization on a color image
 *
 *  \param[in] src                 source cv::Mat of CV_8UC3
 *  \param[in] quantization_levels the number of levels to which to quantize
 *                                 the image
 *  \param[out] dst                destination cv:Mat of CV_8UC3
 */
void Uniform(const cv::Mat& src, const int quantization_levels, cv::Mat& dst) {

  // quantization - divide by 256 divided by however many quantization levels there are
  double levels = 256 / static_cast<double>(quantization_levels);

  // iterate across 3 channels, and then rows and columns to get every pixel
  for (int idx = 0; idx < src.channels(); idx++) {
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
          // maps the source image divided by the levels at each pixel onto the destination
          dst.at<cv::Vec3b>(x, y) = (src.at<cv::Vec3b>(x, y) / levels);
        }
    }
  }
  
}

/** Perform improved grey scale quantization on a color image
 *
 *  \param[in] src                 source cv::Mat of CV_8UC3
 *  \param[in] quantization_levels the number of levels to which to quantize
 *                                 the image
 *  \param[out] dst                destination cv:Mat of CV_8UC3
 */
void Igs(const cv::Mat& src, const int quantization_levels, cv::Mat& dst) {

  // creates variables to be used within loop - remainder starts out as 0 and is changed, with remainder is set immediately
  double levels = 256 / static_cast<double>(quantization_levels);
  int remainder = 0;
  int with_remainder;

  // iterate across 3 channels, rows, and columns
  for (int idx = 0; idx < src.channels(); idx++) {
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
          // finds the value of the pixel + the remainder of the last pixel (starts at 0)
          with_remainder = src.at<cv::Vec3b>(x, y)[idx] + remainder;
          // remainder calculates the pixels that fall off during quantization to store and then add them to the next pixel
          remainder = fmod(with_remainder, static_cast<int>(levels));
          dst.at<cv::Vec3b>(x, y)[idx] = with_remainder / levels;
        }
    }
  }

}

namespace ipcv {

bool Quantize(const cv::Mat& src, const int quantization_levels,
              const QuantizationType quantization_type, cv::Mat& dst) {
  dst.create(src.size(), src.type());

  switch (quantization_type) {
    case QuantizationType::uniform:
      Uniform(src, quantization_levels, dst);
      break;
    case QuantizationType::igs:
      Igs(src, quantization_levels, dst);
      break;
    default:
      cerr << "Specified quantization type is unsupported" << endl;
      return false;
  }

  return true;
}
}
