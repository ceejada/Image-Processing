/** Implementation file for remapping source values to map locations
 *
 *  \file ipcv/geometric_transformation/Remap.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 15 Sep 2018
 */

#include "Remap.h"

#include <iostream>

using namespace std;

namespace ipcv {

/** Remap source values to the destination array at map1, map2 locations
 *
 *  \param[in] src            source cv::Mat of CV_8UC3
 *  \param[out] dst           destination cv::Mat of CV_8UC3 for remapped values
 *  \param[in] map1           cv::Mat of CV_32FC1 (size of the destination map)
 *                            containing the horizontal (x) coordinates at
 *                            which to resample the source data
 *  \param[in] map2           cv::Mat of CV_32FC1 (size of the destination map)
 *                            containing the vertical (y) coordinates at
 *                            which to resample the source data
 *  \param[in] interpolation  interpolation to be used for resampling
 *  \param[in] border_mode    border mode to be used for out of bounds pixels
 *  \param[in] border_value   border value to be used when constant border mode
 *                            is to be used
 */
bool Remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1,
           const cv::Mat& map2, const Interpolation interpolation,
           const BorderMode border_mode, const uint8_t border_value) {
  
  dst.create(map1.size(), src.type());

  // variables to store value at x and value at y
  float src_x;
  float src_y;

  // variables for bilinear
  float x1, x2, y1, y2;
  cv::Vec3b top_left;
  cv::Vec3b top_right;
  cv::Vec3b bottom_left;
  cv::Vec3b bottom_right;
  cv::Vec3b top;
  cv::Vec3b bottom;
  cv::Vec3b bilinear;

  // variables for nearest neighbor
  int nearest_x;
  int nearest_y;

  for (int x = 0; x < dst.rows; x++) {
    for (int y = 0; y < dst.cols; y++) {
      
      src_x = map1.at<float>(x, y);
      src_y = map2.at<float>(x, y);

      // border mode implementation, set out of bounds pizels to 0 or -1
      if (border_mode == BorderMode::REPLICATE) {

        if (src_x < 0) {
          src_x = 0;
        } else if (src_x >= src.cols) {
          src_x = src.cols - 1;
        }

        if (src_y < 0) {
          src_y = 0;
        } else if (src_y >= src.rows) {
          src_y = src.rows - 1;
        }
      }


      // nearest neighbor: rounds to nearest pixel
      if (interpolation == ipcv::Interpolation::NEAREST) {
        nearest_x = std::round(src_x);
        nearest_y = std::round(src_y);

        if (nearest_x >= 0 && nearest_x < src.rows && nearest_y >= 0 && nearest_y < src.cols) {
          dst.at<cv::Vec3b>(x, y) = src.at<cv::Vec3b>(nearest_y, nearest_x);
        } else {
          dst.at<cv::Vec3b>(x, y) = cv::Vec3b(border_value, border_value, border_value);
        }
      }

      // bilinear interpolation: averages missing pixels
      if (interpolation == ipcv::Interpolation::LINEAR) {
        x1 = std::floor(src_x);
        y1 = std::floor(src_y);
        x2 = std::ceil(src_x);
        y2 = std::ceil(src_y);

        float x_weight = src_x - x1;
        float y_weight = src_y - y1;

        top_left = src.at<cv::Vec3b>(y1, x1);
        top_right = src.at<cv::Vec3b>(y1, x2);
        bottom_left = src.at<cv::Vec3b>(y2, x1);
        bottom_right = src.at<cv::Vec3b>(y2, x2); 

        bilinear = top_left * (1 - x_weight) * (1 - y_weight) + 
                      top_right * x_weight * (1 - y_weight) + 
                      bottom_left * y_weight * (1 - x_weight) + 
                      bottom_right * x_weight * y_weight;

        // safety
        if (y1 >= 0 && y2 >= 0 && x1 >= 0 && x2 >= 0 && 
            y1 < src.rows && y2 < src.rows && 
            x1 < src.cols && x2 < src.cols) {
          dst.at<cv::Vec3b>(x, y) = bilinear;  
        } else {
          dst.at<cv::Vec3b>(x, y) = cv::Vec3b(border_value, border_value, border_value);
        }
      
      }
    }
  }

  return true;
}
}
