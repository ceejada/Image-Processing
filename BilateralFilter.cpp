/** Implementation file for bilateral filtering
 *
 *  \file ipcv/bilateral_filtering/BilateralFilter.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 29 Sep 2018
 */

#include "BilateralFilter.h"
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;

namespace ipcv {

/** Bilateral filter an image
 *
 *  \param[in] src             source cv::Mat of CV_8UC3
 *  \param[out] dst            destination cv::Mat of ddepth type
 *  \param[in] sigma_distance  standard deviation of distance/closeness filter
 *  \param[in] sigma_range     standard deviation of range/similarity filter
 *  \param[in] radius          radius of the bilateral filter (if negative, use
 *                             twice the standard deviation of the distance/
 *                             closeness filter)
 *  \param[in] border_mode     pixel extrapolation method
 *  \param[in] border_value    value to use for constant border mode
 */
bool BilateralFilter(const cv::Mat& src, cv::Mat& dst,
                     const double sigma_distance, const double sigma_range,
                     const int radius, const BorderMode border_mode,
                     uint8_t border_value) {
  dst.create(src.size(), src.type());
  cv::Mat srcborder(src.size(), CV_8UC3);
  cv::Mat srcLAB(src.size(), CV_32FC3);
  cv::Mat dstLAB(src.size(), CV_32FC3);

  // if radius is negative, use twice the deviation of distance
  int new_radius;
  if (radius < 0) {
    new_radius = sigma_distance * 2;
  }
  else {
    new_radius = radius;
  }

  // border handling
  if (border_mode == BorderMode::REPLICATE) {
    cv::copyMakeBorder(src, srcborder, new_radius, new_radius, new_radius, new_radius, cv::BORDER_REPLICATE);
  } else {
    cv::copyMakeBorder(src, srcborder, new_radius, new_radius, new_radius, new_radius, cv::BORDER_CONSTANT, border_value);
  }

  // scale border by 255
  srcborder.convertTo(srcborder, CV_32FC3, 1.0 / 255.0);

  // color conversion to LAB
  cv::cvtColor(srcborder, srcLAB, cv::COLOR_BGR2Lab);


  // split srcLAB into three channels, then use L channel and create mat for filtered results
  std::vector<cv::Mat> lab_ch(3);
  cv::split(srcLAB, lab_ch);

  cv::Mat L_ch = lab_ch[0];
  cv::Mat filtered_L(L_ch.size(), L_ch.type());

  // compute closeness kernel
  float distance;
  cv::Mat closeness_kernel = cv::Mat::zeros(2 * new_radius + 1, 2 * new_radius + 1, CV_32FC1);
  for (int row = -new_radius; row <= new_radius; row++) {
    for (int col = -new_radius; col <= new_radius; col++) {
      distance = std::sqrt((row * row) + (col * col));
      closeness_kernel.at<float>(row + new_radius, col + new_radius) = std::exp(-0.5 * (distance / sigma_distance) * (distance / sigma_distance));
    }
  }

  // variables
  float center;
  float similarity;
  float result;
  float normalization_factor;
  float closeness;

  // create similarity kernel
  cv::Mat similarity_kernel(2 * new_radius + 1, 2 * new_radius + 1, CV_32F);

  // iterate through x and y
  for (int y = new_radius; y < srcLAB.rows - new_radius; y++) {
    for (int x = new_radius; x < srcLAB.cols - new_radius; x++) {
      // reset variables
      center = L_ch.at<float>(y, x);
      result = 0.0f;
      normalization_factor = 0.0f;

      // iterate again through radius size
      for (int i = -new_radius; i < new_radius; i++) {
        for (int j = -new_radius; j < new_radius; j++) {
          // compute similarity kernel
          distance = center - L_ch.at<float>(y + i, x + j);
          similarity_kernel.at<float>(i + new_radius, j + new_radius) = std::exp(-0.5f * (distance / sigma_range) * (distance / sigma_range));
        }
      }
      // iterate through radius size after forming similarity kernel
      for (int i = -new_radius; i < new_radius; i++) {
        for (int j = -new_radius; j < new_radius; j++) {
          // find similarity and closeness at location, then compute normalization factor by summing them
          closeness = closeness_kernel.at<float>(i + new_radius, j + new_radius);
          similarity = similarity_kernel.at<float>(i + new_radius, j + new_radius);
          normalization_factor += closeness * similarity;
          result += L_ch.at<float>(y + i, x + j) * closeness * similarity;
      }
    }

    // set filtered mat at (y,x) to result * 1 / normaliation factor
    filtered_L.at<float>(y, x) = result / normalization_factor;


    }
  }

  // set first lab channel to filtered result, then merge all lab channels to dstLAB
  lab_ch[0] = filtered_L;
  cv::merge(lab_ch, dstLAB);

  // convert dstLAB to BGR colorspace, resize dst to ensure that there are no borders
  cv::cvtColor(dstLAB, dst, cv::COLOR_Lab2BGR);
  dst.convertTo(dst, CV_8UC3, 255.0);
  dst = dst(cv::Rect(new_radius, new_radius, src.cols, src.rows));


  return true;
}
}
