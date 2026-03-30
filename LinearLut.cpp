/** Implementation file for image enhancement using linear histogram statistics
 *
 *  \file ipcv/histogram_enhancement/LinearLut.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 3 Sep 2018
 */

#include "LinearLut.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace ipcv {

/** Create a 3-channel (color) LUT using linear histogram enhancement
 *
 *  \param[in] src          source cv::Mat of CV_8UC3
 *  \param[in] percentage   the total percentage to remove from the tails
 *                          of the histogram to find the extremes of the
 *                          linear enhancemnt function
 *  \param[out] lut         3-channel look up table in cv::Mat(3, 256)
 */
bool LinearLut(const cv::Mat& src, const int percentage, cv::Mat& lut) {

  // Insert your code here
  cv::Mat histogram;
  // IMGS 180 function Histogram
  ipcv::Histogram(src, histogram);
  cv::Mat cdf;
  //IMGS 180 function HistogramToCdf
  ipcv::HistogramToCdf(histogram, cdf);
  
  //set mix and max percentages, multiply to convert to decimal form, and divide so that its half on each side
  double min = percentage * 0.002;
  double max = 1 - min;

  cv::Mat cdfMin = cv::Mat::zeros(cdf.size(), CV_64F);
  cv::Mat cdfMax = cv::Mat::zeros(cdf.size(), CV_64F);

  // absolute difference, onto two new matrices
  cv::absdiff(cdf, min, cdfMin);
  cv::absdiff(cdf, max, cdfMax);

  //variables to be used in loop
  double DCoutMax = 255;
  double DCoutMin = 0;
  double slope;
  double intercept;
  double value;

  // create matrix lut
  lut = cv::Mat::zeros(src.channels(), 256, CV_8UC1);

  for (int ch = 0; ch < src.channels(); ch++) {
    // find the min and max locations
    cv::Point cdfMinLoc;
    cv::Point cdfMaxLoc;
    cv::minMaxLoc(cdfMin.row(ch), NULL, NULL, &cdfMinLoc, NULL);
    cv::minMaxLoc(cdfMax.row(ch), NULL, NULL, &cdfMaxLoc, NULL);

    slope = (DCoutMax - DCoutMin) / (cdfMaxLoc.x - cdfMinLoc.x);
    intercept = DCoutMax - (slope * cdfMaxLoc.x);

    // map high and low values to 0 and 255, then set lut to the value of mx+b
    for (int idx = 0; idx < 255; idx++) {
      value = slope * idx + intercept;
      if (value <= 0) {
        value = 0;
      }
      if (value >= 255) {
        value = 255;
      }
      lut.at<uchar>(ch, idx) = static_cast<uchar>(value);
    }
  }
  return true;
}
}
