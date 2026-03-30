/** Implementation file for image enhancement using histogram matching
 *
 *  \file ipcv/histogram_enhancement/MatchingLut.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 3 Sep 2018
 */

#include "MatchingLut.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace ipcv {

/** Create a 3-channel (color) LUT using histogram matching
 *
 *  \param[in] src   source cv::Mat of CV_8UC3
 *  \param[in] h     the histogram in cv:Mat(3, 256) that the
 *                   source is to be matched to
 *  \param[out] lut  3-channel look up table in cv::Mat(3, 256)
 */
bool MatchingLut(const cv::Mat& src, const cv::Mat& h, cv::Mat& lut) {

  // Insert your code here
  cv::Mat histogram;
  cv::Mat cdf;
  //IMGS 180 Histogram function
  ipcv::Histogram(src, histogram);
  //IMGS 180 HistogramToCdf function
  ipcv::HistogramToCdf(histogram, cdf);

  cv::Mat matchCdf;
  //IMGS 180 HistogramToCdf function
  ipcv::HistogramToCdf(h, matchCdf);

  //create lut
  lut = cv::Mat::zeros(src.channels(), 256, CV_8UC1);

  //variables to be used in loops
  cv::Mat temp(1, 256, CV_8UC1);
  cv::Point cdfMinLoc;
  double value;

// iterate through channels and rows
  for (int ch = 0; ch < src.channels(); ch++) {
    for (int idx = 0; idx < 255; idx++) {
      // value for the cdf at each point, then matched onto the temp matrix, then find min location at x and set lut for each pixel
      value = cdf.at<double>(ch, idx); 
      cv::absdiff(matchCdf.row(ch), cv::Scalar(value), temp);  
      cv::minMaxLoc(temp, NULL, NULL, &cdfMinLoc, NULL);
      lut.at<uchar>(ch, idx) = static_cast<uchar>(cdfMinLoc.x);
    }
  }

  return true;
}
}
