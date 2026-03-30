/** Implementation file for computing a CDF from a histogram
 *
 *  \file ipcv/utils/HistogramToCdf.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 17 Mar 2018
 */

#include "HistogramToCdf.h"
#include "imgs/ipcv/utils/HistogramToPdf.h"

namespace ipcv {

void HistogramToCdf(const cv::Mat& h, cv::Mat& cdf) {
  cv::Mat pdf;
  HistogramToPdf(h, pdf);  
  cdf.create(h.size(), CV_64F);
  cdf.setTo(0);

  for (int ch = 0; ch < 3; ++ch) {
    double cumulative_sum = 0.0;

      for (int idx = 0; idx < 256; ++idx) {
        cumulative_sum += pdf.at<double>(ch, idx); 
          cdf.at<double>(ch, idx) = cumulative_sum;
      }

    cdf.row(ch) /= cumulative_sum;
  }
}
}
