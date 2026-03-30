/** Implementation file for computing the PDF from a histogram
 *
 *  \file ipcv/utils/HistogramToPdf.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 17 Mar 2018
 */

#include "HistogramToPdf.h"
#include <iostream>

namespace ipcv {

void HistogramToPdf(const cv::Mat& h, cv::Mat& pdf) {
  pdf.create(h.size(), CV_64F);

  // must normalize (divide) to 1
    for (int i = 0; i < 256; i++) {
    std::cout << "TEST " << i << " " << pdf.at<int>(0, i) << std::endl;
  }
  cv::normalize(h, pdf, 1, 0, cv::NORM_L1, CV_64F);
  for (int i = 0; i < 256; i++) {
    std::cout << "TEST afterrr" << i << " " << pdf.at<int>(0, i) << std::endl;
  }

  pdf *= 3;
}
}

