/** Implementation file for finding Otsu's threshold
 *
 *  \file ipcv/otsus_threshold/OtsusThreshold.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 8 Sep 2018
 */

#include "OtsusThreshold.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"


using namespace std;

namespace ipcv {

/** Find Otsu's threshold for each channel of a 3-channel (color) image
 *
 *  \param[in] src          source cv::Mat of CV_8UC3
 *  \param[out] threshold   threshold values for each channel of a 3-channel
 *                          color image in cv::Vec3b
 */
bool OtsusThreshold(const cv::Mat& src, cv::Vec3b& threshold) {
  threshold = cv::Vec3b(0, 0, 0);
   

  // histogram and histogram to pdf functions from 180
  cv::Mat histogram;
  ipcv::Histogram(src, histogram);
  cv::Mat pdf;
  ipcv::HistogramToPdf(histogram, pdf);

  // variables
  double mu_t;
  double mu_k;
  double omega_k;
  double best_threshold;
  double max = 0;
  double variance;

  // iterate for color channels
  for (int ch = 0; ch < src.channels(); ch++) {

    // reset mu_t for each channel
    mu_t = 0;

    // set mu_t 
    for (int idx = 0; idx < 256; idx++) {
      mu_t += idx * pdf.at<double> (ch, idx);
    }

    // reset variables
    mu_k = 0;
    omega_k = 0;
    max = 0;
    best_threshold = 0;

    // iterate along threshold
    for (int th = 0; th < 255; th++) {
      
      // summation of pdf, summation of pdf times threshold
      mu_k += th * pdf.at<double>(ch, th);
      omega_k += pdf.at<double>(ch, th);


      variance = (mu_t * omega_k - mu_k) * (mu_t * omega_k - mu_k) / (omega_k * (1 - omega_k));


      // find best threshold
      if (variance > max) {
        max = variance;
        best_threshold = th;
      }
      
      }
      
      threshold[ch] = best_threshold;

    

  }
  return true;
}
}





