/** Implementation file for computing an image histogram
 *
 *  \file ipcv/utils/Histogram.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 17 Mar 2018
 */

#include "Histogram.h"

namespace ipcv {

void Histogram(const cv::Mat& src, cv::Mat& h) {
  h = cv::Mat_<int>::zeros(3, 256); 
 
  // create variable pixels --> 3 byte vector for 3 channels of pixels (blue, green, red)
  cv::Vec3b pixels;
  
  // iterate along x and y, for the amount of rows and columns in the source image. set pixels to source image at (x, y) for each value. set h at pixel value for each channel.
  for (int x = 0; x < src.rows; x++) {
    for (int y = 0; y < src.cols; y++) {
      pixels = src.at<cv::Vec3b>(x, y);
      h.at<int>(0, pixels[0])++;
      h.at<int>(1, pixels[1])++;
      h.at<int>(2, pixels[2])++;
    }
  }
 
  // insert code: compute the 3-channel image histogram of the provided source image
  //
  // \param[in] src  source cv::Mat of CV_8UC3
  // \param[out] h  the grey level histogram for the source image
}
}
