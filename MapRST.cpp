/** Implementation file for finding map coordinates for an RST transformation
 *
 *  \file ipcv/geometric_transformation/MapRST.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 26 Sep 2019
 */

#include "MapRST.h"

#include <iostream>
#include <cmath>

#include <Eigen/Dense>

using namespace std;

namespace ipcv {

/** Find the map coordinates (map1, map2) for an RST transformation
 *
 *  \param[in] src           source cv::Mat of CV_8UC3
 *  \param[in] angle         rotation angle (CCW) [radians]
 *  \param[in] scale_x       horizontal scale
 *  \param[in] scale_y       vertical scale
 *  \param[in] translation_x horizontal translation [+ right]
 *  \param[in] translation_y vertical translation [+ up]
 *  \param[out] map1         cv::Mat of CV_32FC1 (size of the destination map)
 *                           containing the horizontal (x) coordinates at
 *                           which to resample the source data
 *  \param[out] map2         cv::Mat of CV_32FC1 (size of the destination map)
 *                           containing the vertical (y) coordinates at
 *                           which to resample the source data
 */
bool MapRST(const cv::Mat src, const double angle, const double scale_x,
            const double scale_y, const double translation_x,
            const double translation_y, cv::Mat& map1, cv::Mat& map2) {

  // variables!
  float new_x;
  float new_y;  

  double cos_angle = cos(angle);
  double sin_angle = sin(angle);

  double center_x = src.cols / 2.0;
  double center_y = src.rows / 2.0;

  // create matrices for RST
  Eigen::Matrix3d Scaling;
  Scaling << 
    scale_x, 0, 0,
    0, scale_y, 0,
    0, 0, 1;

  Eigen::Matrix3d Rotation;
  Rotation << 
    cos_angle, sin_angle, 0,
    -sin_angle,  cos_angle, 0,
    0, 0, 1;

  Eigen::Matrix3d Translation;
  Translation << 
    1, 0, translation_x,
    0, 1, translation_y,
    0, 0, 1;

  Eigen::Matrix3d TranslateToOrigin;
  TranslateToOrigin << 
    1, 0, -center_x,
    0, 1, -center_y,
    0, 0, 1;


  // set original corners of image to resize
  std::vector<cv::Point2f> originalCorners = {
      cv::Point2f(0, 0),
      cv::Point2f(static_cast<float>(src.cols), 0),
      cv::Point2f(static_cast<float>(src.cols), static_cast<float>(src.rows)),
      cv::Point2f(0, static_cast<float>(src.rows))
  };

  // create affine (RST) and inverse
  Eigen::Matrix3d Affine = Scaling * Rotation * TranslateToOrigin;
  Eigen::Matrix3d inverse = Affine.inverse();


  // create new corners by multiplying originals by the matrix
  std::vector<cv::Point2f> newCorners(4);
  for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d corner(originalCorners[i].x, originalCorners[i].y, 1);
      Eigen::Vector3d newCorner = Affine * corner;
      newCorners[i] = cv::Point2f(newCorner[0], newCorner[1]);
  }

  // set min and max for x and y values
  float minX = std::min({newCorners[0].x, newCorners[1].x, newCorners[2].x, newCorners[3].x});
  float maxX = std::max({newCorners[0].x, newCorners[1].x, newCorners[2].x, newCorners[3].x});
  float minY = std::min({newCorners[0].y, newCorners[1].y, newCorners[2].y, newCorners[3].y});
  float maxY = std::max({newCorners[0].y, newCorners[1].y, newCorners[2].y, newCorners[3].y});

  // set new image size with min and maxes
  int new_width = static_cast<int>(maxX - minX);
  int new_height = static_cast<int>(maxY - minY);

  // create maps for x and y that are the size of the new image
  map1.create(new_height, new_width, CV_32FC1);
  map2.create(new_height, new_width, CV_32FC1);

  Eigen::Vector3d vector;

  // loop through x and y for pixels, multiply inverse by xy1 vector
  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      vector = inverse * Eigen::Vector3d(minX + x, minY + y, 1); 

      new_x = vector[0];
      new_y = vector[1];

      new_x -= translation_x; 
      new_y += translation_y; 

      map1.at<float>(y, x) = new_x;
      map2.at<float>(y, x) = new_y;

    }
  }


  return true;
}
}
