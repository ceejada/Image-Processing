/** Implementation file for mapping a source quad on to a target quad
 *
 *  \file ipcv/geometric_transformation/MapQ2Q.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 12 Sep 2020
 */

#include "MapQ2Q.h"

#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

using namespace std;

namespace ipcv {

/** Find the source coordinates (map1, map2) for a quad to quad mapping
 *
 *  \param[in] src       source cv::Mat of CV_8UC3
 *  \param[in] tgt       target cv::Mat of CV_8UC3
 *  \param[in] src_vertices
 *                       vertices cv:Point of the source quadrilateral (CW)
 *                       which is to be mapped to the target quadrilateral
 *  \param[in] tgt_vertices
 *                       vertices cv:Point of the target quadrilateral (CW)
 *                       into which the source quadrilateral is to be mapped
 *  \param[out] map1     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the horizontal (x) coordinates at
 *                       which to resample the source data
 *  \param[out] map2     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the vertical (y) coordinates at
 *                       which to resample the source data
 */
bool MapQ2Q(const cv::Mat src, const cv::Mat tgt,
            const vector<cv::Point> src_vertices,
            const vector<cv::Point> tgt_vertices, cv::Mat& map1,
            cv::Mat& map2) {

  // Insert your code here
  Eigen::VectorXd src_x(4);
  Eigen::VectorXd src_y(4);
  Eigen::VectorXd tgt_x(4);
  Eigen::VectorXd tgt_y(4);

  Eigen::VectorXd srcVec(8);

  // fill vector
  for (size_t idx = 0; idx < 4; idx++) {
    cv::Point src = src_vertices.at(idx);
    cv::Point tgt = tgt_vertices.at(idx);

    src_x(idx) = src.x;
    src_y(idx) = src.y;
    tgt_x(idx) = tgt.x;
    tgt_y(idx) = tgt.y;

    srcVec(idx) = src.x;
    srcVec(idx + 4) = src.y;
  }

  // fill matrix 
  Eigen::MatrixXd a(8, 8);
  a << 
    tgt_x[0], tgt_y[0], 1, 0, 0, 0, -tgt_x[0] * src_x[0], -tgt_y[0] * src_x[0],
    tgt_x[1], tgt_y[1], 1, 0, 0, 0, -tgt_x[1] * src_x[1], -tgt_y[1] * src_x[1],
    tgt_x[2], tgt_y[2], 1, 0, 0, 0, -tgt_x[2] * src_x[2], -tgt_y[2] * src_x[2],
    tgt_x[3], tgt_y[3], 1, 0, 0, 0, -tgt_x[3] * src_x[3], -tgt_y[3] * src_x[3],
    0, 0, 0, tgt_x[0], tgt_y[0], 1, -tgt_x[0] * src_y[0], -tgt_y[0] * src_y[0],
    0, 0, 0, tgt_x[1], tgt_y[1], 1, -tgt_x[1] * src_y[1], -tgt_y[1] * src_y[1],
    0, 0, 0, tgt_x[2], tgt_y[2], 1, -tgt_x[2] * src_y[2], -tgt_y[2] * src_y[2],
    0, 0, 0, tgt_x[3], tgt_y[3], 1, -tgt_x[3] * src_y[3], -tgt_y[3] * src_y[3];


  Eigen::MatrixXd mapToImage = a.inverse() * srcVec;

  map1.create(tgt.size(), CV_32FC1);
  map2.create(tgt.size(), CV_32FC1);

  //newmaptoimage - 3x3 mat with previous values
  Eigen::Matrix3d newMapToImage;
  newMapToImage << mapToImage(0), mapToImage(1), mapToImage(2),
    mapToImage(3), mapToImage(4), mapToImage(5),
    mapToImage(6), mapToImage(7), 1; 


  float new_x;
  float new_y;

  for (int y = 0; y < map1.rows; ++y) {
    for (int x = 0; x < map1.cols; ++x) {
      // make matrix for current coordinates (3x1) [x, y, 1], multiply by maptoimage, then divide by the third value to normalize
      Eigen::MatrixXd current(3, 1);
      current << x, y, 1;
      Eigen::MatrixXd mapped = newMapToImage * current;

      new_x = mapped(0, 0) / mapped(2, 0);
      new_y = mapped(1, 0) / mapped(2, 0);

      // fill maps with points at x and y
      map1.at<float>(y, x) = new_x;
      map2.at<float>(y, x) = new_y;
    }
  }


  return true;
}
}
