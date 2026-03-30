/** Implementation file for finding source image coordinates for a source-to-map
 *  remapping using ground control points
 *
 *  \file ipcv/geometric_transformation/MapGCP.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 15 Sep 2018
 */

#include "MapGCP.h"

#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

using namespace std;

namespace ipcv {

/** Find the source coordinates (map1, map2) for a ground control point
 *  derived mapping polynomial transformation
 *
 *  \param[in] src   source cv::Mat of CV_8UC3
 *  \param[in] map   map (target) cv::Mat of CV_8UC3
 *  \param[in] src_points
 *                   vector of cv::Points representing the ground control
 *                   points from the source image
 *  \param[in] map_points
 *                   vector of cv::Points representing the ground control
 *                   points from the map image
 *  \param[in] order  mapping polynomial order
 *                      EXAMPLES:
 *                        order = 1
 *                          a0*x^0*y^0 + a1*x^1*y^0 +
 *                          a2*x^0*y^1
 *                        order = 2
 *                          a0*x^0*y^0 + a1*x^1*y^0 + a2*x^2*y^0 +
 *                          a3*x^0*y^1 + a4*x^1*y^1 +
 *                          a5*x^0*y^2
 *                        order = 3
 *                          a0*x^0*y^0 + a1*x^1*y^0 + a2*x^2*y^0 + a3*x^3*y^0 +
 *                          a4*x^0*y^1 + a5*x^1*y^1 + a6*x^2*y^1 +
 *                          a7*x^0*y^2 + a8*x^1*y^2 +
 *                          a9*x^0*y^3
 *  \param[out] map1  cv::Mat of CV_32FC1 (size of the destination map)
 *                    containing the horizontal (x) coordinates at which to
 *                    resample the source data
 *  \param[out] map2  cv::Mat of CV_32FC1 (size of the destination map)
 *                    containing the vertical (y) coordinates at which to
 *                    resample the source data
 */
bool MapGCP(const cv::Mat src, const cv::Mat map,
            const vector<cv::Point> src_points,
            const vector<cv::Point> map_points, const int order,
            cv::Mat& map1, cv::Mat& map2) {

  // create maps with map image size
  map1.create(map.size(), CV_32FC1);
  map2.create(map.size(), CV_32FC1);
  map1.setTo(0);
  map2.setTo(0);

  int nsize = src_points.size();

  float src_x;
  float src_y;
  int col;

  // number of terms equation, matrix size is size of src points and number of terms
  int num_terms = (order + 1) * (order + 2) / 2;
  Eigen::MatrixXd a(nsize, num_terms);
  Eigen::VectorXd b_x(nsize);
  Eigen::VectorXd b_y(nsize); 

  // first loop - find src and map points at x and y, set b vectors with src points
  for (int idx = 0; idx < nsize; idx++) {
    const cv::Point& src_pt = src_points.at(idx);
    const cv::Point& map_pt = map_points.at(idx);

    b_x(idx) = src_pt.x;
    b_y(idx) = src_pt.y;

    // iterate through order to set mattrix a
    col = 0;
    for (int deg_x = 0; deg_x <= order; deg_x++) {
      for (int deg_y = 0; deg_y <= order - deg_x; deg_y++) {
        a(idx, col++) = pow(map_pt.x, deg_x) * pow(map_pt.y, deg_y);
      }
    }
  }

  // set coefficients x and y with matrix a and vector b
  Eigen::VectorXd coeffs_x = (a.transpose() * a).inverse() * a.transpose() * b_x;
  Eigen::VectorXd coeffs_y = (a.transpose() * a).inverse() * a.transpose() * b_y;

  // iterate through map pixels to find new x and y for coefficients times x and y to the order degree etc.
  for (int y = 0; y < map.rows; y++) {
    for (int x = 0; x < map.cols; x++) {
      
      src_x = 0;
      src_y = 0;
      col = 0;

      for (int deg_x = 0; deg_x <= order; deg_x++) {
        for (int deg_y = 0; deg_y <= order - deg_x; deg_y++) {
          src_x += coeffs_x(col) * pow(x, deg_x) * pow(y, deg_y);
          src_y += coeffs_y(col) * pow(x, deg_x) * pow(y, deg_y);

          col++;
        }
      }

      // map1 for x values, map2 for y values
      map1.at<float>(y, x) = src_x;
      map2.at<float>(y, x) = src_y;


    }
  }
  return true;
}
}
