/** Implementation file for finding source image coordinates for a source-to-map
 *  remapping using ground control points
 *
 *  \file ipcv/geometric_transformation/MapPolar.cpp
 *  \author Carly Adams
 *  \date 10 Oct 2024
 */

#include "MapPolar.h"

#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/core.hpp>

using namespace std;

namespace ipcv {

bool MapPolar(const cv::Mat src, const bool use_log, cv::Mat& map1, cv::Mat& map2) {

    // create maps
    map1 = cv::Mat(src.size(), CV_32FC1);
    map2 = cv::Mat(src.size(), CV_32FC1);

    //
    float new_x;
    float new_y;
    float b;
    float rho;
    float theta;

    // log transform 
    if (use_log) {

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                b = pow(10, (log10(src.rows) / src.cols));
                rho = pow(b, x) - 1;
                theta = -2 * M_PI * y / src.rows;

                new_x = rho * cos(theta) + (src.cols / 2);
                new_y = rho * sin(theta) + (src.rows / 2);

                map1.at<float>(y, x) = new_x;
                map2.at<float>(y, x) = new_y;
            }
        }
    }
    

    // polar transform 
     else {

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                b = pow(src.rows / 2, 2) + pow(src.cols / 2, 2);
                rho = (pow(b, 0.5) / src.cols) * x;
                theta = (2 * M_PI / src.cols) * y - M_PI;

                new_x = rho * cos(theta) + (src.cols / 2);
                new_y = rho * sin(theta) + (src.rows / 2);

                map1.at<float>(y, x) = new_x;
                map2.at<float>(y, x) = new_y;
         }
        }
     }


return true;
}

}
