// 
// find the largest magnitude, let it pass --> in order of magnitude size, pass the components using a notch filter
// add them together --> make a mask with lots of notch filters, add sinusoids, etc. 
// 
// display 6 images - original image, fourier transform (log magnitude)
// fourier coefficients used log mag 
// current component + current component scaled --> scale between 0 and 255
// summed components
//
// p - pause (waitkey), q - escape 

// opencv: video capture - build a video from frames for extra credit

#include <iostream>
#include <cmath>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    bool verbose = false;
    string src_filename = "";

    po::options_description options("Options");
    options.add_options()("help,h", "display this message")(
        "verbose,v", po::bool_switch(&verbose), "verbose [default is silent]")(
        "source-filename,i", po::value<string>(&src_filename), "source filename");
    po::positional_options_description positional_options;
    positional_options.add("source-filename", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                .options(options)
                .positional(positional_options)
                .run(),
            vm);
    po::notify(vm);
    cv::Mat src = cv::imread(src_filename, cv::IMREAD_GRAYSCALE);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows) - src.rows;
    int n = cv::getOptimalDFTSize(src.cols) - src.cols;
    cv::copyMakeBorder(src, padded, 0, m, 0, n, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexf(padded.size(), CV_32FC2);
    cv::merge(planes, 2, complexf);
    cv::dft(complexf, complexf);

    cv::Mat magnitude = ipcv::DftMagnitude(complexf, 0);
    cv::Mat shifted = ipcv::DftShift(magnitude);

    cv::Mat logmag = cv::Mat::zeros(complexf.size(), CV_32F);
    cv::Mat logmagNormal;
    cv::log(shifted + cv::Scalar::all(1), logmag);
    cv::normalize(logmag, logmagNormal, 0, 1, cv::NORM_MINMAX); // 1 or 255?

    cv::imshow("Original Image", src);
    cv::imshow("Fourier Transform", logmagNormal);
    cv::waitKey(0);

    cv::Mat mask;
    cv::Mat currentComponent; //= cv::Mat::zeros(complexf.size(), CV_32F);
    cv::Mat currentComponentScaled = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat summedComponents = cv::Mat::zeros(src.size(), CV_32F);
    cv::Point maxLoc;
    cv::Mat coeffs = cv::Mat::zeros(complexf.size(), CV_32F);
    double max_value = 1.0;
    bool paused = false;

    while(max_value > 0) {
        mask = cv::Mat::zeros(complexf.size(), CV_32FC2);
        cv::minMaxLoc(logmag, nullptr, &max_value, nullptr, &maxLoc);
        mask.at<cv::Vec2f>(maxLoc.y, maxLoc.x) = complexf.at<cv::Vec2f>(maxLoc.y, maxLoc.x);
        mask.at<cv::Vec2f>(complexf.rows - maxLoc.y, complexf.cols - maxLoc.x) = complexf.at<cv::Vec2f>(maxLoc.y, maxLoc.x);

        cv::idft(mask, currentComponent, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        cv::normalize(currentComponent, currentComponentScaled, 0, 255, cv::NORM_MINMAX);
    
        cv::imshow("Current Component Scaled", currentComponentScaled);
      
        summedComponents += currentComponent;
        cv::imshow("Summed Components", summedComponents);
        cv::normalize(currentComponent, currentComponent, 0, 1, cv::NORM_MINMAX);
        cv::imshow("Current Component", currentComponent);

        cv::log(logmag + cv::Scalar::all(1), coeffs);
        cv::imshow("Fourier Coefficients", coeffs);
        logmag.at<float>(maxLoc) = 0;

        char key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
        else if (key == 'p' || key == 'P') {
            paused = !paused;
        }

        if (paused) {
            cv::waitKey(0);
        }
        while (paused) {
            char key_pause = cv::waitKey(1);
            if (key_pause == 'p' || key_pause == 'P') {
                paused = false;
            }
        }

    }

}
