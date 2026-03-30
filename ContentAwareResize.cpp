#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;

namespace ipcv {


// function to find original energy using sobel gradient
cv::Mat findEnergy(const cv::Mat& src) {
    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Mat energy;
    cv::Mat absgradX;
    cv::Mat absgradY;
    
    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> gradch(3);
    cv::split(src, channels);

    for (int ch = 0; ch < src.channels(); ch++) {
        cv::Sobel(channels[ch], gradientX, CV_16S, 0, 1);
        cv::Sobel(channels[ch], gradientY, CV_16S, 1, 0);
        cv::convertScaleAbs(gradientX, absgradX);
        cv::convertScaleAbs(gradientY, absgradY);
        cv::addWeighted(absgradX, 0.5, absgradY, 0.5, 0, gradch[ch]);
    }
    cv::merge(gradch, energy);
    cv::cvtColor(energy, energy, cv::COLOR_BGR2GRAY);
    energy.convertTo(energy, CV_64F, 1.0/255.0);
    cv::normalize(energy, energy, 0, 255, cv::NORM_MINMAX);
    energy.convertTo(energy, CV_8U);
    return energy;
}

// function to find cumulative energy
cv::Mat findCumulativeEnergy(const cv::Mat& src, const cv::Mat& energy) {
    double minimum_energy;
    cv::Mat cumulative_energy = cv::Mat::zeros(src.rows, src.cols, CV_64F);
        for (int y = 1; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                minimum_energy = cumulative_energy.at<double>(y - 1, x);
                if (x > 0) {
                    minimum_energy = std::min(minimum_energy, cumulative_energy.at<double>(y - 1, x - 1));
                }
                if (x < cumulative_energy.cols - 1) {
                    minimum_energy = std::min(minimum_energy, cumulative_energy.at<double>(y - 1, x + 1));
                }
                cumulative_energy.at<double>(y, x) += energy.at<uchar>(y, x) + minimum_energy;
            }
        }
    return cumulative_energy;
}


bool ContentAwareResize(cv::Mat& src, cv::Mat& dst, const int dst_height, const int dst_width) {
    dst.create(dst_height, dst_width, src.type());
    cv::Mat gradientX;    
    cv::Mat gradientY;
    cv::Mat energy;
    cv::Mat absgradX;
    cv::Mat absgradY;
    
    energy = findEnergy(src);
    cv::Mat newsrc = src.clone();
    cv::Mat newenergy = energy.clone();

    // resize width larger - repeats 4 times for height/width larger/smaller, with slight changes
    while (newsrc.cols < dst_width) {
        newenergy = findEnergy(newsrc);
        cv::Mat cumulative_energy = cv::Mat::zeros(newsrc.rows, newsrc.cols, CV_64F);
        cumulative_energy = findCumulativeEnergy(newsrc, newenergy);

        // finding the optimal seam
        std::vector<int> vertical_seam(newsrc.rows);
        cv::Point min_location;
        double min_value;
        cv::minMaxLoc(cumulative_energy.row(newsrc.rows - 1), &min_value, nullptr, &min_location, nullptr);
        vertical_seam[newsrc.rows - 1] = min_location.x;

        for (int y = newsrc.rows - 2; y >= 0; y--) {
            int x = vertical_seam[y + 1];
            int left = std::max(0, x - 1);
            int right = std::min(newsrc.cols - 1, x + 1);

            cv::Mat roi = cumulative_energy.row(y).colRange(left, right + 1);
            cv::minMaxLoc(roi, &min_value, nullptr, &min_location, nullptr);

            vertical_seam[y] = left + min_location.x;
        }

        cv::Mat carved_image = cv::Mat::zeros(newsrc.rows, newsrc.cols + 1, newsrc.type());
        cv::Mat carved_energy = cv::Mat::zeros(newsrc.rows, newenergy.cols + 1, newenergy.type());


        // assign values adding seam
        for (int y = 0; y < carved_image.rows; y++) {
            int seam = vertical_seam[y];

            for (int x = 0; x < seam; x++) {
                carved_image.at<cv::Vec3b>(y, x) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y, x) = newenergy.at<uchar>(y, x);
            }

            carved_image.at<cv::Vec3b>(y, seam) = newsrc.at<cv::Vec3b>(y, seam);
            carved_energy.at<uchar>(y, seam) = newenergy.at<uchar>(y, seam);

            for (int x = seam; x < carved_image.cols - 1; x++) {
                carved_image.at<cv::Vec3b>(y, x + 1) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y, x + 1) = newenergy.at<uchar>(y, x);
            }
        }

        newsrc = carved_image.clone();
        newenergy = carved_energy.clone();
    }

    // resize height larger
    while (newsrc.rows < dst_height) {
        newenergy = findEnergy(newsrc);
        cv::Mat cumulative_energy = cv::Mat::zeros(newsrc.rows, newsrc.cols, CV_64F);
        cumulative_energy = findCumulativeEnergy(newsrc, newenergy);

        std::vector<int> horizontal_seam(newsrc.cols);
        cv::Point min_location;
        double min_value;
        cv::minMaxLoc(cumulative_energy.col(newsrc.cols - 1), &min_value, nullptr, &min_location, nullptr);
        horizontal_seam[newsrc.cols - 1] = min_location.y;

        for (int x = newsrc.cols - 2; x >= 0; x--) {
            int y = horizontal_seam[x + 1];
            int top = std::max(0, y - 1);
            int bottom = std::min(newsrc.rows - 1, y + 1);

            cv::Mat roi = cumulative_energy.col(x).rowRange(top, bottom + 1);
            cv::minMaxLoc(roi, &min_value, nullptr, &min_location, nullptr);

            horizontal_seam[x] = top + min_location.y;
        }

        cv::Mat carved_image = cv::Mat::zeros(newsrc.rows + 1, newsrc.cols, newsrc.type());
        cv::Mat carved_energy = cv::Mat::zeros(newsrc.rows + 1, newenergy.cols, newenergy.type());

        for (int x = 0; x < carved_image.cols; x++) {
            int seam = horizontal_seam[x];

            for (int y = 0; y < seam; y++) {
                carved_image.at<cv::Vec3b>(y, x) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y, x) = newenergy.at<uchar>(y, x);
            }

            carved_image.at<cv::Vec3b>(seam, x) = newsrc.at<cv::Vec3b>(seam, x);
            carved_energy.at<uchar>(seam, x) = newenergy.at<uchar>(seam, x);

            for (int y = seam; y < carved_image.rows - 1; y++) {
                carved_image.at<cv::Vec3b>(y + 1, x) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y + 1, x) = newenergy.at<uchar>(y, x);
            }
        }


        newsrc = carved_image.clone();
        newenergy = carved_energy.clone();


    }

    // resize width smaller
    while (newsrc.cols > dst_width) {
        newenergy = findEnergy(newsrc);
        cv::Mat cumulative_energy = cv::Mat::zeros(newsrc.rows, newsrc.cols, CV_64F);
       cumulative_energy = findCumulativeEnergy(newsrc, newenergy);


        std::vector<int> vertical_seam(newsrc.rows);
        cv::Point min_location;
        double min_value;
        cv::minMaxLoc(cumulative_energy.row(newsrc.rows - 1), &min_value, nullptr, &min_location, nullptr);
        vertical_seam[newsrc.rows - 1] = min_location.x;

        for (int y = newsrc.rows - 2; y>= 0; y--) {
            int x = vertical_seam[y + 1];
            int left = std::max(0, x - 1);
            int right = std::min(newsrc.cols - 1, x + 1);

            cv::Mat roi = cumulative_energy.row(y).colRange(left, right + 1);
            cv::minMaxLoc(roi, &min_value, nullptr, &min_location, nullptr);

            vertical_seam[y] = left + min_location.x;

        }

        cv::Mat carved_image = cv::Mat::zeros(newsrc.rows, newsrc.cols - 1, newsrc.type());
        cv::Mat carved_energy = cv::Mat::zeros(newsrc.rows, newenergy.cols - 1, newenergy.type());

        for (int y = 0; y < carved_image.rows; y++) {
            int seam = vertical_seam[y];

            for (int x = 0; x < seam; x++) {
                carved_image.at<cv::Vec3b>(y, x) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y, x) = energy.at<uchar>(y, x);
            }
            for (int idx = seam; idx < carved_image.cols; idx++) {
                carved_image.at<cv::Vec3b>(y, idx) = newsrc.at<cv::Vec3b>(y, idx + 1);
                carved_energy.at<uchar>(y, idx) = energy.at<uchar>(y, idx + 1);
            }

        }

        newsrc = carved_image.clone();
        newenergy = carved_energy.clone();

    }

    // resize height smaller
    while (newsrc.rows > dst_height) {     
        newenergy = findEnergy(newsrc);
        cv::Mat cumulative_energy = cv::Mat::zeros(newsrc.rows, newsrc.cols, CV_64F);
        cumulative_energy = findCumulativeEnergy(newsrc, newenergy);

        std::vector<int> horizontal_seam(newsrc.cols);
        cv::Point min_location;
        double min_value;
        cv::minMaxLoc(cumulative_energy.col(newsrc.cols - 1), &min_value, nullptr, &min_location, nullptr);
        horizontal_seam[newsrc.cols - 1] = min_location.y;

        for (int x = newsrc.cols - 2; x >= 0; x--) {
            int y = horizontal_seam[x + 1];
            int top = std::max(0, y - 1);
            int bottom = std::min(newsrc.rows - 1, y + 1);

            cv::Mat roi = cumulative_energy.col(x).rowRange(top, bottom + 1);
            cv::minMaxLoc(roi, &min_value, nullptr, &min_location, nullptr);

            horizontal_seam[x] = top + min_location.y;
        }

        cv::Mat carved_image = cv::Mat::zeros(newsrc.rows - 1, newsrc.cols, newsrc.type());
        cv::Mat carved_energy = cv::Mat::zeros(newenergy.rows - 1, newenergy.cols, newenergy.type());

        for (int x = 0; x < carved_image.cols; x++) {
            int seam = horizontal_seam[x];

            for (int y = 0; y < seam; y++) {
                carved_image.at<cv::Vec3b>(y, x) = newsrc.at<cv::Vec3b>(y, x);
                carved_energy.at<uchar>(y, x) = newenergy.at<uchar>(y, x);
            }
            for (int idx = seam; idx < carved_image.rows; idx++) {
                carved_image.at<cv::Vec3b>(idx, x) = newsrc.at<cv::Vec3b>(idx + 1, x);
                carved_energy.at<uchar>(idx, x) = newenergy.at<uchar>(idx + 1, x);
            }
        }

        newsrc = carved_image.clone();
        newenergy = carved_energy.clone();
    }

    // set destination to the updated source image
    dst = newsrc;
    return true;

}
}
