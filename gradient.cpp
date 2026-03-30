#include <ctime>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imgs/ipcv/gradient/Sobel.h"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  bool verbose = false;
  string src_filename = "";
  string dst_filename = "";


  po::options_description options("Options");
  options.add_options()("help,h", "display this message")(
      "verbose,v", po::bool_switch(&verbose), "verbose [default is silent]")(
      "source-filename,i", po::value<string>(&src_filename), "source filename")(
      "destination-filename,o", po::value<string>(&dst_filename),
      "destination filename");

  po::positional_options_description positional_options;
  positional_options.add("source-filename", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(options)
                .positional(positional_options)
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << "Usage: " << argv[0] << " [options] source-filename" << endl;
    cout << options << endl;
    return EXIT_SUCCESS;
  }

  if (!boost::filesystem::exists(src_filename)) {
    cerr << "Provided source file does not exists" << endl;
    return EXIT_FAILURE;
  }

  cv::Mat src = cv::imread(src_filename, cv::IMREAD_GRAYSCALE);

  int delta;

  ipcv::BorderMode border_type;
  border_type = ipcv::BorderMode::REPLICATE;
//  cv::BorderTypes border_type;
//  border_type = cv::BORDER_REPLICATE;

  if (verbose) {
    cout << "Source filename: " << src_filename << endl;
    cout << "Size: " << src.size() << endl;
    cout << "Channels: " << src.channels() << endl;
    cout << "Destination filename: " << dst_filename << endl;
  }

  cv::Mat dst;
  cv::Mat gy, gx;

  clock_t startTime = clock();

  ipcv::Sobel(src, dst, delta, border_type);

  cv::Mat magnitude(src.size(), CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float Gx = dst.at<float>(y, x);
            float Gy = dst.at<float>(y, x); 
            magnitude.at<float>(y, x) = sqrt(Gx * Gx + Gy * Gy); 
        }
    }

  clock_t endTime = clock();


  if (verbose) {
    cout << "Elapsed time: "
         << (endTime - startTime) / static_cast<double>(CLOCKS_PER_SEC)
         << " [s]" << endl;
  }

    std::cout << "numer of channels: " << src.channels() << std::endl;


  if (dst_filename.empty()) {
    cv::imshow(src_filename, src);
    cv::imshow(src_filename + " [Filtered]", magnitude);
    cv::waitKey(0);
  } else {
    cv::imwrite(dst_filename, magnitude);
  }

  return EXIT_SUCCESS;
}
