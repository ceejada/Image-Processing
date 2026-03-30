#include <iostream>
#include <cmath>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imgs/ipcv/aware_resize/ContentAwareResize.h"


using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  string src_filename = "";
  string dst_filename = "";
  int height_offset = 0;
  int width_offset = 0;
  int dst_height;
  int dst_width;


  po::options_description options("Options");
  options.add_options()("help,h", "display this message")(
      "source-filename,i", po::value<string>(&src_filename), "source filename")(
      "destination-filename,o", po::value<string>(&dst_filename),
      "destination filename")("width-offset,w", po::value<int>(&width_offset), "width offset")("height-offset,t", po::value<int>(&height_offset), "height offset");

  po::positional_options_description positional_options;
  positional_options.add("source-filename", -1);
/*
  po::options_description options("Options");
  options.add_options()("help,h", "display this message")(
      "source-filename,i", po::value<string>(&src_filename), "source filename")(
      "destination-filename,o", po::value<string>(&dst_filename),
      "destination filename")("dst-width,w", po::value<int>(&dst_width), "dst width")("dst-height,t", po::value<int>(&dst_height), "dst height");

  po::positional_options_description positional_options;
  positional_options.add("source-filename", -1);

*/
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

  cv::Mat src = cv::imread(src_filename, cv::IMREAD_COLOR);

  dst_height = src.rows + height_offset;
  dst_width = src.cols + width_offset;
  cv::Mat dst;

  ipcv::ContentAwareResize(src, dst, dst_height, dst_width);

  cv::imshow("original", src);
  cv::imshow("new image", dst);
  cv::waitKey(0);
    
}