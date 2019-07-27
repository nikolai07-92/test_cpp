#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
#include "ImageProcessing.h"

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char** argv)
{
	auto t1 = Clock::now(); //program start time
	int threads = 0; // number of threads
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR); // Image loading
	if (img.empty()) {
		std::cout << "Image not found: " << argv[1] << std::endl;
		return 0;
	}
	//check second param
	if (argc > 2) {
        threads = atoi(argv[2]);
    }

	std::cout << "Work with " << argv[1];
	ImageProcessing img_proc_obj(threads);
	cv::imwrite("blur.jpg", img_proc_obj.GaussianBlur(img)); // save result image
	std::cout << "Out image: blur.jpg " << std::endl;

	auto t2 = Clock::now(); // program end time
	std::cout << "runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() //program execution time
        << " milliseconds" << std::endl;

	return 0;
}
