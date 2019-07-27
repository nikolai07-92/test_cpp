#include <opencv2/highgui/highgui.hpp>
#include "ImageProcessing.h"
#include <iostream>
#include <thread>
#include <unistd.h>

#define _USE_MATH_DEFINES
#include <math.h>

/*
Constructor
\param[in] cores Number of hreads
*/
ImageProcessing::ImageProcessing(int cores)
{
	if (cores) {
        _cores = cores;
    }
	else {
		_cores = sysconf(_SC_NPROCESSORS_ONLN); // get number of cores
	}
	_i_working_thread = _cores;
	std::cout << " (" << _cores << " threads)" << std::endl;
}

ImageProcessing::~ImageProcessing()
{
}
/*
Convert an image from color t grayscale
\param[in,out] image_src Target image
*/
void ImageProcessing::cvtToGray(cv::Mat image_src)
{
	int channels    = image_src.channels();
	int width	    = image_src.cols;
	int height	    = image_src.rows;
	uchar b, g, r, avg;
	uchar* f_ptr	= image_src.data;

	for (int colIdx = 0; colIdx < width; colIdx++) {
		for (int rowIdx = 0; rowIdx < height; rowIdx++)	{
			b = f_ptr[channels * (rowIdx*width + colIdx)];
			g = f_ptr[channels * (rowIdx*width + colIdx) + 1];
			r = f_ptr[channels * (rowIdx*width + colIdx) + 2];
			avg = (b + g + r) / channels;
			for (size_t ch = 0; ch < channels; ch++) {
                f_ptr[channels * (rowIdx * width + colIdx) + ch] = avg;
            }
		}
	}
}
/*
Gauss filter kernel calculation
\param[in] kernel_radius
\return Kernel matrix
*/
Kernel ImageProcessing::getGaussian(int kernel_radius)
{
	Kernel kernel(2 * kernel_radius + 1, std::vector<double>(2 * kernel_radius + 1)); // kernel matrix

	double sigma 	= kernel_radius / 2.;
	double mean 	= kernel.size() / 2;
	double sum 	    = 0;

	//Calculate value
	for (int row = 0; row < kernel.size(); row++) {
		for (int col = 0; col < kernel[row].size(); col++) {
			kernel[row][col] = exp(-0.5 * (pow((col - mean) / sigma, 2.0) + pow((row - mean) / sigma, 2.0)))
				/ (2 * M_PI * sigma * sigma);
			sum += kernel[row][col];
		}
    }
	// Normalization
	for (int row = 0; row < kernel.size(); row++) {
		for (int col = 0; col < kernel[row].size(); col++) {
			kernel[row][col] /= sum;
        }
    }
	return kernel;
}
/*
Function of preparing the image and starting the function of blur
\param[in] image Target image
\return Blur image
*/
cv::Mat ImageProcessing::GaussianBlur(cv::Mat image)
{
	std::vector<int>		    row_borders	(_cores);
	std::vector<std::thread>	v_threads	(_cores);

	_image		= image;
	_blurImage	= cv::Mat::zeros    (image.rows, image.cols, CV_8UC1);
	_kernel		= getGaussian       (3);

	cvtToGray(image);

	int row_start	= 0;
	int	row_end		= 0;
	//Distribution of parts of image on the threads and start of the gauss blur function
	for (unsigned int i = 0; i < _cores; i++) {
		row_start = row_end;
		row_end += floor(image.rows / _cores);
		if ((image.rows % _cores) > i) {
            row_end++;
        }
		v_threads[i] = std::thread(&ImageProcessing::threadFunction, this, row_start, row_end);
	}
	// Waitin for execution of threads
	while (_i_working_thread > 0) {
		sleep(0.05);
	}
	for (unsigned int i = 0; i < _cores; i++) {
		v_threads[i].detach();
		v_threads[i].~thread();
	}

	return _blurImage;
}
/*
Thread function
\param[in] row_start Start line of this thread
\param[in] row_end End line of this thread
*/
void ImageProcessing::threadFunction(int row_start, int row_end)
{
	applyFilter(_image, _blurImage, _kernel, row_start, row_end);
	_i_working_thread--;
}
/*
Image blur function
\param[in] image_src Target image
\param[out] image_dst Blur image
\param[in] filter Gauss kernel
\param[in] row_start Start line of this thread
\param[in] row_end End line of this thread
*/
void ImageProcessing::applyFilter(cv::Mat image_src, cv::Mat image_dst, Kernel filter, int row_start, int row_end)
{
	int i, j, h, w, src_h, src_w;
	int height		    = image_src.rows;
	int width		    = image_src.cols;
	int filterHeight	= filter.size();
	int filterWidth		= filter[0].size();
	int radius		    = (filterHeight - 1) / 2;

	uchar* dst_ptr = image_dst.data; // ptr to image data
	uchar* src_ptr = image_src.data; // ptr to image data

	for (i = row_start; i < row_end; i++) {
		for (j = 0; j < width; j++) {
			for (h = i - radius; h < filterHeight + i - radius; h++) {
				for (w = j - radius; w < filterWidth + j - radius; w++) {
					src_h = h;
					src_w = w;
					// for border processing
					if (src_h >= height) {
						src_h -= (src_h - height) * 2 + 1;
					}
					// for border processing
					if (src_w >= width) {
                        src_w -= (src_w - width) * 2 + 1;
                    }
					double filter_val = filter[h - i + radius][w - j + radius];
					double src_val = src_ptr[image_src.channels() * (abs(src_h) * width + abs(src_w))];
					dst_ptr[i * width + j] += filter_val * src_val;
				}
			}
		}
	}
}
