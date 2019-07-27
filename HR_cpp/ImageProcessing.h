#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

typedef std::vector<std::vector<double> > Kernel;

class ImageProcessing
{
public:
	ImageProcessing		(int cores = 0);
	~ImageProcessing	();

	cv::Mat GaussianBlur(cv::Mat image);
private:
	int		_cores;
	int		_i_working_thread;
	Kernel		_kernel;
	cv::Mat 	_blurImage, _image;


	void	applyFilter		(cv::Mat image_src, cv::Mat image_dst, Kernel filter, int row_start, int row_end);
	void	cvtToGray		(cv::Mat image_src);
	void	threadFunction		(int row_start, int row_end);
	Kernel	getGaussian		(int kernel_radius);
};

