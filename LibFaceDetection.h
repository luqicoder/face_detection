#pragma once
#include "FaceDetection.h"
#include"facedetect-dll.h"
#include<iostream>
#include<opencv2\opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include<ctime>
#include <list>
#include <math.h>
#include <vector>
#include <string>
#include <Eigen/Dense>

#define DETECT_BUFFER_SIZE 0x20000


class LibFaceDetection : public FaceDetection
{
public:
	LibFaceDetection() = default;
	size_t face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const override;
	void face_detection_showImage(const cv::Mat& image) const override;
};

