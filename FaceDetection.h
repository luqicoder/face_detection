//-----------------------------------------------------------------------
// Copyright (c) 2020 by luqi
// file  : FaceDetection.h
// since : 2020/8/13
// description  :  face detection
// TODO: 
//-----------------------------------------------------------------------
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class FaceDetection
{
public:
	FaceDetection() = default;
	static bool Init(int mode, bool useGPU = false, int devId = 1);
	virtual size_t face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const;
	virtual void face_detection_showImage(const cv::Mat& image) const;

	void affineTransform(const cv::Mat& image, cv::Mat& affineImg, const std::vector<cv::Point2f> landmark_);
	void affineTransformImage(const cv::Mat& image, cv::Mat& affineImg, std::vector<cv::Point2f> landmark_);

	FaceDetection(const FaceDetection&) = delete;
	FaceDetection& operator=(const FaceDetection&) = delete;
	virtual ~FaceDetection() = default;

private:
	bool inital(int mode = 0, int gpu = 0);
	float* SVD(float a, float b, float c, float d);
	float** transformationFromPoints(float landmark[5][2], float coord5Points[5][2]);
	cv::Mat warpImage(const cv::Mat& img, float landmark[5][2], float coord5Points[5][2]);
	static FaceDetection* face_detect;
};

