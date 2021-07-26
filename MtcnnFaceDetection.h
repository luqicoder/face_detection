#pragma once
#include "FaceDetection.h"
class MtcnnFaceDetection :
	public FaceDetection
{
public:
	MtcnnFaceDetection() = default;
	size_t face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const;
	void face_detection_showImage(const cv::Mat& image) const;
};

