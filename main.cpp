#include "FaceDetection.h"

int main()
{
	cv::Mat image = cv::imread("mult_face.jpg");
	FaceDetection::Init(1);
	FaceDetection face;
	std::vector<std::vector<cv::Point2f>> landmarks;
	std::vector<cv::Rect> rects;
	face.face_detection_showImage(image);
	//face.
	//cv::Mat affImg;
	//face.affineTransformImage(image, affImg, landmarks.at(0));
	//std::cout << i << std::endl;
	return 0;
}