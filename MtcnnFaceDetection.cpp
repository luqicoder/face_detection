#include "MtcnnFaceDetection.h"
#include "network.h"
#include "mtcnn.h"

size_t MtcnnFaceDetection::face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const
{
	//reset data
	landmarks->clear();
	rects->clear();

	mtcnn find(image.rows, image.cols);
	find.findFace(image, *landmarks, *rects);
	std::size_t sum = landmarks->size();
	return sum;
}

void MtcnnFaceDetection::face_detection_showImage(const cv::Mat& image) const
{
	std::vector<std::vector<cv::Point2f>> landmarks;
	std::vector<cv::Rect> rects;
	std::size_t sum = face_detection(image, &landmarks, &rects);
	for (std::size_t i = 0; i < sum; ++i)
	{
		rectangle(image, rects.at(i), Scalar(0, 0, 255), 2, 8, 0);
		for(std::size_t j = 0; j < 5; ++j)
			circle(image, landmarks.at(i).at(j), 1, cv::Scalar(0, 255, 0));
	}

	cv::imshow("detection_face", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
