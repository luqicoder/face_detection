#include "LibFaceDetection.h"

/****************************************
	function: face_detection
	author  : luqi
	date    : 2020/08/13 10:53
	purpose : 
	input   :
		parm1  : 
		parm2  : 
	output  :
		parm1  : 
	return  :
*****************************************/
size_t LibFaceDetection::face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const
{
	//reset data
	landmarks->clear();
	rects->clear();

	cv::Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	int* pResults = NULL;
	unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
	//是否进行人脸特征点检测 0: 不进行人脸特征点检测1: 进行人脸特征点检测
	int doLandmark = 1;
	//clock_t startTime = clock();
	pResults = facedetect_multiview_reinforce(pBuffer,
		(unsigned char*)(gray.ptr(0)),
		gray.cols,
		gray.rows,
		(int)gray.step,
		1.2f, 2, 48, 0,
		doLandmark);
	//clock_t endTime = clock();
	//printf("%d faces detected.\t\n", (pResults ? *pResults : 0));
	cv::Mat result_multiview = image.clone();;
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		//points向量用于存储5个特征点位置
		std::vector<cv::Point2f> points;
		short* p = ((short*)(pResults + 1)) + 142 * i;
		//x,y为检测框左上角位置
		int x = p[0];
		int y = p[1];
		//w,h为宽高度
		int w = p[2];
		int h = p[3];
		//置信度（越大人脸的可能性越大）
		int neighbors = p[4];
		//眼睛关注角度（左正右负，正脸角度为0）
		int angle = p[5];
		cv::Rect rect = cv::Rect(x, y, w, h);
		rects->push_back(rect);


		//左眼外角，内角-36,39
		//右眼内角，外角-42,45
		//5个特征点，左眼，右眼，鼻尖，左嘴角，右嘴角
		//						  30,    48,     54
		if (doLandmark)
		{
			int left_x = ((int)p[6 + 2 * 36] + (int)p[6 + 2 * 39]) / 2;
			int left_y = ((int)p[6 + 2 * 36 + 1] + (int)p[6 + 2 * 39 + 1]) / 2;
			int right_x = ((int)p[6 + 2 * 42] + (int)p[6 + 2 * 45]) / 2;
			int right_y = ((int)p[6 + 2 * 42 + 1] + (int)p[6 + 2 * 45 + 1]) / 2;
			points.push_back(cv::Point2f(left_x, left_y));
			points.push_back(cv::Point2f(right_x, right_y));
			//circle(result_multiview, cv::Point(left_x, left_y), 1, cv::Scalar(0, 255, 0));
			//circle(result_multiview, cv::Point(right_x, right_y), 1, cv::Scalar(0, 255, 0));
			int j[5] = { 30,48,54 };
			for (int i = 0; i < 3; i++)
			{
				int x = (int)p[6 + 2 * j[i]];
				int y = (int)p[6 + 2 * j[i] + 1];
				circle(result_multiview, cv::Point(x, y), 1, cv::Scalar(0, 255, 0));
				points.push_back(cv::Point2f(x, y));
			}
			landmarks->push_back(points);
		}
	}

	free(pResults);
	return landmarks->size();
}

void LibFaceDetection::face_detection_showImage(const cv::Mat& image) const
{
	cv::Mat copy_img = image.clone();
	std::vector<std::vector<cv::Point2f>> landmarks;
	std::vector<cv::Rect> rects;
	int n = face_detection(image, &landmarks, &rects);
	for (int i = 0; i < n; ++i)
	{
		rectangle(copy_img, rects.at(i), cv::Scalar(0, 255, 0), 2);
		std::vector<cv::Point2f> landmark = landmarks.at(i);
		for (int j = 0; j < 2; ++j)
		{
			int x = (int)landmark.at(j).x;
			int y = (int)landmark.at(j).y;
			circle(copy_img, cv::Point(x, y), 1, cv::Scalar(0, 255, 0));
		}
		for (int j = 2; j < 5; ++j)
		{
			int x = (int)landmark.at(j).x;
			int y = (int)landmark.at(j).y;
			circle(copy_img, cv::Point(x, y), 1, cv::Scalar(0, 255, 0));
		}
	}
	cv::imshow("face_detection_showImage", copy_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}





