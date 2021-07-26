#include "FaceDetection.h"
#include "LibFaceDetection.h"
#include "MtcnnFaceDetection.h"

FaceDetection* FaceDetection::face_detect = nullptr;

bool FaceDetection::Init(int mode, bool useGPU, int devId)
{
	FaceDetection* copy_face = face_detect;
	if (mode == 0)
	{
		face_detect = new LibFaceDetection();
	}
	else if (mode == 1)
	{
		face_detect = new MtcnnFaceDetection();
	}
	if (copy_face != nullptr)
		delete copy_face;
	return true;
}
// author  : luqi
// date    : 2020/08/13 10:53
// purpose : 
/*******************************************
   parm1[mode]: 一张包含人脸的图像矩阵
   parm2[outpt]: 检测到的每个人脸区域位置
   return      : 检测到人脸个数
********************************************/
/*-------------------------------------------
   mode[input]: 人脸检测模式
   gpu [input]: 选择是否选用gpu
-------------------------------------------*/
bool FaceDetection::inital(int mode, int gpu)
{
	return false;
}

size_t FaceDetection::face_detection(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>* landmarks, std::vector<cv::Rect>* rects) const
{
	return face_detect->face_detection(image, landmarks, rects);
}

void FaceDetection::face_detection_showImage(const cv::Mat& image) const
{
	face_detect->face_detection_showImage(image);
}

void FaceDetection::affineTransform(const cv::Mat& image, cv::Mat& affineImg, const std::vector<cv::Point2f> landmark_)
{
	float landmark[5][2] = { { 576, 467 },
	{ 728, 460 },
	{ 655.184, 557.436 },
	{ 582.223, 625.247 },
	{ 733.019, 629.918 } };
	cv::Mat paintImg = image.clone();
	for (int i = 0; i < 5; i++)
	{
		landmark[i][0] = landmark_.at(i).x;
		landmark[i][1] = landmark_.at(i).y;
	}
	//96*112
	/*float coord5Points[5][2] = { {30.2946, 51.6963},
		{65.5318, 51.6963},
		{48.0252, 71.7366},
		{33.5493, 92.3655},
		{62.7299, 92.3655} };*/
	float coord5Points[5][2] = { {38.2946, 51.6963},
		{73.5318, 51.6963},
		{56.0252, 71.7366},
		{41.5493, 92.3655},
		{70.7299, 92.3655} };

	cv::Mat resTemp = warpImage(image, landmark, coord5Points);
	cv::Range r1, r2;
	r1.start = 0; r1.end = cv::min(112, resTemp.rows);
	r2.start = 0; r2.end = cv::min(112, resTemp.cols);
	//std::cout << resTemp.rows << "  " << resTemp.cols << std::endl;
	affineImg = cv::Mat::Mat(resTemp, r1, r2);
	if (resTemp.rows < 112 || resTemp.cols < 112)
		affineImg.resize(112, 112);
}

void FaceDetection::affineTransformImage(const cv::Mat& image, cv::Mat& affineImg, std::vector<cv::Point2f> landmark_)
{
	//cv::imshow("src", image);
	float landmark[5][2] = { { 576, 467 },
	{ 728, 460 },
	{ 655.184, 557.436 },
	{ 582.223, 625.247 },
	{ 733.019, 629.918 } };
	cv::Mat paintImg = image.clone();
	for (int i = 0; i < 5; i++)
	{
		landmark[i][0] = landmark_.at(i).x;
		landmark[i][1] = landmark_.at(i).y;
	}

	/*float coord5Points[5][2] = { {30.2946, 51.6963},
		{65.5318, 51.6963},
		{48.0252, 71.7366},
		{33.5493, 92.3655},
		{62.7299, 92.3655} };*/
	float coord5Points[5][2] = { {38.2946, 51.6963},
		{73.5318, 51.6963},
		{56.0252, 71.7366},
		{41.5493, 92.3655},
		{70.7299, 92.3655} };
	/*float coord5Points[5][2] = { {34.2946, 51.6963},
		{69.5318, 51.6963},
		{52.0252, 71.7366},
		{37.5493, 92.3655},
		{66.7299, 92.3655} };*/
	cv::Mat resTemp = warpImage(image, landmark, coord5Points);
	cv::Range r1, r2;
	r1.start = 0; r1.end = 112;
	r2.start = 0; r2.end = 112;
	affineImg = cv::Mat::Mat(resTemp, r1, r2);
	cv::imshow("face", affineImg);
	cv::waitKey(0);
}


//矩阵[[a, b], [c, d]]
float* FaceDetection::SVD(float a, float b, float c, float d)
{

	float* p = (float*)malloc(10 * sizeof(float));

	std::vector<std::vector<float>> vec{ { a, b },
	{ c, d } };
	const int rows{ 2 }, cols{ 2 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
	Eigen::MatrixXf singular_values = svd.singularValues();
	Eigen::MatrixXf left_singular_vectors = svd.matrixU();
	Eigen::MatrixXf right_singular_vectors = svd.matrixV();

	for (int i = 0; i < 4; i++) {
		p[i] = left_singular_vectors.data()[i];
	}

	for (int i = 4, j = 0; i < 6; i++, j++) {
		p[i] = singular_values.data()[j];
	}

	for (int i = 6, j = 0; i < 10; i++, j++) {
		p[i] = right_singular_vectors.data()[j];
	}
	return p;

}


float** FaceDetection::transformationFromPoints(float landmark[5][2], float coord5Points[5][2]) {

	//求均值 c1, c2
	float c1[1][2], c2[1][2];
	float tempNum0 = 0.0, tempNum1 = 0.0;
	for (int i = 0; i < 5; i++) {
		tempNum0 += landmark[i][0];
		tempNum1 += landmark[i][1];
	}
	c1[0][0] = tempNum0 / 5.0;
	c1[0][1] = tempNum1 / 5.0;
	tempNum0 = 0.0; tempNum1 = 0.0;
	for (int i = 0; i < 5; i++) {
		tempNum0 += coord5Points[i][0];
		tempNum1 += coord5Points[i][1];
	}
	c2[0][0] = tempNum0 / 5.0;
	c2[0][1] = tempNum1 / 5.0;

	for (int i = 0; i < 5; i++) {
		landmark[i][0] -= c1[0][0];
		coord5Points[i][0] -= c2[0][0];

		landmark[i][1] -= c1[0][1];
		coord5Points[i][1] -= c2[0][1];
	}

	//求标准差 s1, s2
	float s1, s2;
	tempNum0 = 0; tempNum1 = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 2; j++) {
			tempNum0 += landmark[i][j];
			tempNum1 += coord5Points[i][j];
		}
	}
	float meanLandmark = tempNum0 / 10.0;
	float meanCoord5Points = tempNum1 / 10.0;
	tempNum0 = 0.0; tempNum1 = 0.0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 2; j++) {
			tempNum0 += (float)pow(landmark[i][j] - meanLandmark, 2);
			tempNum1 += (float)pow(coord5Points[i][j] - meanCoord5Points, 2);
		}
	}
	s1 = sqrt(tempNum0 / 10.0);
	s2 = sqrt(tempNum1 / 10.0);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 2; j++) {
			landmark[i][j] /= s1;
			coord5Points[i][j] /= s2;
		}
	}

	//奇异值分解
	//1)矩阵运算a.T * b
	float a = 0.0, b = 0.0, c = 0.0, d = 0.0;
	for (int i = 0; i < 5; i++) {
		a += landmark[i][0] * coord5Points[i][0];
		b += landmark[i][0] * coord5Points[i][1];
		c += landmark[i][1] * coord5Points[i][0];
		d += landmark[i][1] * coord5Points[i][1];
	}

	float* p = SVD(a, b, c, d);
	float U[2][2], S[2][1], Vt[2][2];
	U[0][0] = p[0]; U[0][1] = p[1];
	U[1][0] = p[2]; U[1][1] = p[3];

	S[0][0] = p[4]; S[1][0] = p[5];

	Vt[0][0] = p[6]; Vt[0][1] = p[7];
	Vt[1][0] = p[8]; Vt[1][1] = p[9];

	float R[2][2] = { {0.0,0.0}, {0.0, 0.0 } };

	//R = (U * Vt).T
	for (int i = 0; i < 2; i++) {
		/*cout << i << " " << U[0][i] * Vt[i][0] << endl;*/
		R[0][0] += U[0][i] * Vt[i][0];
		R[1][0] += U[0][i] * Vt[i][1];
		R[0][1] += U[1][i] * Vt[i][0];
		R[1][1] += U[1][i] * Vt[i][1];
	}

	//numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])
	float** M = (float**)malloc(3 * sizeof(float*));
	for (int i = 0; i < 3; i++)
	{
		M[i] = (float*)malloc(3 * sizeof(float));
	}
	M[0][0] = (s2 / s1) * R[0][0]; M[0][1] = (s2 / s1) * R[0][1];
	M[0][2] = c2[0][0] - (s2 / s1) * (R[0][0] * c1[0][0] + R[0][1] * c1[0][1]);
	M[1][0] = (s2 / s1) * R[1][0]; M[1][1] = (s2 / s1) * R[1][1];
	M[1][2] = c2[0][1] - (s2 / s1) * (R[1][0] * c1[0][0] + R[1][1] * c1[0][1]);
	M[2][0] = 0.0; M[2][1] = 0.0; M[2][2] = 1.0;

	return M;
}

cv::Mat FaceDetection::warpImage(const cv::Mat& img, float landmark[5][2], float coord5Points[5][2]) {
	cv::Mat imgDst;
	float** M = transformationFromPoints(landmark, coord5Points);
	cv::Mat warpMat(2, 3, CV_32FC1);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			warpMat.at<float>(i, j) = M[i][j];
		}
	}
	warpAffine(img, imgDst, warpMat, img.size());
	return imgDst;
}
