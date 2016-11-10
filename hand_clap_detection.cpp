#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;
using namespace cv;

///////////全局变量 
int clapnum=0;///////////记录拍手次数

//8邻接种子算法，并返回每块区域的边缘框
void Seed_Filling(const cv::Mat& binImg, cv::Mat& labelImg, int& labelNum, int(&ymin)[20], int(&ymax)[20], int(&xmin)[20], int(&xmax)[20])   //种子填充法  
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	labelImg.release();
	binImg.convertTo(labelImg, CV_32SC1);
	int label = 1;
	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(j, i));     // 像素位置: <j,i>  
				++label;  // 没有重复的团，开始新的标签 
				ymin[label] = i;
				ymax[label] = i;
				xmin[label] = j;
				xmax[label] = j;
				while (!neighborPixels.empty())
				{
					std::pair<int, int> curPixel = neighborPixels.top(); //如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它  
					int curX = curPixel.first;
					int curY = curPixel.second;
					labelImg.at<int>(curY, curX) = label;

					neighborPixels.pop();

					if ((curX>0) && (curY>0) && (curX<(cols - 1)) && (curY<(rows - 1)))
					{
						if (labelImg.at<int>(curY - 1, curX) == 1)						//上
						{
							neighborPixels.push(std::pair<int, int>(curX, curY - 1));
							//ymin[label] = curY - 1;
						}
						if (labelImg.at<int>(curY + 1, curX) == 1)						//下
						{
							neighborPixels.push(std::pair<int, int>(curX, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
						}
						if (labelImg.at<int>(curY, curX - 1) == 1)						//左
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY));
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY, curX + 1) == 1)						//右
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY));
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;
						}
						if (labelImg.at<int>(curY - 1, curX - 1) == 1)					//左上
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY + 1, curX + 1) == 1)					//右下
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;

						}
						if (labelImg.at<int>(curY + 1, curX - 1) == 1)					//左下
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY - 1, curX + 1) == 1)					//右上
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;

						}
					}
				}
			}
		}
	}
	labelNum = label - 1;

}

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMarkers(const cv::Mat& markerImage) {

		// Convert to image of ints
		markerImage.convertTo(markers, CV_32S);
	}
	cv::Mat process(const cv::Mat &image) {

		// Apply watershed
		cv::watershed(image, markers);
		return markers;
	}
	// Return result in the form of an image
	cv::Mat getSegmentation() {

		cv::Mat tmp;
		// all segment with label higher than 255
		// will be assigned value 255
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}

	// Return watershed in the form of an image
	cv::Mat getWatersheds() {
		cv::Mat tmp;
		markers.convertTo(tmp, CV_8U, 255, 255);
		return tmp;
	}
};

///////////////对比度和亮度调整（用户）//////////
int g_nContrastValue; //对比度值
int g_nBrightValue;  //亮度值
Mat g_srcImage, g_dstImage;

/*static void ContrastAndBright(int, void *)
{

	// 三个for循环，执行运算 g_dstImage(i,j) = a*g_srcImage(i,j) + b
	for (int y = 0; y < g_srcImage.rows; y++)
	{
		for (int x = 0; x < g_srcImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}
}*/

//////////////////////////Color Transfer////////////////
void clip(Mat& img, float minval, float maxval)
{
	CV_Assert(maxval > minval);
	size_t row = img.rows;
	size_t col = img.cols;
	for (size_t i = 0; i != row; ++i)
	{
		float* temp = img.ptr<float>(i);
		for (size_t j = 0; j != col; ++j)
		{
			if (temp[j] < minval)
			{
				temp[j] = minval;
			}
			if (temp[j] > maxval)
			{
				temp[j] = maxval;
			}
		}
	}
}
void colorTransfer(const Mat& src, Mat& dst)
{
	Mat labsrc, labdst;
	cvtColor(src, labsrc, COLOR_BGR2Lab);
	cvtColor(dst, labdst, COLOR_BGR2Lab);
	labsrc.convertTo(labsrc, CV_32FC3);
	labdst.convertTo(labdst, CV_32FC3);
	//计算三个通道的均值与方差
	Scalar meansrc, stdsrc, meandst, stddst;
	meanStdDev(labsrc, meansrc, stdsrc);
	meanStdDev(labdst, meandst, stddst);
	//三通道分离
	vector<Mat> srcsplit, dstsplit;
	split(labsrc, srcsplit);
	split(labdst, dstsplit);
	//每个通道减去均值
	dstsplit[0] -= meandst[0];
	dstsplit[1] -= meandst[1];
	dstsplit[2] -= meandst[2];
	//每个通道缩放
	dstsplit[0] = stddst[0] / stdsrc[0] * dstsplit[0];
	dstsplit[1] = stddst[1] / stdsrc[0] * dstsplit[1];
	dstsplit[2] = stddst[2] / stdsrc[0] * dstsplit[2];
	//加上源图像的均值
	dstsplit[0] += meansrc[0];
	dstsplit[1] += meansrc[1];
	dstsplit[2] += meansrc[2];
	//控制溢出
	//clip(dstsplit[0], 0.0f, 255.0f);
	//clip(dstsplit[1], 0.0f, 255.0f);
	//clip(dstsplit[2], 0.0f, 255.0f);
	//转换为单字节单通道
	dstsplit[0].convertTo(dstsplit[0], CV_8UC1);
	dstsplit[1].convertTo(dstsplit[1], CV_8UC1);
	dstsplit[2].convertTo(dstsplit[2], CV_8UC1);
	//合并每个通道
	merge(dstsplit, dst);
	//从lab空间转换到RGB空间
	cvtColor(dst, dst, COLOR_Lab2BGR);
}

Mat LOGO1 = imread("LOGO1.jpg",1);
Mat LOGO2 = imread("LOGO2.jpg",1);
Mat LOGO3 = imread("LOGO3.jpg",1);
Mat LOGO4 = imread("LOGO4.jpg",1);
Mat LOGO5 = imread("LOGO5.jpg", 1);
Mat mask = imread("LOGO.jpg", 0);



//////////////////////////////////////////////////////////////////




int main()
{
	//设置视频读入，括号里面的数字是摄像头的选择，一般自带的是0
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat binImage, tmp;
	//////WB/////
	Mat srcImage, dstImage;//白平衡前、后
	vector<Mat> g_vChannels;
	/////WB//////
	Mat Y, Cr, Cb;
	vector<Mat> channels;
	//模板图片，是Cr颜色通道的人手图像截图
	Mat tmpl = imread("bwz.jpg", CV_8UC1);
	Mat clap = imread("DEMO2.jpg", CV_8UC1);
	//************************************************
	Mat src1 = imread("purpleflower.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src2 = imread("ocean_sunset.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src3 = imread("autumn.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src4 = imread("woods.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src5 = imread("IMG_2.jpg", CV_LOAD_IMAGE_COLOR);



	//设定对比度和亮度的初值
	/*g_nContrastValue = 80;
	g_nBrightValue = 80;*/

	

bool stop = false;
	while (!stop)
	{
		//读入视频帧，转换颜色空间，并分割通道
		cap >> frame;
		
		//////////////进行自动白平衡//////////////
		//分离通道
		frame.copyTo(srcImage);
		split(srcImage, g_vChannels);
		Mat imageBlueChannel = g_vChannels.at(0);
		Mat imageGreenChannel = g_vChannels.at(1);
		Mat imageRedChannel = g_vChannels.at(2);

		double imageBlueChannelAvg = 0;
		double imageGreenChannelAvg = 0;
		double imageRedChannelAvg = 0;

		//求各通道的平均值
		imageBlueChannelAvg = mean(imageBlueChannel)[0];
		imageGreenChannelAvg = mean(imageGreenChannel)[0];
		imageRedChannelAvg = mean(imageRedChannel)[0];

		//求出个通道所占增益
		double K = (imageRedChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;
		double Kb = K / imageBlueChannelAvg;
		double Kg = K / imageGreenChannelAvg;
		double Kr = K / imageRedChannelAvg;

		//更新白平衡后的各通道BGR值
		addWeighted(imageBlueChannel, Kb, 0, 0, 0, imageBlueChannel);
		addWeighted(imageGreenChannel, Kg, 0, 0, 0, imageGreenChannel);
		addWeighted(imageRedChannel, Kr, 0, 0, 0, imageRedChannel);

		merge(g_vChannels, dstImage);//图像各通道合并
		dstImage.copyTo(frame);
	
		//////////白平衡结束/////////////*/
//////////////////对比度与亮度///////////     卡帧明显
	/*dstImage.copyTo(g_srcImage);

		createTrackbar("对比度：", "frame", &g_nContrastValue, 300, ContrastAndBright);
		createTrackbar("亮   度：", "frame", &g_nBrightValue, 200, ContrastAndBright);
		
		ContrastAndBright(g_nContrastValue, 0);
		ContrastAndBright(g_nBrightValue, 0);

		//dstImage.copyTo(frame);
		g_dstImage.copyTo(frame);*/
		////////////////////////////////////
		cvtColor(frame, binImage, CV_BGR2GRAY);
		frame.copyTo(tmp);
		cvtColor(tmp, tmp, CV_BGR2YCrCb);
		split(tmp, channels);
		Cr = channels.at(1);
		Cb = channels.at(2);

		//肤色检测，输出二值图像
		for (int j = 1; j < Cr.rows - 1; j++)
		{
			uchar* currentCr = Cr.ptr< uchar>(j);
			uchar* currentCb = Cb.ptr< uchar>(j);
			uchar* current = binImage.ptr< uchar>(j);
			for (int i = 1; i < Cb.cols - 1; i++)
			{
				if ((currentCr[i] > 140) && (currentCr[i] < 170) && (currentCb[i] > 77) && (currentCb[i] < 123))
					current[i] = 255;
				else
					current[i] = 0;
			}
		}

		//形态学处理
		//dilate(binImage, binImage, Mat());
		dilate(binImage, binImage, Mat());

		//分水岭算法
		cv::Mat fg;
		cv::erode(binImage, fg, cv::Mat(), cv::Point(-1, -1), 6);
		// Identify image pixels without objects
		cv::Mat bg;
		cv::dilate(binImage, bg, cv::Mat(), cv::Point(-1, -1), 6);
		cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);
		// Show markers image
		cv::Mat markers(binImage.size(), CV_8U, cv::Scalar(0));
		markers = fg + bg;
		// Create watershed segmentation object
		WatershedSegmenter segmenter;
		segmenter.setMarkers(markers);
		segmenter.process(frame);
		Mat waterShed;
		waterShed = segmenter.getWatersheds();
		//imshow("watershed", waterShed);
		//获得区域边框
		threshold(waterShed, waterShed, 1, 1, THRESH_BINARY_INV);

		//8向种子算法，给边框做标记
		Mat labelImg;
		int label, ymin[20], ymax[20], xmin[20], xmax[20];
		Seed_Filling(waterShed, labelImg, label, ymin, ymax, xmin, xmax);

		//根据标记，对每块候选区就行缩放，并与模板比较
		Size dsize = Size(tmpl.cols, tmpl.rows);
		Size dsize2 = Size(clap.cols,clap.rows);
		//**********************************************************************

		float simi[20];
		float simi2[20];
		for (int i = 0; i < label; i++)
		{
			simi[i] = 1;
			if (((xmax[2 + i] - xmin[2 + i])>50) && ((ymax[2 + i] - ymin[2 + i]) > 50))
			{
				rectangle(frame, Point(xmin[2 + i], ymin[2 + i]), Point(xmax[2 + i], ymax[2 + i]), Scalar::all(255), 2, 8, 0);
				Mat rROI = Mat(dsize, CV_8UC1);
				resize(Cr(Rect(xmin[2 + i], ymin[2 + i], xmax[2 + i] - xmin[2 + i], ymax[2 + i] - ymin[2 + i])), rROI, dsize);
				Mat result;
				matchTemplate(rROI, tmpl, result, CV_TM_SQDIFF_NORMED);
				//**************************************************
				Mat rROI2 = Mat(dsize2, CV_8UC1);
				resize(Cr(Rect(xmin[2 + i], ymin[2 + i], xmax[2 + i] - xmin[2 + i], ymax[2 + i] - ymin[2 + i])), rROI2, dsize2);
				Mat result2;
				matchTemplate(rROI2, clap, result2, CV_TM_SQDIFF_NORMED);
				
				simi[i] = result.ptr<float>(0)[0];
				simi2[20] = result2.ptr<float>(0)[0];
				//cout << simi[i] << endl;
			}
		}

		//统计一下区域中的肤色区域比例
		float fuseratio[20];
		for (int k = 0; k < label; k++)
		{
			fuseratio[k] = 1;
			if (((xmax[2 + k] - xmin[2 + k])>50) && ((ymax[2 + k] - ymin[2 + k]) > 50))
			{
				int fusepoint = 0;
				for (int j = ymin[2 + k]; j < ymax[2 + k]; j++)
				{
					uchar* current = binImage.ptr< uchar>(j);
					for (int i = xmin[2 + k]; i < xmax[2 + k]; i++)
					{
						if (current[i] == 255)
							fusepoint += 1;
					}
				}
				fuseratio[k] = float(fusepoint) / ((xmax[2 + k] - xmin[2 + k])*(ymax[2 + k] - ymin[2 + k]));
				//cout << fuseratio[k] << endl;
			}
		}

		//给符合阈值条件的位置画框
		for (int i = 0; i < label; i++)
		{
			if ((simi[i]<0.01) && (fuseratio[i]<0.3))//TAG 原值0.65
				rectangle(frame, Point(xmin[2 + i], ymin[2 + i]), Point(xmax[2 + i], ymax[2 + i]), Scalar::all(255), 2, 8, 0);
			if ((simi2[i] < 0.001) && (fuseratio[i] < 0.23))//TAG 原值0.65
			{
				rectangle(frame, Point(xmin[2 + i], ymin[2 + i]), Point(xmax[2 + i], ymax[2 + i]), Scalar::all(255), 2, 8, 0);
				clapnum++;
				if (clapnum == 6)
					clapnum = 0;
			}

		}

		/////////////////////切换镜头风格//////////////
		Mat LOGO1ROI = frame(Rect(100, 150, LOGO1.cols, LOGO1.rows));
		Mat LOGO2ROI = frame(Rect(400, 200, LOGO2.cols, LOGO2.rows));
		Mat LOGO3ROI = frame(Rect(50, 50, LOGO3.cols, LOGO3.rows));
		Mat LOGO4ROI = frame(Rect(10, 250, LOGO4.cols, LOGO4.rows));
		Mat LOGO5ROI = frame(Rect(380, 50, LOGO5.cols, LOGO5.rows));

		cout << "test:"<<clapnum << endl;
		if (clapnum == 1)
		{
			colorTransfer(src1, frame);
			LOGO1.copyTo(LOGO1ROI, mask);
		}
		if (clapnum == 2)
		{
			colorTransfer(src2, frame);
			LOGO2.copyTo(LOGO2ROI, mask);
		}
		if (clapnum == 3)
		{
			colorTransfer(src3, frame);
			LOGO3.copyTo(LOGO3ROI, mask);
		}
		if (clapnum == 4)
		{
			colorTransfer(src4, frame);
			LOGO4.copyTo(LOGO4ROI, mask);
		}
		if (clapnum == 5)
		{
			colorTransfer(src5, frame);
			LOGO5.copyTo(LOGO5ROI, mask);
		}

		imshow("frame", frame);
		//processor.writeNextFrame(frame);
		imshow("test", binImage);

	if (waitKey(1) >= 0)
			stop = true;
	}
	cout << "ss" << endl;
	cv::waitKey();
	return 0;
}
