//=====================================================================================================================
// Include guards
//=====================================================================================================================

#ifndef MAIN_HPP
#define MAIN_HPP

//=====================================================================================================================
// Defines and macros
//=====================================================================================================================
#ifdef _WIN32
#ifdef PCD_EXPORTS
#define PCD_API __declspec(dllexport)
#else
#define PCD_API __declspec(dllimport)
#endif
#else
#define PCD_API
#endif

//=====================================================================================================================
// Includes
//=====================================================================================================================
#include <vs/PCD/CFile.hpp>
#include <vs/PCD/CMarkPoint.hpp>
#include <vs/PCD/CSlot.hpp>

#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <opencv2/ml.hpp>

// Main
int main(int argc, const char *argv[]);

// Methods
void writeNegativeTrainingSet(std::vector<std::string> &f_file_bmp_names, std::list<cv::Mat> &f_negative_samples_list,
							  CFile &f_file_object, uint8_t f_width, uint8_t f_height);

void writePositiveTrainingSet(std::vector<std::string> &f_file_json_names, std::vector<std::string> &f_file_bmp_names,
							  std::list<CMarkPoint> &f_mark_point_list, CFile &f_file_object, uint8_t f_width, uint8_t f_height);

void writeAnnotationsTestSet(std::vector<std::string> &f_file_json_names, std::vector<std::string> &f_file_bmp_names,
							 std::list<CMarkPoint> &f_mark_point_list, CFile &f_file_object, uint8_t f_width, uint8_t f_height);

void featuresExtractor(CFile &f_file_object, std::vector<std::string> &f_file_bmp_names,
					   std::list<cv::Mat> &f_detector_lst, uint8_t f_width, uint8_t f_height);

void stretchHistogram(cv::Mat &f_image);

void computeHOG(const cv::Mat &f_image, std::vector<cv::Mat> &f_gradient_lst, uint8_t f_width, uint8_t f_height);

void computeHOGs(const cv::Mat &f_image, std::vector<cv::Mat> &f_gradient_lst, uint8_t f_width, uint8_t f_height);

void convertToML(const std::vector<cv::Mat> &f_train_samples, cv::Mat &f_trainData);

void trainDetectors(CFile &f_file_object, const std::string &f_model_path);

void predictImages(const std::string &f_model_path, CFile &f_file_object, uint8_t f_width, uint8_t f_height);

float distanceSample(cv::Mat &f_sample, const cv::Ptr<cv::ml::SVM> &f_svm);

void slotInference(std::list<CSlot> &f_slot_list, std::vector<cv::Rect> &f_right_rectangles, std::vector<cv::Rect> &f_up_rectangles,
				   std::vector<cv::Rect> &f_left_rectangles, std::vector<cv::Rect> &f_down_rectangles, cv::Mat &f_image);

void correctAndDrawLeftSlots(const std::vector<cv::Rect> &f_left_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list);
void correctAndDrawRightSlots(const std::vector<cv::Rect> &f_right_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list);
void correctAndDrawUpSlots(const std::vector<cv::Rect> &f_up_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list);
void correctAndDrawDownSlots(const std::vector<cv::Rect> &f_down_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list);

float angleOf(const cv::Point2f &f_p1, const cv::Point2f &f_p2);

void groupRectanglesModified(std::vector<cv::Rect> &rectList, int groupThreshold, double eps, std::vector<int> *weights, std::vector<double> *levelWeights);
#endif

	//=====================================================================================================================
	// End of File
	//=====================================================================================================================
