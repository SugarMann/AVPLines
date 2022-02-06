//=====================================================================================================================
// Include guards
//=====================================================================================================================

#ifndef CFILE_HPP
#define CFILE_HPP

//=====================================================================================================================
// Defines and macros
//=====================================================================================================================
// #ifdef _WIN32
// #    ifdef PCD_EXPORTS
// #       define PCD_API __declspec(dllexport)
// #    else
// #       define PCD_API __declspec(dllimport)
// #    endif
// #else
// #    define PCD_API
// #endif

//=====================================================================================================================
// Includes
//=====================================================================================================================
#include <string>
#include <vector>
#include <list>

#include "vs/PCD/CMarkPoint.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

class CFile
{
private:
	//Private variables
	std::string m_path;

public:
	//CFile constructors
	CFile(std::string f_path);
	CFile();

	//Getters
	std::string getPath();
	//Setters
	void setPath(std::string f_path);

	//Methods
	void fileNamesByExtension(std::string f_extension, std::vector<std::string> &f_file_names);

	void readJson(const std::vector<std::string> &f_file_names, std::list<CMarkPoint> &f_mark_point_list);
	void writeCSV(const std::string &f_filename, cv::Mat &f_m);
	void readCSV(cv::Mat &f_m);

	void readBmp(const std::vector<std::string> &f_file_names, std::list<CMarkPoint> &f_mark_point_list);
	void readBmp(const std::vector<std::string> &f_file_names, std::list<cv::Mat> &f_image_list);

	void makePositiveTrainingSet(std::list<CMarkPoint> &f_mark_point_list, uint8_t f_width, uint8_t f_height);
	void makeNegativeTrainingSet(std::list<cv::Mat> &f_image_list, uint8_t f_width, uint8_t f_height);
	void makeAnnotationsTestSet(std::list<CMarkPoint> &f_mark_point_list, uint8_t f_width, uint8_t f_height);

	void logResults(uint32_t f_frameCounter, const std::string f_logFilePath, const std::vector<cv::Rect> &f_croppedImages);
	void logGetFramesMember(const std::vector<cv::Rect> &f_croppedImages, rapidjson::Document &f_document, rapidjson::Value &f_frames, uint32_t f_frameCounter);
};
#endif

//=====================================================================================================================
// End of File
//=====================================================================================================================
