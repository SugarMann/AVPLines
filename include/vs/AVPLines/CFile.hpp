//=====================================================================================================================
// Include guards
//=====================================================================================================================

#ifndef CFILE_HPP
#define CFILE_HPP

//=====================================================================================================================
// Defines and macros
//=====================================================================================================================
// #ifdef _WIN32
// #    ifdef AVPLINES_EXPORTS
// #       define AVPLINES_API __declspec(dllexport)
// #    else
// #       define AVPLINES_API __declspec(dllimport)
// #    endif
// #else
// #    define AVPLINES_API
// #endif

//=====================================================================================================================
// Includes
//=====================================================================================================================
#include <string>
#include <vector>
#include <list>

#include "vs/AVPLines/CMarkPoint.hpp"

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
	void setPath( std::string f_path );

	//Methods
	void fileNamesByExtension(std::string f_extension, std::vector< std::string >& f_file_names);

	void readJson(const std::vector< std::string >& f_file_names, std::list< CMarkPoint >& f_mark_point_list);
	void writeCSV(const std::string& f_filename, cv::Mat& f_m);
	void readCSV(cv::Mat& f_m);

	void readBmp(const std::vector< std::string >& f_file_names, std::list< CMarkPoint >& f_mark_point_list);
	void readBmp(const std::vector< std::string >& f_file_names, std::list< cv::Mat >& f_image_list);

	void makePositiveTrainingSet(std::list< CMarkPoint >& f_mark_point_list, uint8_t f_width, uint8_t f_height);
	void makeNegativeTrainingSet(std::list< cv::Mat >& f_image_list, uint8_t f_width, uint8_t f_height);

};
#endif


//=====================================================================================================================
// End of File
//=====================================================================================================================

