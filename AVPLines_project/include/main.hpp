
#include <CFile.hpp>
#include <CMarkPoint.hpp>

#include <string>
#include <iostream>
#include <list>
#include <vector>

//Main
int main(int argc, const char* argv[]); 

//Methods
void writeNegativeTrainingSet( std::vector< std::string >& f_file_bmp_names, std::list< cv::Mat >& f_negative_samples_list, 
	CFile& f_file_object, uint8_t f_width, uint8_t f_height );

void writePositiveTrainingSet( std::vector< std::string >& f_file_json_names, std::vector< std::string >& f_file_bmp_names,
	std::list< CMarkPoint >& f_mark_point_list, CFile& f_file_object, uint8_t f_width, uint8_t f_height );

void featuresExtractor( CFile& f_file_object, std::vector< std::string >& f_file_bmp_names,
	std::list< cv::Mat >& f_detector_lst, uint8_t f_width, uint8_t f_height);

void stretchHistogram( cv::Mat& f_image );

void computeHOG( const cv::Mat& f_image, std::vector< cv::Mat >& f_gradient_lst, uint8_t f_width, uint8_t f_height );

void convertToML( const std::vector< cv::Mat >& f_train_samples, cv::Mat& f_trainData );

void trainDetectors( CFile& f_file_object, const std::string& f_model_path );

void predictImages( const std::string& f_model_path, CFile& f_file_object, uint8_t f_width, uint8_t f_height );

