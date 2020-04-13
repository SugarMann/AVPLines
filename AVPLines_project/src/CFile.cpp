#include "CFile.hpp"

#include <json/json.h>
#include <Windows.h>
#include <fstream>
#include <iterator>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

//CFile constructors
CFile::CFile(std::string f_path)
{
	m_path = f_path;
};
CFile::CFile() { m_path = ""; };

//CFile getters
std::string CFile::getPath() { return m_path; };

//CFile setters
void CFile::setPath(std::string f_path) { m_path = f_path; };

//Methods
void CFile::fileNamesByExtension(std::string f_extension, std::vector<std::string>& f_file_names)
{
	std::string f_search_path = m_path + "\\*.*" + f_extension;
	WIN32_FIND_DATA f_fd;
	HANDLE f_hFind = ::FindFirstFile(f_search_path.c_str(), &f_fd);

	if (f_hFind != INVALID_HANDLE_VALUE)
	{
		while (::FindNextFile(f_hFind, &f_fd))
		{
			if (!(f_fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				f_file_names.push_back(f_fd.cFileName);
		}
	}
}

void CFile::readJson(const std::vector<std::string>& f_file_names, std::list<CMarkPoint>& f_mark_point_list)
{
	//Variables
	cv::Point2d l_point1, l_point2;
	std::vector<cv::Point2d> l_points_vector;
	CMarkPoint l_mark_point_object;
	Json::Value l_mark_point_json;

	//Read JSON file
	for (size_t i = 0; i < f_file_names.size(); i++)
	{
		std::ifstream mark_point_file(m_path + "\\" + f_file_names[i], std::ifstream::binary);
		mark_point_file >> l_mark_point_json;

		//Loop to save all MarkPoints
		for (int j = 0; j < l_mark_point_json["lenght"].asInt(); j++)
		{
			//Data are organized in groups of two points
			l_point1 = { l_mark_point_json["data"][j][0].asDouble(), l_mark_point_json["data"][j][1].asDouble() };
			l_point2 = { l_mark_point_json["data"][j][2].asDouble(), l_mark_point_json["data"][j][3].asDouble() };
			//Save points in vector
			l_points_vector.push_back(l_point1);
			l_points_vector.push_back(l_point2);
			//Save vectors of points in object list
			l_mark_point_object.setPoints(l_points_vector);
		}

		l_mark_point_object.setLength(l_mark_point_json["lenght"].asInt());
		f_mark_point_list.push_back(l_mark_point_object);

		//Clear variables
		l_points_vector.clear();
	}
}

void CFile::readBmp(const std::vector< std::string >& f_file_names, std::list< CMarkPoint >& f_mark_point_list)
{
	std::list<CMarkPoint>::iterator l_it = f_mark_point_list.begin();
	cv::Mat l_image;

	for (size_t i = 0; i<f_file_names.size(); i++)
	{
		l_image = cv::imread(m_path + "\\" + f_file_names[i]);
		l_it->setImage(l_image);
		std::advance(l_it, 1);

	}
}

void CFile::readBmp(const std::vector< std::string >& f_file_names, std::list< cv::Mat >& f_image_list)
{
	for ( std::string l_file : f_file_names )
	{
		f_image_list.push_back( cv::imread(m_path + "\\" + l_file) );
	}
}

void CFile::makePositiveTrainingSet(std::list<CMarkPoint>& f_mark_point_list)
{
	cv::Mat l_image;
	cv::Rect l_roi; //Roi markpoint
	cv::Mat l_crop;

	std::vector<cv::Point2d> l_points;
	int l_cont = 0;

	for (CMarkPoint markpoint : f_mark_point_list)
	{
		l_image = markpoint.getImage();
		l_points = markpoint.getPoints();
		cv::Point2d l_direction_point;
		cv::Point2d l_central_point;
		double l_diff_x, l_diff_y;
		int l_direction; // 0 - right, 1 - up, 2 - left, 3 - down

		//Loop through each MarkPoint
		for (int i = 0; i < markpoint.getLenght()*2; i++)
		{
			//Only pair points in list are central points
			if (i % 2 == 0) {

				l_central_point = l_points[i];
				l_direction_point = l_points[i+1.0];

				//Correct negative and extended positions
				l_roi.x = l_central_point.x - 32.0;
				if (l_roi.x < 0.0)
					l_roi.x = 0.0;

				l_roi.y = l_central_point.y - 32.0;
				if (l_roi.y < 0.0)
					l_roi.y = 0.0;

				l_roi.width = 64.0;
				if ((double)l_roi.y + (double)l_roi.width > 600.0)
					l_roi.y = l_image.cols - l_roi.width;

				l_roi.height = 64.0;
				if ((double)l_roi.x + (double)l_roi.height > 600.0)
					l_roi.x = l_image.rows - l_roi.height;

				l_crop = l_image(l_roi);

				//Make 4 training sets depending of directions of MarkPoints
				l_diff_x = std::abs(l_direction_point.x - l_central_point.x);
				l_diff_y = std::abs(l_direction_point.y - l_central_point.y);

				//If the difference between 'x' coordenates is bigger than 'y' coordinates, the direction is right or left
				if (l_diff_x > l_diff_y) {
					if (l_direction_point.x - l_central_point.x < 0)
						l_direction = 2;
					else
						l_direction = 0;
				}
				//Else the direction is up or down
				else {
					if (l_direction_point.y - l_central_point.y < 0)
						l_direction = 1;
					else
						l_direction = 3;
				}

				//Debug show croped images
				//cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
				//cv::imshow("Display", l_crop);
				//cv::waitKey(20);
				
				//Write images in folder
				l_cont++;
				cv::imwrite(m_path + "\\dataset\\" + std::to_string(l_direction) + "\\" + std::to_string(l_cont) + ".bmp", l_crop);

				//If we flip the image to opposite direction, we can multiplicate by 2 our dataset for each detector
				switch (l_direction)
				{
				case 0:
					cv::flip(l_crop, l_crop, 1);
					l_cont++;
					cv::imwrite(m_path + "\\dataset\\2\\" + std::to_string(l_cont) + ".bmp", l_crop);
					break;
				case 2:
					cv::flip(l_crop, l_crop, 1);
					l_cont++;
					cv::imwrite(m_path + "\\dataset\\0\\" + std::to_string(l_cont) + ".bmp", l_crop);
 					break;
				case 1:
					cv::flip(l_crop, l_crop, 0);
					l_cont++;
					cv::imwrite(m_path + "\\dataset\\3\\" + std::to_string(l_cont) + ".bmp", l_crop);
					break;
				case 3:
					cv::flip(l_crop, l_crop, 0);
					l_cont++;
					cv::imwrite(m_path + "\\dataset\\1\\" + std::to_string(l_cont) + ".bmp", l_crop);
					break;
				default:
					break;
				}
			}
		}
	}
}

void CFile::makeNegativeTrainingSet( std::list< cv::Mat >& f_image_list )
{
	//Variables
	cv::Rect l_roi; //Roi markpoint
	cv::Mat l_crop;
	int l_count_white, l_count_black, l_count_green;
	int l_count = 0;
	cv::Vec3b l_pixel_value;

	for ( cv::Mat l_image : f_image_list )
	{

		l_count++;
		//Debug show image
		//cv::namedWindow("Display image", cv::WINDOW_AUTOSIZE);
		//cv::imshow("Display image", l_image);
		//cv::waitKey(10);

		for ( int i = 0, increment = 64; i < l_image.rows; i += increment )
		{
			for ( int j = 0, increment2 = 64; j < l_image.cols; j += increment2 )
			{
				l_count_black = 0;
				l_count_white = 0;
				l_count_green = 0;
				l_roi.x = i;
				l_roi.y = j;
				l_roi.width = 64;
				l_roi.height = 64;

				if ( (l_roi.x + l_roi.width) > l_image.cols )
					break;
				else if ( (l_roi.y + l_roi.height) > l_image.rows )
					break;
				else
					l_crop = l_image(l_roi);

				//debug show all cropped images
				//cv::namedWindow("DisplayBefore", cv::WINDOW_KEEPRATIO);
				//cv::imshow("DisplayBefore", l_crop);
				//cv::waitKey(20);

				//Black and white pixels counter
				for ( int x = 0; x < l_crop.rows; x++ )
				{
					for ( int y = 0; y < l_crop.cols; y++ )
					{
						l_pixel_value = l_crop.at<cv::Vec3b>(x, y);

						if ( l_pixel_value == cv::Vec3b(255, 255, 255) )
							l_count_white++;
						if ( l_pixel_value == cv::Vec3b(0, 0, 0) )
							l_count_black++;
						if (l_pixel_value[0] <= 50 && l_pixel_value[1] >= 200 && l_pixel_value[2] <= 50)
							l_count_green++;
					}
				}

				//If more than 30% of pixels are white or black, it won't be processed
				int threshold = l_crop.cols * l_crop.rows * 0.3;
				if ( l_count_white > threshold )
					continue;
				if ( l_count_black > threshold )
					continue;
				if ( l_count_green > 0 )
					continue;

				//Debug show finally croped images
				//cv::namedWindow("Display", cv::WINDOW_KEEPRATIO);
				//cv::imshow("Display", l_crop);
				//cv::waitKey(20);

				//increase counter and write negative dataset
				l_count++;
				cv::cvtColor(l_crop, l_crop, cv::COLOR_RGB2GRAY);
				cv::imwrite("resources/negative_samples/negative_" + std::to_string(l_count) + ".bmp", l_crop);

			}
		}
	}
}

void CFile::writeCSV(const std::string& f_filename, cv::Mat& f_m)
{
	std::ofstream l_myfile;
	l_myfile.open(f_filename.c_str());
	l_myfile << cv::format(f_m, cv::Formatter::FMT_CSV) << std::endl;
	l_myfile.close();
}

void CFile::readCSV( cv::Mat& f_m )
{
	cv::Ptr<cv::ml::TrainData> l_raw_data = cv::ml::TrainData::loadFromCSV(m_path, 0, -2, 0);
	f_m = l_raw_data->getSamples();
}