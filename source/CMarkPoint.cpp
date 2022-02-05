//=====================================================================================================================
// Includes
//=====================================================================================================================
#include "vs/CPD/CMarkPoint.hpp"

//MarkPoint constructors
CMarkPoint::CMarkPoint(int f_length, std::vector<cv::Point2d> f_points_vector, cv::Mat f_image)
{
	m_length = f_length;
	m_points_vector = f_points_vector;
	m_image = f_image;
}
CMarkPoint::CMarkPoint() { m_length = 0; }

//Getters
int CMarkPoint::getLenght() { return m_length; }
std::vector<cv::Point2d> CMarkPoint::getPoints() { return m_points_vector; }
cv::Mat CMarkPoint::getImage() { return m_image; };

//Setters
void CMarkPoint::setLength(int f_length) { m_length = f_length; }
void CMarkPoint::setPoints(std::vector<cv::Point2d> f_points_vector) { m_points_vector = f_points_vector; }
void CMarkPoint::setImage(cv::Mat f_image) { m_image = f_image; }
