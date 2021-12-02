//=====================================================================================================================
// Includes
//=====================================================================================================================
#include "vs/AVPLines/CSlot.hpp"

//Slot constructors
CSlot::CSlot(float f_heading, cv::Point2i f_point_1, cv::Point2i f_point_2, cv::Point2i f_point_3, cv::Point2i f_point_4,
	int8_t f_id, int16_t f_width, std::string f_orientation)
{
	m_heading = f_heading;
	m_point_1 = f_point_1;
	m_point_2 = f_point_2;
	m_point_3 = f_point_3;
	m_point_4 = f_point_4;
	m_id = f_id;
	m_width = f_id;
	m_orientation = f_orientation;
}
CSlot::CSlot() {}

//Getters
float CSlot::getHeading() { return m_heading; }
int8_t CSlot::getId() { return m_id; }
cv::Point2i CSlot::getPoint1() { return m_point_1; }
cv::Point2i CSlot::getPoint2() { return m_point_2; }
cv::Point2i CSlot::getPoint3() { return m_point_3; }
cv::Point2i CSlot::getPoint4() { return m_point_4; }
int16_t CSlot::getWidth() { return m_width; }
std::string CSlot::getOrientation() { return m_orientation; }


//Setters
void CSlot::setHeading ( float f_heading ) { m_heading = f_heading; }
void CSlot::setId ( int8_t f_id ) { m_id = f_id; }
void CSlot::setPoint1 ( cv::Point2i f_point_1 ) { m_point_1 = f_point_1; }
void CSlot::setPoint2 ( cv::Point2i f_point_2 ) { m_point_2 = f_point_2; }
void CSlot::setPoint3 ( cv::Point2i f_point_3 ) { m_point_3 = f_point_3; }
void CSlot::setPoint4 ( cv::Point2i f_point_4 ) { m_point_4 = f_point_4; }
void CSlot::setWidth ( int16_t f_width ) { m_width = f_width; }
void CSlot::setOrientation(std::string f_orientation) { m_orientation = f_orientation; }