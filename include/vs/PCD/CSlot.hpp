//=====================================================================================================================
// Include guards
//=====================================================================================================================

#ifndef CSLOT_HPP
#define CSLOT_HPP

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
#include <opencv2/core/core.hpp>
#include <vector>

class CSlot
{

private:

	//			  ^
	//		   	  | Heading = 90�
	//			  |
	//
    // (P1)T<---width--->T(P2)
	//	   |             |
	//	   |             |
	//	   |    Slot     |
	//	   |             |
	//	   |             |
	// (P3)|             |(P4)

	//Private variables
	float m_heading; //Orientation slot
	cv::Point2i m_point_1;
	cv::Point2i m_point_2;
	cv::Point2i m_point_3;
	cv::Point2i m_point_4;
	int8_t m_id;
	int16_t m_width; //In pixels
	std::string m_orientation;

public:
	//CSlot constructors
	CSlot(float f_heading, cv::Point2i f_point_1, cv::Point2i f_point_2, cv::Point2i f_point_3, cv::Point2i f_point_4, 
		int8_t f_id, int16_t f_width, std::string f_orientation);
	CSlot();

	//Getters
	float getHeading();
	int8_t getId();
	cv::Point2i getPoint1();
	cv::Point2i getPoint2();
	cv::Point2i getPoint3();
	cv::Point2i getPoint4();
	int16_t getWidth();
	std::string getOrientation();

	//Setters
	void setHeading ( float f_heading );
	void setPoint1 ( cv::Point2i f_point_1 );
	void setPoint2 ( cv::Point2i f_point_2 );
	void setPoint3 ( cv::Point2i f_point_3 );
	void setPoint4 ( cv::Point2i f_point_4 );
	void setId ( int8_t f_id );
	void setWidth ( int16_t f_width );
	void setOrientation ( std::string f_orientation );

};
#endif


//=====================================================================================================================
// End of File
//=====================================================================================================================

