//=====================================================================================================================
//  C O P Y R I G H T
//---------------------------------------------------------------------------------------------------------------------
//  Copyright (c) 2020 by  CTAG - Centro Tecnológico de Automoción de Galicia. All rights reserved.
//
//  This file is property of CTAG. Any unauthorized copy, use or distribution is an offensive act against international 
//  law and may be prosecuted under federal law. Its content is company confidential.
//  
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
//! @file           main.cpp
//! @brief          File with the main function to execute a sample of the CTAG Vision Systems C++ Template
//! @authors        Author1 Name (email@ctag.com)
//!                 Author2 Name (email@ctag.com)
//! @date           2020
//! @copyright      CTAG - Centro Tecnológico de Automoción de Galicia.
//! @details        More elaborated file description [If needed]
//! @note           Additional important information/change [If needed]. Note that this four tags: todo, note, bug and
//!                 warning could be used also for classes, functions or other entities.
//! @bug            Known bugs to solve in future versions [If any]
//! @todo           Pending issues that should be handled [If any]
//! @warning        Important known warnings before use [If any]. WARNING: If the comment exceeds the length of the
//!                 line, next line should be idented.
//---------------------------------------------------------------------------------------------------------------------


//=====================================================================================================================
// Includes
//=====================================================================================================================
#include "vs/template/header1.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <string>


//=====================================================================================================================
// Definitions
//=====================================================================================================================


//=====================================================================================================================
// Constants 
//=====================================================================================================================


//=====================================================================================================================
// Implementation 
//=====================================================================================================================


//! @brief          Main function to execute the template_project.
//! @param[in]      argc Non-negative value wiht the number of arguments passed to the program
//! @param[in]      argv Pointer to the first element of an array of pointers with the arguments passed to the program.
//!                 The last element of the array is a null pointer.
//! @returns        Error code
int main(int argc, char *argv[])
{
	// USE OUR INTERNAL FUNCTION
	int32_t l_inputA = 10;
	int32_t l_inputB = 20;
	int32_t l_inputC = 30;
	int32_t l_output = tmpl::INTERNAL_CONST;

	std::cout << "Variables before calling doCalculations..." << std::endl;
	std::cout << "  Input A: " <<  l_inputA << std::endl;
	std::cout << "  Input B: " << l_inputB << std::endl;
	std::cout << "  Input C: " << l_inputC << std::endl;
	std::cout << "  Output: " << l_output << std::endl << std::endl;

	tmpl::doCalculations(l_inputA, l_inputB, l_inputC, l_output);

	std::cout << "Variables after calling doCalculations" << std::endl;
	std::cout << "  Input A: " << l_inputA << std::endl;
	std::cout << "  Input B: " << l_inputB << std::endl;
	std::cout << "  Input C: " << l_inputC << std::endl;
	std::cout << "  Output: " << l_output << std::endl << std::endl;


	// MANAGE THE CLASS WE HAVE DEFINED
	tmpl::TemplateClass l_test;
	std::cout << "l_test initial state..." << std::endl;
	std::cout << "  Internal Value: " << l_test.getInternalValue() << std::endl << std::endl;
	
	// Manual initialization
	l_test.initialize(40);
	std::cout << "l_test state after manual initialization..." << std::endl;
	std::cout << "  Internal Value: " << l_test.getInternalValue() << std::endl << std::endl;

	// Processing
	l_test.process();
	std::cout << "l_test state after processing..." << std::endl;
	std::cout << "  Internal Value: " << l_test.getInternalValue() << std::endl << std::endl;


	// SHOW A SAMPLE IMAGE USING OPENCV
	cv::Mat l_image;

	// Read Image
	l_image = cv::imread("../../../../../resources/shared/logos/ctag_logo_color_1280x800.png", 1);   // In general, 
															// sample images may be stored at the "extras/" folder in 
															// the root of the project repo (thus, not being backed up 
															// at the Git remote, but in the "git_companion" shared 
															// folder). 
															// In this case we use a shared logo image from the 
															// `resources` repository.

	// Check Errors
	if (!l_image.data) // Check 
	{
		std::cout << "ERROR: Image could not be opened/found." << std::endl;
		return -1;
	}
	else
	{
		std::cout << "Prepared to show CTAG Logo..." << std::endl;
	}

	// Show Image
	namedWindow("CTAG Logo Window", cv::WINDOW_AUTOSIZE);
	imshow("CTAG Logo Window", l_image);

	// Wait for a keystroke and return
	cv::waitKey(0);
	return 0;
}


//=====================================================================================================================
// End of File
//=====================================================================================================================

