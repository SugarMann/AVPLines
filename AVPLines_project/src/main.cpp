#include <stdlib.h>
#include <stdio.h>

#include <main.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

//Define to write the markpoint dataset
//#define MAKE_TRAINING_SET

//Define to trace debug
//#define DEBUG_LABEL_TRACE

//Define to extract features
#define EXTRACT_FEATURES

//Define to train detectors
#define TRAIN_DETECTORS

int main( int argc, const char* argv[] )
{
	//Variables
	std::vector<std::string> l_file_json_names;			//Names of json files where it is the markpoints information
	std::vector<std::string> l_file_bmp_names;			//Names of image files
	std::list<CMarkPoint> l_mark_point_list;			//Mark points list
	std::list<cv::Mat> l_detector_lst;					//Detector list for each direction slot parking

	CFile l_file_object;

	if (argc != 2) // argc must be 2 for correct execution
	{
		std::cout << "usage: " << argv[0] << " <path>\n"; // We assume that argv[0] is the program name
		return 1;
	}
	else
	{
		//Set the path in CFile object
		l_file_object.setPath( argv[1] );

		#ifdef MAKE_TRAINING_SET
		writeTrainingSet( l_file_json_names, l_file_bmp_names, l_mark_point_list, l_file_object );
		#endif // MAKE_TRAINING_SET

		#ifdef EXTRACT_FEATURES
		featuresExtractor(l_file_object, l_file_bmp_names, l_detector_lst);
		#endif // EXTRACT_FEATURES

		#ifdef TRAIN_DETECTORS

		#endif // TRAIN_DETECTORS

		cv::Mat asd;
	}

	return 0;
}

// Write training set module
//---------------------------------------------------------
void writeTrainingSet( std::vector<std::string>& f_file_json_names, std::vector<std::string>& f_file_bmp_names, std::list<CMarkPoint>& f_mark_point_list, CFile& f_file_object )
{
	//Make a vector with all json file names in "positive samples"
	f_file_object.fileNamesByExtension( "json", f_file_json_names );
	//Make a vector with all bmp file names in "positive samples"
	f_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );

	//Make a JSON MarkPoints object list, where each object have all MarkPoints in one image
	f_file_object.readJson( f_file_json_names, f_mark_point_list );
	f_file_json_names.clear();
	//Set related image for every MarkPoint object
	f_file_object.readBmp( f_file_bmp_names, f_mark_point_list );
	f_file_bmp_names.clear();
	//Make the training set images with MarkPoint list information
	f_file_object.makeTrainingSet( f_mark_point_list );
}
//---------------------------------------------------------

// Extract features module
//---------------------------------------------------------
void featuresExtractor( CFile& f_file_object, std::vector<std::string>& f_file_bmp_names, std::list<cv::Mat>& f_detector_lst )
{
	//Variables
	cv::Mat l_mat;
	CFile l_file_object;
	std::vector<cv::Mat> l_gradient_vec;

	//4 differents detector are trained (right, up, left, down markpoints directions)
	for ( int i = 0; i <= 3; i++ )
	{
		//Prepare different paths for each detector
		l_file_object.setPath( f_file_object.getPath() + "\\dataset\\" + std::to_string(i) );

		//Read images and save them in a markpoint list 
		l_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );
		std::list<CMarkPoint> l_mark_point_list( (int)f_file_bmp_names.size(), CMarkPoint() );
		l_file_object.readBmp( f_file_bmp_names, l_mark_point_list );

		for ( CMarkPoint l_mark_point: l_mark_point_list )
		{
			//Image to grayscale
			cv::cvtColor(l_mark_point.getImage(), l_mat, cv::COLOR_RGB2GRAY);

			#ifdef DEBUG_LABEL_TRACE
			cv::imshow( "Gray image", l_mat );
			cv::waitKey(0);
			#endif // DEBUG_LABEL_TRACE

			//1st Feature - Stretch the grayscale histogram
			stretchHistogram(l_mat);

			#ifdef DEBUG_LABEL_TRACE
			cv::imshow( "Stretched gray image", l_mat );
			cv::waitKey(0);
			#endif // DEBUG_LABEL_TRACE

			//2nd Feature - Gradient Magnitude
			computeHOG( l_mat, l_gradient_vec, i );
			
		}
		convertToML( l_gradient_vec, l_mat );
		f_detector_lst.push_back( l_mat );
		l_gradient_vec.clear();
		f_file_bmp_names.clear();
		l_mark_point_list.clear();

	}
}

void stretchHistogram( cv::Mat& f_image )
{
	//Variables
	double_t l_min, l_max;
	uint8_t* l_pixelPtr = (uint8_t*)f_image.data;

	//Calculate min and max values
	cv::minMaxLoc(f_image, &l_min, &l_max);
	
	//Calculate new intensity to stretch histogram
	//Link -> https://en.wikipedia.org/wiki/Normalization_(image_processing)/
	for (int i = 0; i < f_image.rows; i++) 
	{
		for (int j = 0; j < f_image.cols; j++) 
		{
			l_pixelPtr[i*f_image.cols + j] = (l_pixelPtr[i * f_image.cols + j] - l_min) * (((255 - 0) / (l_max - l_min)) + 0);
		}
	}
}

void computeHOG( const cv::Mat& f_image, std::vector<cv::Mat>& f_gradient_lst, int f_orientation )
{
	//Variables
	std::vector<float> l_descriptors;

	//Histogram Of Gradients descriptor
	cv::HOGDescriptor hog(
		cv::Size(64, 64),							//winSize
		cv::Size(16, 16),							//blockSize
		cv::Size(8, 8),								//blockStride
		cv::Size(8, 8),								//cellSize
		9,											//nbins
		1,											//derivAper
		-1,											//winSigma
		cv::HOGDescriptor::HistogramNormType(0),	//histogramNormType
		0.2,										//L2HysThersh
		true,										//gammal correction
		64,											//nlevels
		true										//use signed gradients
		);
	
	//Calculate magnitude and angle descriptors with hog and save them in gradients list
	hog.compute( f_image, l_descriptors );
	f_gradient_lst.push_back( cv::Mat( l_descriptors ).clone() );
}

// Convert training/testing set to be used by OpenCV Machine Learning algorithms.
// TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
// Transposition of samples are made if needed.

void convertToML( const std::vector<cv::Mat>& f_train_samples, cv::Mat& f_trainData )
{
	//Variables
	const int l_rows = static_cast<int> ( f_train_samples.size() );
	const int l_cols = static_cast<int> (std::max(f_train_samples[0].cols, f_train_samples[0].rows) );
	cv::Mat l_tmp( 1, l_cols, CV_32FC1 ); //< used for transposition if needed

	f_trainData = cv::Mat( l_rows, l_cols, CV_32FC1 );
	
	for ( size_t i = 0; i < f_train_samples.size(); ++i )
	{
		CV_Assert( f_train_samples[i].cols == 1 || f_train_samples[i].rows == 1 );
		if ( f_train_samples[i].cols == 1 )
		{
			cv::transpose( f_train_samples[i], l_tmp );
			l_tmp.copyTo( f_trainData.row( static_cast<int>(i) ) );
		}
		else if ( f_train_samples[i].rows == 1 )
		{
			f_train_samples[i].copyTo( f_trainData.row( static_cast<int>(i) ) );
		}
	}
}
//---------------------------------------------------------

// Training detectors module
//---------------------------------------------------------

//---------------------------------------------------------