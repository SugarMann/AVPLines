#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include <main.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

//Define to write the markpoint dataset
//#define MAKE_TRAINING_SET

//Define to trace debug
//#define DEBUG_LABEL_TRACE

//Define to extract features
//#define EXTRACT_FEATURES

//Define to train detectors
#define TRAIN_DETECTORS

//Define to predict how works detectors
#define PREDICT_IMAGES

//Define to trace times of processing
#define TIME_PROCESS

// Some colors to draw with
enum { RED, ORANGE, YELLOW, GREEN };
static cv::Scalar colors[] =
{
	cv::Scalar(0, 0, 255),
	cv::Scalar(0, 128, 255),
	cv::Scalar(0, 255, 255),
	cv::Scalar(0, 255, 0),
};

int main( int argc, const char* argv[] )
{
	//Variables
	std::vector< std::string > l_file_json_names;			//Names of json files where it is the markpoints information
	std::vector< std::string > l_file_bmp_names;			//Names of image files
	std::list< CMarkPoint > l_mark_point_list;				//Mark points list
	std::list< cv::Mat > l_negative_samples_list;			//Negative samples list
	std::list< cv::Mat > l_detector_lst;					//Detector list for each direction slot parking features and the last one is for negative samples
	std::string l_model_path;
	uint8_t l_width = 64, l_height = 64;

	CFile l_file_object;

	if ( argc < 2 || argc > 3 ) // argc must be 2 for correct execution
	{
		std::cout << "\n--> Error:\nUsage: " << argv[0] << " <Dataset_path> <Model_file_path>\n"; // We assume that argv[0] is the program name
		return 1;
	}
	else
	{
		//Set the path in CFile object
		l_file_object.setPath( argv[1] );
		l_model_path = argv[2];

		#ifdef MAKE_TRAINING_SET
		writePositiveTrainingSet( l_file_json_names, l_file_bmp_names, l_mark_point_list, l_file_object, l_width, l_height );
		writeNegativeTrainingSet( l_file_bmp_names, l_negative_samples_list, l_file_object, l_width, l_height ); 
		#endif // MAKE_TRAINING_SET

		#ifdef EXTRACT_FEATURES
		featuresExtractor( l_file_object, l_file_bmp_names, l_detector_lst, l_width, l_height );
		#endif // EXTRACT_FEATURES

		#ifdef TRAIN_DETECTORS
		trainDetectors( l_file_object, l_model_path);
		#endif // TRAIN_DETECTORS

		#ifdef PREDICT_IMAGES
		predictImages( l_model_path, l_file_object, l_width, l_height);
		#endif // PREDICT_IMAGES

	}

	return 0;
}

// Write training set module
//---------------------------------------------------------
void writeNegativeTrainingSet( std::vector< std::string >& f_file_bmp_names, std::list< cv::Mat >& f_negative_samples_list, 
	CFile& f_file_object, uint8_t f_width, uint8_t f_height)
{
	//Local variables
	CFile l_file_object = f_file_object;
	
	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Writing negative TrainingSet:" << std::endl;

	l_file_object.setPath(l_file_object.getPath() + "\\negative_samples\\negView");
	l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);
	l_file_object.readBmp(f_file_bmp_names, f_negative_samples_list);
	//Make the training negative set og images
	l_file_object.makeNegativeTrainingSet(f_negative_samples_list, f_width, f_height);

}

void writePositiveTrainingSet( std::vector< std::string >& f_file_json_names, std::vector< std::string >& f_file_bmp_names,
	std::list< CMarkPoint >& f_mark_point_list, CFile& f_file_object, uint8_t f_width, uint8_t f_height )
{
	//Local variables
	CFile l_file_object = f_file_object;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Writing postive TrainingSet:" << std::endl;

	l_file_object.setPath(l_file_object.getPath() + "\\positive_samples");
	//Make a vector with all json file names in "positive samples"
	l_file_object.fileNamesByExtension( "json", f_file_json_names );
	//Make a vector with all bmp file names in "positive samples"
	l_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );

	//Make a JSON MarkPoints object list, where each object have all MarkPoints in one image
	l_file_object.readJson( f_file_json_names, f_mark_point_list );
	f_file_json_names.clear();
	//Set related image for every MarkPoint object
	l_file_object.readBmp( f_file_bmp_names, f_mark_point_list );
	f_file_bmp_names.clear();
	//Make the training set images with MarkPoint list information
	l_file_object.makePositiveTrainingSet(f_mark_point_list, f_width, f_height );
}
//---------------------------------------------------------

// Extract features module
//---------------------------------------------------------
void featuresExtractor( CFile& f_file_object, std::vector<std::string>& f_file_bmp_names,
	std::list<cv::Mat>& f_detector_lst, uint8_t f_width, uint8_t f_height )
{
	//Variables
	cv::Mat l_mat;
	CFile l_file_object;
	std::vector<cv::Mat> l_positive_gradient_vec, l_negative_gradient_vec;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Extracting features step:" << std::endl;

	//----------------------------------------
	//Features extractor for positive samples
	//----------------------------------------
	std::cout << "	--> Extracting positive features." << std::endl;

	//4 differents detector are trained (right, up, left, down markpoints directions)
	for ( int i = 0; i <= 3; i++ )
	{
		//Prepare different paths for each detector
		l_file_object.setPath( f_file_object.getPath() + "\\positive_samples\\dataset\\" + std::to_string(i) );

		//Read images and save them in a markpoint list 
		l_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );
		std::list<CMarkPoint> l_mark_point_list( static_cast<int>( f_file_bmp_names.size() ), CMarkPoint() );
		l_file_object.readBmp( f_file_bmp_names, l_mark_point_list );

		//Extract features
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
			computeHOG( l_mat, l_positive_gradient_vec, f_width, f_height);
			
		}
		//Convert data to Machine Learning format
		convertToML( l_positive_gradient_vec, l_mat );
		f_detector_lst.push_back( l_mat );
		
		//Save in csv
		std::string l_filename = l_file_object.getPath() + "\\features\\features_" + std::to_string(i) + ".csv";
		l_file_object.writeCSV(l_filename, l_mat);

		//Clear variables
		l_positive_gradient_vec.clear();
		f_file_bmp_names.clear();
		l_mark_point_list.clear();

	}

	//----------------------------------------
	//Features extractor for negative samples
	//----------------------------------------
	std::cout << "	--> Extracting negative features." << std::endl;

	// Prepare path
	l_file_object.setPath(f_file_object.getPath() + "\\negative_samples\\dataset\\0" );

	//Read images and save them in a negative list iamges
	l_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );
	std::list<cv::Mat> l_negative_images_lst;
	l_file_object.readBmp(f_file_bmp_names, l_negative_images_lst);

	//Extract Features
	for (cv::Mat l_negative_img : l_negative_images_lst)
	{
		//Image to grayscale
		cv::cvtColor(l_negative_img, l_negative_img, cv::COLOR_RGB2GRAY);

	#ifdef DEBUG_LABEL_TRACE
		cv::imshow("Gray image", l_negative_img);
		cv::waitKey(0);
	#endif // DEBUG_LABEL_TRACE

		//1st Feature - Stretch the grayscale histogram
		stretchHistogram(l_negative_img);

	#ifdef DEBUG_LABEL_TRACE
		cv::imshow("Stretched gray image", l_negative_img);
		cv::waitKey(0);
	#endif // DEBUG_LABEL_TRACE

		//2nd Feature - Gradient Magnitude
		computeHOG(l_negative_img, l_negative_gradient_vec, f_width, f_height);
	}
	//Convert data to Machine Learning format
	convertToML(l_negative_gradient_vec, l_mat);
	f_detector_lst.push_back(l_mat);

	//Save in csv file
	std::string l_filename = l_file_object.getPath() + "\\features\\features_0.csv";
	l_file_object.writeCSV(l_filename, l_mat);

	//Clear variables
	l_positive_gradient_vec.clear();
	f_file_bmp_names.clear();
	l_negative_images_lst.clear();
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

void computeHOG( const cv::Mat& f_image, std::vector<cv::Mat>& f_gradient_lst, uint8_t f_width, uint8_t f_height )
{
	//Variables
	std::vector<float> l_descriptors;

	//Histogram Of Gradients descriptor
	//Link -> https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

	cv::HOGDescriptor hog(
		cv::Size(f_width, f_height),							//winSize
		cv::Size(f_width/2, f_height/2),							//blockSize
		cv::Size(f_width/4, f_height/4),						//blockStride
		cv::Size(f_width/4, f_height/4),						//cellSize
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

/* @brief Convert training/testing set to be used by OpenCV Machine Learning algorithms. 
          Transposition of samples are made if needed.
   @param TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
   @return Void.
 */
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
void trainDetectors( CFile& f_file_object, const std::string& f_model_path )
{
	//Variables
	CFile l_file_object;
	cv::Mat l_positive_gradients, l_negative_gradients;
	std::vector<cv::Mat> l_gradients; // pos - direction: 0 - right, 1 - top, 2 - left, 3 - bottom.
	std::vector<cv::Mat> l_labels;
	cv::Mat l_error_mat, l_test_idx_mat;
	float l_error;

	//Train detectors. 4 differents detector are trained (right, up, left, down markpoints directions).
	//---------------------------------------------------------
	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Train step:" << std::endl;
	std::cout << "    --> Reading datasets" << std::endl;

	//Read each feature csv file
	//---------------------------------------------------------
	for (int i = 0; i <= 3; i++)
	{
		//Prepare different paths for each detector
		l_file_object.setPath(f_file_object.getPath() + "\\positive_samples\\dataset\\" + std::to_string(i) +
			"\\features\\features_" + std::to_string(i) + ".csv");
		l_file_object.readCSV(l_positive_gradients);
		l_gradients.push_back(l_positive_gradients);
		
		//Prepare labels
		cv::Mat l_positive_label( cv::Size( 1,l_positive_gradients.rows ), CV_32SC1 );
		l_positive_label = 1; // 1 for positives
		l_labels.push_back(l_positive_label);
	}
	
	//Read csv file for negative features
	l_file_object.setPath(f_file_object.getPath() + "\\negative_samples\\dataset\\0\\features\\features_0.csv");
	l_file_object.readCSV(l_negative_gradients);
	//Prepare negative labels
	cv::Mat l_negative_label( cv::Size( 1, l_negative_gradients.rows ), CV_32SC1 );
	l_negative_label = 0; // 0 for negatives

	//Add to gradients the negative part
	for (cv::Mat& l_mat : l_gradients)
	{
		l_mat.push_back(l_negative_gradients);
		cv::flip(l_mat, l_mat, 0);
	}

	//Add to labels the negative part
	for (cv::Mat& l_mat : l_labels)
	{
		l_mat.push_back(l_negative_label);
		cv::flip(l_mat, l_mat, 0);
	}

	//Boost default parameters:
	//	- boostType = Boost::REAL;
	//	- weakCount = 100;
	//	- weightTrimRate = 0.95;
	//	- CVFolds = 0;
	//	- maxDepth = 1;

	//Prepare boost object parameters
	cv::Ptr<cv::ml::Boost> boost_ = cv::ml::Boost::create();
	boost_->setBoostType(cv::ml::Boost::LOGIT);
	boost_->setWeakCount(100);
	boost_->setWeightTrimRate(1.0);
	boost_->setMaxDepth(10);
	boost_->setUseSurrogates(false);

	// Same steps for each detector
	for (int i = 0; i <= 3; i++)
	{
		//Prepare train data
		cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(l_gradients[i], cv::ml::ROW_SAMPLE, l_labels[i]);

		 //Select percentage for the training
		data->setTrainTestSplitRatio(0.8, true);
		std::cout << "    --> Number of train samples: " << data->getNTrainSamples() << std::endl;
		std::cout << "    --> Number of test samples: " << data->getNTestSamples() << std::endl;

		//Write csv train data test
		//l_test_idx_mat = data->getTestResponses();
		//l_file_object.writeCSV( f_model_path + "\\resources\\" + "Test_" + std::to_string(i) + ".csv", l_test_idx_mat);
	
		//Train data
		boost_->train(data);
	
		if (boost_->isTrained())
			std::cout << "    --> Model trained" << std::endl;

		//Calculate error over the split test data
		l_error = boost_->calcError(data, true, l_error_mat);
		//Write csv error model
		//l_file_object.writeCSV(f_model_path + "\\resources\\" + "ErrorModel_" + std::to_string(i) + ".csv", l_error_mat);
		std::cout << "    --> Error percentage over test: " << l_error << std::endl;

		//Calculate error over the split train data
		l_error = boost_->calcError(data, false, l_error_mat);
		std::cout << "    --> Error percentage over train: " << l_error << std::endl;

		boost_->save(f_model_path + "\\resources\\Model_" + std::to_string(i) + ".yml" );
		std::cout << "    --> Model saved" << std::endl;

		//Clear boost object
		boost_->clear();
	}


}
//---------------------------------------------------------

// Predict module
//---------------------------------------------------------
void predictImages( const std::string& f_model_path, CFile& f_file_object, uint8_t f_width, uint8_t f_height )
{
	//Variables
	std::vector < cv::Ptr <cv::ml::Boost> > l_boost_detectors;
	CFile l_file_object;
	std::list< cv::Mat > l_image_list;
	std::vector< std::string > l_file_names;
	cv::Mat l_crop_image, l_painted_image;
	std::vector < cv::Mat > l_gradient;
	float l_response_0, l_response_1, l_response_2, l_response_3;
	cv::Rect2d l_rect_selected;
	bool l_is_roi_selected = false;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Test step:" << std::endl;

	//Make a list of images for test
	l_file_object.setPath(f_file_object.getPath() + "\\positive_samples");
	l_file_object.fileNamesByExtension("bmp", l_file_names);
	l_file_object.readBmp(l_file_names, l_image_list);

	//Read and load ML models
	for (uint8_t i = 0; i <= 3; i++)
	{
		l_boost_detectors.push_back( cv::ml::StatModel::load<cv::ml::Boost>(f_model_path + "\\resources\\Model_" + std::to_string(i) + ".yml") );
	}

	for (cv::Mat l_image : l_image_list)
	{

		#ifdef TIME_PROCESS
		//Process time counter initialize
		int64_t l_time1, l_time2;
		l_time1 = cv::getTickCount();
		#endif	// TIME_PROCESS

		//Show original image
		//cv::imshow("Original Image", l_image);
		//cv::waitKey(0);
		l_image.copyTo( l_painted_image );

		//Select the roi that we don't need to process ( black box car in surround images )
		if ( !l_is_roi_selected)
		{
			l_rect_selected = cv::selectROI(l_image);
			l_is_roi_selected = true;
		}

		//Create grid over the original image to get the differents square pieces for process and predict
		for (uint16_t y = 0; y < l_image.rows; y += f_height/1.5)
		{
			for (uint16_t x = 0; x < l_image.cols; x += f_width/1.5)
			{
				//Size restriction
				if ( x + f_width > l_image.cols || y + f_height > l_image.rows )
					continue;

				//Car restrictions
				if ( x >= l_rect_selected.x &&
					y >= l_rect_selected.y &&
					x <= l_rect_selected.width + l_rect_selected.x &&
					y <= l_rect_selected.height + l_rect_selected.y)
					continue;
				if ( x + f_width >= static_cast<uint16_t> ( l_rect_selected.x ) &&
					 y + f_height >= static_cast<uint16_t> ( l_rect_selected.y ) &&
					 x + f_width <= static_cast<uint16_t> ( l_rect_selected.width + l_rect_selected.x ) &&
					 y + f_height <= static_cast<uint16_t> ( l_rect_selected.height + l_rect_selected.y ) )
					continue;
				if (x >= l_rect_selected.x && 
					y + f_height >= static_cast<uint16_t> ( l_rect_selected.y ) &&
					x <= l_rect_selected.width + l_rect_selected.x && 
					y + f_height <= static_cast<uint16_t> ( l_rect_selected.height + l_rect_selected.y ) )
					continue;
				if (x + f_width >= static_cast<uint16_t> ( l_rect_selected.x ) &&
					y >= l_rect_selected.y &&
					x + f_width <= static_cast<uint16_t> ( l_rect_selected.width + l_rect_selected.x ) &&
					y <= l_rect_selected.height + l_rect_selected.y)
					continue;

				//Crop image
				cv::Rect l_rect = cv::Rect(x, y, f_width, f_height);
				l_crop_image = l_image(l_rect);

				//Convert to grayscale
				cv::cvtColor(l_crop_image, l_crop_image, cv::COLOR_RGB2GRAY);

				//cv::imshow("Gray Image", l_crop_image);
				//cv::waitKey(0);

				//1st feature, improve contrast stretching grayscale histogram
				stretchHistogram(l_crop_image);

				//cv::imshow("Stretched histogram Image", l_crop_image);
				//cv::waitKey(0);

				//2nd feature, calculate hog for that piece of image
				computeHOG(l_crop_image, l_gradient, f_width, f_height);

				//Predict with 4 detectors
				l_response_0 = l_boost_detectors[0]->predict(l_gradient.back());
				l_response_1 = l_boost_detectors[1]->predict(l_gradient.back());
				l_response_2 = l_boost_detectors[2]->predict(l_gradient.back());
				l_response_3 = l_boost_detectors[3]->predict(l_gradient.back());

				//Paint results
				if (l_response_0 == 1)
				{
					cv::rectangle(l_painted_image, l_rect, colors[RED]);
				}
				else if (l_response_1 == 1)
				{
					cv::rectangle(l_painted_image, l_rect, colors[ORANGE]);
				}
				else if (l_response_2 == 1)
				{
					cv::rectangle(l_painted_image, l_rect, colors[YELLOW]);
				}
				else if (l_response_3 == 1)
				{
					cv::rectangle(l_painted_image, l_rect, colors[GREEN]);
				}
				else
				{
					cv::rectangle(l_painted_image, l_rect, cv::Scalar(255,255,255));
				}

				//Clean variables
				if (!l_gradient.empty())
				{
					l_gradient.pop_back();
				}

			}
		}

		// Text info
		cv::putText(l_painted_image, "RED -> Right", cv::Point2d(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[RED], 1, cv::LINE_AA);
		cv::putText(l_painted_image, "ORANGE -> Up", cv::Point2d(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[ORANGE], 1, cv::LINE_AA);
		cv::putText(l_painted_image, "YELLOW -> Left", cv::Point2d(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[YELLOW], 1, cv::LINE_AA);
		cv::putText(l_painted_image, "GREEN -> Down", cv::Point2d(10, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[GREEN], 1, cv::LINE_AA);

		cv::imshow("Results", l_painted_image);
		cv::waitKey(0);

		#ifdef TIME_PROCESS
			//Obtaining time of process
			l_time2 = cv::getTickCount();
			l_time2 = l_time2 - l_time1;
			//Convert to miliseconds
			l_time2 = 1000 * l_time2 / cv::getTickFrequency();
			l_time1 = 1000 * l_time2;

			std::cout << "---------------------------------------------------------" << std::endl;
			std::cout << "\n\n--> Time of process:" << l_time2 << " ms" << std::endl;
		#endif // TIME_PROCESS

	}
}
//---------------------------------------------------------