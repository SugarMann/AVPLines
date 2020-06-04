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
//#define TRAIN_DETECTORS

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

//Global variables
int m_number_detectors = 3; //nº of detectors - 1
float m_confidence = 0.5f;
bool hogs_sel = true;

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

	//4 differents detector are trained (right, up, left, down markpoints directions)
	for ( int i = 0; i <= 3; i++ )
	{
		std::cout << "	--> Reading positive dataset number " << std::to_string(i) << "." << std::endl;

		//Prepare different paths for each detector
		l_file_object.setPath( f_file_object.getPath() + "\\positive_samples\\dataset\\" + std::to_string(i) );

		//Read images and save them in a markpoint list 
		l_file_object.fileNamesByExtension( "bmp", f_file_bmp_names );
		std::list<CMarkPoint> l_mark_point_list( static_cast<int>( f_file_bmp_names.size() ), CMarkPoint() );
		l_file_object.readBmp( f_file_bmp_names, l_mark_point_list );

		std::cout << "	--> Extracting positive features for dataset number " << std::to_string(i) << "." << std::endl;
		//Extract features
		for ( CMarkPoint l_mark_point: l_mark_point_list )
		{
			//Image to grayscale
			cv::cvtColor(l_mark_point.getImage(), l_mat, cv::COLOR_RGB2GRAY);

			#ifdef DEBUG_LABEL_TRACE
			cv::imshow( "Gray image", l_mat );
			cv::waitKey(0);
			#endif // DEBUG_LABEL_TRACE

			//Another way to remove noise
			cv::GaussianBlur(l_mat, l_mat, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
			
			#ifdef DEBUG_LABEL_TRACE
			cv::imshow("Cleaned image", l_mat);
			cv::waitKey(0);
			#endif // DEBUG_LABEL_TRACE

			//Stretch the grayscale histogram
			stretchHistogram(l_mat);

			#ifdef DEBUG_LABEL_TRACE
			cv::imshow( "Stretched gray image", l_mat );
			cv::waitKey(0);
			#endif // DEBUG_LABEL_TRACE

			//Gradient Magnitude and Orientation
			if (hogs_sel)
			{
				computeHOGs(l_mat, l_positive_gradient_vec, f_width, f_height);
			}
			else {
				computeHOG(l_mat, l_positive_gradient_vec, f_width, f_height);
			}
			
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

		//Another way to remove noise
		cv::GaussianBlur(l_negative_img, l_negative_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

		//Stretch the grayscale histogram
		stretchHistogram(l_negative_img);

	#ifdef DEBUG_LABEL_TRACE
		cv::imshow("Stretched gray image", l_negative_img);
		cv::waitKey(0);
	#endif // DEBUG_LABEL_TRACE

		//Gradient Magnitude and Orientation
		if (hogs_sel)
		{
			computeHOGs(l_negative_img, l_negative_gradient_vec, f_width, f_height);
		}
		else {
			computeHOG(l_negative_img, l_negative_gradient_vec, f_width, f_height);
		}

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

void computeHOGs(const cv::Mat& f_image, std::vector<cv::Mat>& f_gradient_lst, uint8_t f_width, uint8_t f_height)
{
	//Variables
	std::vector<float> l_descriptors, l_descriptors_up, l_descriptors_down;
	cv::Mat l_down_image, l_up_image;

	//Histogram Of Gradients descriptor
	//Link -> https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

	//Down-scale and upscale the image to filter out the noise
	cv::pyrDown(f_image, l_down_image, cv::Size(f_image.cols / 2, f_image.rows / 2));
	cv::pyrUp(f_image, l_up_image, cv::Size(f_image.cols * 2, f_image.rows * 2));

	cv::HOGDescriptor hog(
		cv::Size(f_width, f_height),				//winSize
		cv::Size(f_width / 2, f_height / 2),		//blockSize
		cv::Size(f_width / 4, f_height / 4),		//blockStride
		cv::Size(f_width / 4, f_height / 4),		//cellSize
		9,											//nbins
		1,											//derivAper
		-1,											//winSigma
		cv::HOGDescriptor::HistogramNormType(0),	//histogramNormType
		0.2,										//L2HysThersh
		true,										//gammal correction
		64,											//nlevels
		true										//use signed gradients
	);

	cv::HOGDescriptor hog_down(
		cv::Size(l_down_image.cols, l_down_image.rows),				//winSize
		cv::Size(l_down_image.cols / 2, l_down_image.rows / 2),		//blockSize
		cv::Size(l_down_image.cols / 4, l_down_image.rows / 4),		//blockStride
		cv::Size(l_down_image.cols / 4, l_down_image.rows / 4),		//cellSize
		9,											//nbins
		1,											//derivAper
		-1,											//winSigma
		cv::HOGDescriptor::HistogramNormType(0),	//histogramNormType
		0.2,										//L2HysThersh
		true,										//gammal correction
		64,											//nlevels
		true										//use signed gradients
	);

	cv::HOGDescriptor hog_up(
		cv::Size(l_up_image.cols, l_up_image.rows),				//winSize
		cv::Size(l_up_image.cols / 2, l_up_image.rows / 2),		//blockSize
		cv::Size(l_up_image.cols / 4, l_up_image.rows / 4),		//blockStride
		cv::Size(l_up_image.cols / 4, l_up_image.rows / 4),		//cellSize
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
	hog.compute(f_image, l_descriptors);
	//Calculate descriptor for pyramid up image
	hog_up.compute(l_up_image, l_descriptors_up);
	//Reserve and store descriptors
	l_descriptors.reserve(l_descriptors.size() * 3);
	l_descriptors.insert(l_descriptors.end(), l_descriptors_up.begin(), l_descriptors_up.end());
	//Calculate descriptor for pyramid down image
	hog_down.compute(l_down_image, l_descriptors_down);
	l_descriptors.insert(l_descriptors.end(), l_descriptors_down.begin(), l_descriptors_down.end());
	//Push back into gradient list -> 3 descriptors (normal, upImage and downImage)
	f_gradient_lst.push_back(cv::Mat(l_descriptors).clone());
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
	for (int i = 0; i <= m_number_detectors; i++)
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

	//Prepare boost object parameters
	cv::Ptr< cv::ml::Boost > boost_ = cv::ml::Boost::create();
	boost_->setBoostType(cv::ml::Boost::GENTLE);
	boost_->setWeakCount(100);
	boost_->setWeightTrimRate(0.99);
	boost_->setMaxDepth(6);
	boost_->setUseSurrogates(false);

	//Prepare SVM object parameters
	cv::Ptr< cv::ml::SVM > svm_ = cv::ml::SVM::create();
	svm_->setType(cv::ml::SVM::C_SVC);
	svm_->setKernel(cv::ml::SVM::RBF);
	svm_->setC(2.5);
	svm_->setGamma(0.50625);

	std::cout << "    --> Number of detectors to train: " << m_number_detectors+1 << std::endl;
	// Same steps for each detector
	for (int i = 0; i <= m_number_detectors; i++)
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

		std::cout << "    --> Training models" << std::endl;

		//Boost train data
		try {
		boost_->train(data);

		if (boost_->isTrained())
			std::cout << "    --> Boost model trained" << std::endl;
		}
		catch (const std::exception & e) { // referencia a base de un objeto polimorfo
			std::cout << e.what();
		}

		//SVM train data
		try {
			//svm_->trainAuto(data);
			svm_->trainAuto(data);
		}
		catch (const std::exception & e) { // referencia a base de un objeto polimorfo
			std::cout << e.what();
		}

		if (svm_->isTrained())
			std::cout << "    --> SVM model trained" << std::endl;

		//Calculate error over the split test data  for Boost model
		l_error = boost_->calcError(data, true, l_error_mat);
		std::cout << "    --> Boost error percentage over test: " << l_error << std::endl;
		//Write csv error model
		//l_file_object.writeCSV(f_model_path + "\\resources\\" + "ErrorModel_" + std::to_string(i) + ".csv", l_error_mat);

		//Calculate error over the split test data for SVM model
		l_error = svm_->calcError(data, true, l_error_mat);
		std::cout << "    --> SVM error percentage over test: " << l_error << std::endl;

		//Calculate error over the split train data  for Boost model
		l_error = boost_->calcError(data, false, l_error_mat);
		std::cout << "    --> Boost error percentage over train: " << l_error << std::endl;

		//Calculate error over the split train data  for SVM model
		l_error = svm_->calcError(data, false, l_error_mat);
		std::cout << "    --> SVM error percentage over train: " << l_error << std::endl;

		svm_->save( f_model_path + "\\resources\\SVM_MarkPoint_Model_" + std::to_string(i) + ".yml" );
		boost_->save( f_model_path + "\\resources\\Boost_MarkPoint_Model_" + std::to_string(i) + ".yml" );
		std::cout << "    --> Models saved" << std::endl;

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
	std::vector < cv::Ptr <cv::ml::SVM> > l_svm_detectors;
	CFile l_file_object;
	std::list< cv::Mat > l_image_list;
	std::vector< std::string > l_file_names;
	cv::Mat l_crop_image, l_painted_boost_img, l_painted_svm_img;
	std::vector < cv::Mat > l_gradient;
	std::vector < float > l_boost_responses, l_svm_responses;
	cv::Rect2d l_rect_selected;
	bool l_is_roi_selected = false;
	float l_distance;
	std::vector < float > l_confidence_vector;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Test step:" << std::endl;

	//Make a list of images for test
	l_file_object.setPath(f_file_object.getPath() + "\\..\\TestSet\\ForMarkingPoints"); //\\positive_samples
	l_file_object.fileNamesByExtension("bmp", l_file_names);
	l_file_object.readBmp(l_file_names, l_image_list);

	//Read and load ML boost models
	for (uint8_t i = 0; i <= m_number_detectors; i++)
	{
		l_boost_detectors.push_back( cv::ml::StatModel::load<cv::ml::Boost>(f_model_path + "\\resources\\Boost_MarkPoint_Model_" + std::to_string(i) + ".yml") );
	}

	//Read and load ML SVM models
	for (uint8_t i = 0; i <= m_number_detectors; i++)
	{
		l_svm_detectors.push_back(cv::ml::StatModel::load<cv::ml::SVM>(f_model_path + "\\resources\\SVM_MarkPoint_Model_" + std::to_string(i) + ".yml"));
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
		l_image.copyTo( l_painted_boost_img );
		l_image.copyTo(l_painted_svm_img);

		//Select the roi that we don't need to process ( black box car in surround images )
		if ( !l_is_roi_selected)
		{
			l_rect_selected = cv::selectROI(l_image);
			l_is_roi_selected = true;
		}

		//Create grid over the original image to get the differents square pieces for process and predict
		for (uint16_t y = 0; y < l_image.rows; y += f_height/2)
		{
			for (uint16_t x = 0; x < l_image.cols; x += f_width/2)
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

				//Down-scale and upscale the image to filter out the noise
				cv::GaussianBlur(l_crop_image, l_crop_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

				//cv::imshow("Filtering noise", l_crop_image);
				//cv::waitKey(0);

				//Improve contrast stretching grayscale histogram
				stretchHistogram(l_crop_image);

				//cv::imshow("Stretched histogram Image", l_crop_image);
				//cv::waitKey(0);

				//Calculate hog for that piece of image
				if (hogs_sel)
				{
					computeHOGs(l_crop_image, l_gradient, f_width, f_height);
				}
				else {
					computeHOG(l_crop_image, l_gradient, f_width, f_height);
				}

				//Traspose features matrix
				l_gradient.back() = l_gradient.back().t();

				//Predict with Boost detectors 
				for (uint8_t i = 0; i <= m_number_detectors; i++) 
				{
					l_boost_responses.push_back( l_boost_detectors[i]->predict(l_gradient.back()) );
				}

				//Predict with SVM detectors
				for (uint8_t i = 0; i <= m_number_detectors; i++)
				{
					l_svm_responses.push_back( l_svm_detectors[i]->predict(l_gradient.back()) );
					if (l_svm_responses[i] == 1)
					{
						l_distance = distanceSample(l_gradient.back(), l_svm_detectors[i]);
						if (l_distance != 0)
							l_confidence_vector.push_back( 1.0 / ( 1.0 + std::exp(-l_distance) ) );
					}
					else {
						l_confidence_vector.push_back(0);
					}
					
				}

				for (uint8_t i = 0; i <= m_number_detectors; i++)
				{
					if (l_confidence_vector[i] >= m_confidence)
					{
						switch (i)
						{
						case 0:
							cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
							break;
						case 1:
							cv::rectangle(l_painted_svm_img, l_rect, colors[ORANGE]);
							break;
						case 2:
							cv::rectangle(l_painted_svm_img, l_rect, colors[YELLOW]);
							break;
						case 3:
							cv::rectangle(l_painted_svm_img, l_rect, colors[GREEN]);
							break;
						}
					}
				}


				//Paint boost results
				switch (l_boost_responses.size())
				{
				case 1:
					if (l_boost_responses[0] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[RED]);
					}
					else
					{
						//cv::rectangle(l_painted_boost_img, l_rect, cv::Scalar(255,255,255));
					}
					break;
				case 2:
					if (l_boost_responses[0] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[RED]);
					}
					else if (l_boost_responses[1] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[ORANGE]);
					}
					else
					{
						//cv::rectangle(l_painted_boost_img, l_rect, cv::Scalar(255,255,255));
					}
					break;
				case 3:
					if (l_boost_responses[0] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[RED]);
					}
					else if (l_boost_responses[1] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[ORANGE]);
					}
					else if (l_boost_responses[2] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[YELLOW]);
					}
					else
					{
						//cv::rectangle(l_painted_boost_img, l_rect, cv::Scalar(255,255,255));
					}
					break;
				case 4:
					if (l_boost_responses[0] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[RED]);
					}
					else if (l_boost_responses[1] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[ORANGE]);
					}
					else if (l_boost_responses[2] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[YELLOW]);
					}
					else if (l_boost_responses[3] == 1)
					{
						cv::rectangle(l_painted_boost_img, l_rect, colors[GREEN]);
					}
					else
					{
						//cv::rectangle(l_painted_boost_img, l_rect, cv::Scalar(255,255,255));
					}
					break;
				}

				//Paint svm results

				//switch (l_svm_responses.size()) 
				//{
				//case 1:
				//	if (l_svm_responses[0] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
				//	}
				//	else
				//	{
				//		//cv::rectangle(l_painted_svm_img, l_rect, cv::Scalar(255, 255, 255));
				//	}
				//	break;
				//case 2:
				//	if (l_svm_responses[0] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
				//	}
				//	else if (l_svm_responses[1] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[ORANGE]);
				//	}
				//	else
				//	{
				//		//cv::rectangle(l_painted_svm_img, l_rect, cv::Scalar(255, 255, 255));
				//	}
				//	break;
				//case 3:
				//	if (l_svm_responses[0] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
				//	}
				//	else if (l_svm_responses[1] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[ORANGE]);
				//	}
				//	else if (l_svm_responses[2] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[YELLOW]);
				//	}
				//	else
				//	{
				//		//cv::rectangle(l_painted_svm_img, l_rect, cv::Scalar(255, 255, 255));
				//	}
				//	break;
				//case 4:
				//	if (l_svm_responses[0] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
				//	}
				//	else if (l_svm_responses[1] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[ORANGE]);
				//	}
				//	else if (l_svm_responses[2] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[YELLOW]);
				//	}
				//	else if (l_svm_responses[3] == 1)
				//	{
				//		cv::rectangle(l_painted_svm_img, l_rect, colors[GREEN]);
				//	}
				//	else
				//	{
				//		//cv::rectangle(l_painted_svm_img, l_rect, cv::Scalar(255, 255, 255));
				//	}
				//	break;

				//}

				//Clean variables
				if (!l_gradient.empty())
				{
					l_gradient.pop_back(); 
					l_boost_responses.clear();
					l_svm_responses.clear();
					l_confidence_vector.clear();
				}

			}
		}

		// Text info
		cv::putText(l_painted_svm_img, "RED -> Right", cv::Point2d(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[RED], 1, cv::LINE_AA);
		cv::putText(l_painted_svm_img, "ORANGE -> Up", cv::Point2d(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[ORANGE], 1, cv::LINE_AA);
		cv::putText(l_painted_svm_img, "YELLOW -> Left", cv::Point2d(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[YELLOW], 1, cv::LINE_AA);
		cv::putText(l_painted_svm_img, "GREEN -> Down", cv::Point2d(10, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[GREEN], 1, cv::LINE_AA);

		cv::putText(l_painted_boost_img, "RED -> Right", cv::Point2d(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[RED], 1, cv::LINE_AA);
		cv::putText(l_painted_boost_img, "ORANGE -> Up", cv::Point2d(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[ORANGE], 1, cv::LINE_AA);
		cv::putText(l_painted_boost_img, "YELLOW -> Left", cv::Point2d(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[YELLOW], 1, cv::LINE_AA);
		cv::putText(l_painted_boost_img, "GREEN -> Down", cv::Point2d(10, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[GREEN], 1, cv::LINE_AA);

		cv::imshow("Boost Results", l_painted_boost_img);
		cv::imshow("SVM Results", l_painted_svm_img);
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

float distanceSample(cv::Mat& f_sample, const cv::Ptr <cv::ml::Boost>& f_boost)
{
	//Variables
	cv::Mat l_result;
	float l_distance;

	f_boost->predict(f_sample, l_result, cv::ml::StatModel::Flags::RAW_OUTPUT);
	l_distance = l_result.at<float>(0, 0);
	return l_distance;
}

float distanceSample(cv::Mat& f_sample, const cv::Ptr <cv::ml::SVM>& f_svm)
{
	//Variables
	cv::Mat l_result;
	float l_distance;

	f_svm->predict(f_sample, l_result, cv::ml::StatModel::Flags::RAW_OUTPUT);
	l_distance = l_result.at<float>(0, 0);
	return l_distance;
}
//---------------------------------------------------------