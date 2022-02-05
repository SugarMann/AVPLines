//=====================================================================================================================
// Includes
//=====================================================================================================================
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>

#include <vs/AVPLines/main.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#define PI 3.14159265

// Define reading video
#define READ_VIDEO

// Define reading images
// #define READ_IMAGES

// Define to write the markpoint dataset
//  #define MAKE_TRAINING_SET

// Define to write the markpoint dataset
//  #define MAKE_TEST_SET

// Define to extract features
//  #define EXTRACT_FEATURES

// Define to train detectors
#define TRAIN_DETECTORS

// Define to predict how works detectors
// #define PREDICT_IMAGES

// Define to trace times of processing
#define TIME_PROCESS

// Define to trace debug
// #define DEBUG_VISUAL_TRACE

// Define shell trace
#define DEBUG_PROMPT_TRACE

// Define log predictions
// #define LOG_RESULTS

// Some colors to draw with
enum
{
	RED,
	ORANGE,
	YELLOW,
	GREEN
};
static cv::Scalar colors[] =
	{
		cv::Scalar(0, 0, 255),
		cv::Scalar(0, 128, 255),
		cv::Scalar(0, 255, 255),
		cv::Scalar(0, 255, 0),
};

// Global variables
int m_number_detectors = 3; // number of detectors - 1 (minus one)
int m_number_divisions = 6; // number of division to process in the image
float m_group_rectangles_scale = 0.2f;
// float m_confidence = 0.5f;
bool hogs_sel = true;
float m_max_dist_slot_short = 190;
float m_min_dist_slot_short = 135;
float m_max_dist_slot_long = 395;
float m_min_dist_slot_long = 305;
int16_t m_max_slot_overlap = 20;

int main(int argc, const char *argv[])
{
	// Variables
	std::vector<std::string> l_file_json_names; // Names of json files where it is the markpoints information
	std::vector<std::string> l_file_bmp_names;	// Names of image files
	std::list<CMarkPoint> l_mark_point_list;	// Mark points list
	std::list<cv::Mat> l_negative_samples_list; // Negative samples list
	std::list<cv::Mat> l_detector_lst;			// Detector list for each direction slot parking features and the last one is for negative samples
	std::string l_model_path;
	uint8_t l_width = 64, l_height = 64;

	CFile l_file_object;

	if (argc < 2 || argc > 3) // argc must be 2 for correct execution
	{
		std::cout << "\n--> Error:\nUsage: " << argv[0] << " <Dataset_path> <Model_file_path>\n"; // We assume that argv[0] is the program name
		return 1;
	}
	else
	{
		// Set the path in CFile object
		l_file_object.setPath(argv[1]);
		l_model_path = argv[2];

#ifdef MAKE_TRAINING_SET
		writePositiveTrainingSet(l_file_json_names, l_file_bmp_names, l_mark_point_list, l_file_object, l_width, l_height);
		l_file_object.setPath(argv[1]);
		writeNegativeTrainingSet(l_file_bmp_names, l_negative_samples_list, l_file_object, l_width, l_height);
		l_file_object.setPath(argv[1]);
#endif // MAKE_TRAINING_SET

#ifdef MAKE_TEST_SET
		writeAnnotationsTestSet(l_file_json_names, l_file_bmp_names, l_mark_point_list, l_file_object, l_width, l_height);
		l_file_object.setPath(argv[1]);
#endif // MAKE_TEST_SET

#ifdef EXTRACT_FEATURES
		featuresExtractor(l_file_object, l_file_bmp_names, l_detector_lst, l_width, l_height);
#endif // EXTRACT_FEATURES

#ifdef TRAIN_DETECTORS
		trainDetectors(l_file_object, l_model_path);
#endif // TRAIN_DETECTORS

#ifdef PREDICT_IMAGES
		predictImages(l_model_path, l_file_object, l_width, l_height);
#endif // PREDICT_IMAGES
	}

	return 0;
}

// Write training set module
//---------------------------------------------------------
void writeNegativeTrainingSet(std::vector<std::string> &f_file_bmp_names, std::list<cv::Mat> &f_negative_samples_list,
							  CFile &f_file_object, uint8_t f_width, uint8_t f_height)
{
	// Local variables
	CFile l_file_object = f_file_object;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Writing negative TrainingSet:" << std::endl;

	l_file_object.setPath(l_file_object.getPath() + "\\negative_samples\\negView");
	l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);
	l_file_object.readBmp(f_file_bmp_names, f_negative_samples_list);
	// Make the training negative set og images
	l_file_object.makeNegativeTrainingSet(f_negative_samples_list, f_width, f_height);
}

void writePositiveTrainingSet(std::vector<std::string> &f_file_json_names, std::vector<std::string> &f_file_bmp_names,
							  std::list<CMarkPoint> &f_mark_point_list, CFile &f_file_object, uint8_t f_width, uint8_t f_height)
{
	// Local variables
	CFile l_file_object = f_file_object;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Writing postive TrainingSet:" << std::endl;

	l_file_object.setPath(l_file_object.getPath() + "\\positive_samples");
	// Make a vector with all json file names in "positive samples"
	l_file_object.fileNamesByExtension("json", f_file_json_names);
	// Make a vector with all bmp file names in "positive samples"
	l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);

	// Make a JSON MarkPoints object list, where each object have all MarkPoints in one image
	l_file_object.readJson(f_file_json_names, f_mark_point_list);
	f_file_json_names.clear();
	// Set related image for every MarkPoint object
	l_file_object.readBmp(f_file_bmp_names, f_mark_point_list);
	f_file_bmp_names.clear();
	// Make the training set images with MarkPoint list information
	l_file_object.makePositiveTrainingSet(f_mark_point_list, f_width, f_height);
}

void writeAnnotationsTestSet(std::vector<std::string> &f_file_json_names, std::vector<std::string> &f_file_bmp_names,
							 std::list<CMarkPoint> &f_mark_point_list, CFile &f_file_object, uint8_t f_width, uint8_t f_height)
{
	// Local variables
	CFile l_file_object = f_file_object;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Writing annotations TestSet:" << std::endl;

	l_file_object.setPath(l_file_object.getPath() + "\\..\\TestSet\\ForMarkingPoints");
	// Make a vector with all json file names in "positive samples"
	l_file_object.fileNamesByExtension("json", f_file_json_names);
	// Make a vector with all bmp file names in "positive samples"
	l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);

	// Make a JSON MarkPoints object list, where each object have all MarkPoints in one image
	l_file_object.readJson(f_file_json_names, f_mark_point_list);
	f_file_json_names.clear();
	// Set related image for every MarkPoint object
	l_file_object.readBmp(f_file_bmp_names, f_mark_point_list);
	f_file_bmp_names.clear();
	// Make the training set images with MarkPoint list information
	l_file_object.makeAnnotationsTestSet(f_mark_point_list, f_width, f_height);
}

//---------------------------------------------------------
// Extract features module
//---------------------------------------------------------
void featuresExtractor(CFile &f_file_object, std::vector<std::string> &f_file_bmp_names,
					   std::list<cv::Mat> &f_detector_lst, uint8_t f_width, uint8_t f_height)
{
	// Variables
	cv::Mat l_mat;
	CFile l_file_object;
	std::vector<cv::Mat> l_positive_gradient_vec, l_negative_gradient_vec;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Extracting features step:" << std::endl;

	//----------------------------------------
	// Features extractor for positive samples
	//----------------------------------------

	// 4 differents detector are trained (right, up, left, down markpoints directions)
	for (int i = 0; i <= 3; i++)
	{
		std::cout << "	--> Reading positive dataset number " << std::to_string(i) << "." << std::endl;

		// Prepare different paths for each detector
		l_file_object.setPath(f_file_object.getPath() + "\\positive_samples\\dataset\\" + std::to_string(i));

		// Read images and save them in a markpoint list
		l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);
		std::list<CMarkPoint> l_mark_point_list(static_cast<int>(f_file_bmp_names.size()), CMarkPoint());
		l_file_object.readBmp(f_file_bmp_names, l_mark_point_list);

		std::cout << "	--> Extracting positive features for dataset number " << std::to_string(i) << "." << std::endl;
		// Extract features
		for (CMarkPoint l_mark_point : l_mark_point_list)
		{
			// Image to grayscale
			cv::cvtColor(l_mark_point.getImage(), l_mat, cv::COLOR_RGB2GRAY);

#ifdef DEBUG_VISUAL_TRACE
			cv::imshow("Gray image", l_mat);
			cv::waitKey(0);
#endif // DEBUG_LABEL_TRACE

			// Another way to remove noise
			cv::GaussianBlur(l_mat, l_mat, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

#ifdef DEBUG_VISUAL_TRACE
			cv::imshow("Cleaned image", l_mat);
			cv::waitKey(0);
#endif // DEBUG_LABEL_TRACE

			// Stretch the grayscale histogram
			stretchHistogram(l_mat);

#ifdef DEBUG_VISUAL_TRACE
			cv::imshow("Stretched gray image", l_mat);
			cv::waitKey(0);
#endif // DEBUG_LABEL_TRACE

			// Gradient Magnitude and Orientation
			if (hogs_sel)
			{
				computeHOGs(l_mat, l_positive_gradient_vec, f_width, f_height);
			}
			else
			{
				computeHOG(l_mat, l_positive_gradient_vec, f_width, f_height);
			}
		}
		// Convert data to Machine Learning format
		convertToML(l_positive_gradient_vec, l_mat);
		f_detector_lst.push_back(l_mat);

		// Save in csv
		std::string l_filename = l_file_object.getPath() + "\\features\\features_" + std::to_string(i) + ".csv";
		l_file_object.writeCSV(l_filename, l_mat);

		// Clear variables
		l_positive_gradient_vec.clear();
		f_file_bmp_names.clear();
		l_mark_point_list.clear();
	}

	//----------------------------------------
	// Features extractor for negative samples
	//----------------------------------------
	std::cout << "	--> Extracting negative features." << std::endl;

	// Prepare path
	l_file_object.setPath(f_file_object.getPath() + "\\negative_samples\\dataset\\0");

	// Read images and save them in a negative list iamges
	l_file_object.fileNamesByExtension("bmp", f_file_bmp_names);
	std::list<cv::Mat> l_negative_images_lst;
	l_file_object.readBmp(f_file_bmp_names, l_negative_images_lst);

	// Extract Features
	for (cv::Mat l_negative_img : l_negative_images_lst)
	{
		// Image to grayscale
		cv::cvtColor(l_negative_img, l_negative_img, cv::COLOR_RGB2GRAY);

#ifdef DEBUG_VISUAL_TRACE
		cv::imshow("Gray image", l_negative_img);
		cv::waitKey(0);
#endif // DEBUG_LABEL_TRACE

		// Another way to remove noise
		cv::GaussianBlur(l_negative_img, l_negative_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

		// Stretch the grayscale histogram
		stretchHistogram(l_negative_img);

#ifdef DEBUG_VISUAL_TRACE
		cv::imshow("Stretched gray image", l_negative_img);
		cv::waitKey(0);
#endif // DEBUG_LABEL_TRACE

		// Gradient Magnitude and Orientation
		if (hogs_sel)
		{
			computeHOGs(l_negative_img, l_negative_gradient_vec, f_width, f_height);
		}
		else
		{
			computeHOG(l_negative_img, l_negative_gradient_vec, f_width, f_height);
		}
	}
	// Convert data to Machine Learning format
	convertToML(l_negative_gradient_vec, l_mat);
	f_detector_lst.push_back(l_mat);

	// Save in csv file
	std::string l_filename = l_file_object.getPath() + "\\features\\features_0.csv";
	l_file_object.writeCSV(l_filename, l_mat);

	// Clear variables
	l_positive_gradient_vec.clear();
	f_file_bmp_names.clear();
	l_negative_images_lst.clear();
}

void stretchHistogram(cv::Mat &f_image)
{
	// Variables
	double_t l_min, l_max;
	uint8_t *l_pixelPtr = (uint8_t *)f_image.data;

	// Calculate min and max values
	cv::minMaxLoc(f_image, &l_min, &l_max);

	// Calculate new intensity to stretch histogram
	// Link -> https://en.wikipedia.org/wiki/Normalization_(image_processing)/
	for (int i = 0; i < f_image.rows; i++)
	{
		for (int j = 0; j < f_image.cols; j++)
		{
			l_pixelPtr[i * f_image.cols + j] = (l_pixelPtr[i * f_image.cols + j] - l_min) * (((255 - 0) / (l_max - l_min)) + 0);
		}
	}
}

void computeHOG(const cv::Mat &f_image, std::vector<cv::Mat> &f_gradient_lst, uint8_t f_width, uint8_t f_height)
{
	// Variables
	std::vector<float> l_descriptors;

	// Histogram Of Gradients descriptor
	// Link -> https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

	cv::HOGDescriptor hog(
		cv::Size(f_width, f_height),			 // winSize
		cv::Size(f_width / 2, f_height / 2),	 // blockSize
		cv::Size(f_width / 4, f_height / 4),	 // blockStride
		cv::Size(f_width / 4, f_height / 4),	 // cellSize
		9,										 // nbins
		1,										 // derivAper
		-1,										 // winSigma
		cv::HOGDescriptor::HistogramNormType(0), // histogramNormType
		0.2,									 // L2HysThersh
		true,									 // gammal correction
		64,										 // nlevels
		true									 // use signed gradients
	);

	// Calculate magnitude and angle descriptors with hog and save them in gradients list
	hog.compute(f_image, l_descriptors);
	f_gradient_lst.push_back(cv::Mat(l_descriptors).clone());
}

void computeHOGs(const cv::Mat &f_image, std::vector<cv::Mat> &f_gradient_lst, uint8_t f_width, uint8_t f_height)
{
	// Variables
	std::vector<float> l_descriptors, l_descriptors_up, l_descriptors_down;
	cv::Mat l_down_image, l_up_image;

	// Histogram Of Gradients descriptor
	// Link -> https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

	// Down-scale and upscale the image to filter out the noise
	cv::pyrDown(f_image, l_down_image, cv::Size(f_image.cols / 2, f_image.rows / 2));
	cv::pyrUp(f_image, l_up_image, cv::Size(f_image.cols * 2, f_image.rows * 2));

	cv::HOGDescriptor hog(
		cv::Size(f_width, f_height),			 // winSize
		cv::Size(f_width / 2, f_height / 2),	 // blockSize
		cv::Size(f_width / 4, f_height / 4),	 // blockStride
		cv::Size(f_width / 4, f_height / 4),	 // cellSize
		9,										 // nbins
		1,										 // derivAper
		-1,										 // winSigma
		cv::HOGDescriptor::HistogramNormType(0), // histogramNormType
		0.2,									 // L2HysThersh
		true,									 // gammal correction
		64,										 // nlevels
		true									 // use signed gradients
	);

	cv::HOGDescriptor hog_down(
		cv::Size(l_down_image.cols, l_down_image.rows),			// winSize
		cv::Size(l_down_image.cols / 2, l_down_image.rows / 2), // blockSize
		cv::Size(l_down_image.cols / 4, l_down_image.rows / 4), // blockStride
		cv::Size(l_down_image.cols / 4, l_down_image.rows / 4), // cellSize
		9,														// nbins
		1,														// derivAper
		-1,														// winSigma
		cv::HOGDescriptor::HistogramNormType(0),				// histogramNormType
		0.2,													// L2HysThersh
		true,													// gammal correction
		64,														// nlevels
		true													// use signed gradients
	);

	cv::HOGDescriptor hog_up(
		cv::Size(l_up_image.cols, l_up_image.rows),			// winSize
		cv::Size(l_up_image.cols / 2, l_up_image.rows / 2), // blockSize
		cv::Size(l_up_image.cols / 4, l_up_image.rows / 4), // blockStride
		cv::Size(l_up_image.cols / 4, l_up_image.rows / 4), // cellSize
		9,													// nbins
		1,													// derivAper
		-1,													// winSigma
		cv::HOGDescriptor::HistogramNormType(0),			// histogramNormType
		0.2,												// L2HysThersh
		true,												// gammal correction
		64,													// nlevels
		true												// use signed gradients
	);

	// Calculate magnitude and angle descriptors with hog and save them in gradients list
	hog.compute(f_image, l_descriptors);
	// Calculate descriptor for pyramid up image
	hog_up.compute(l_up_image, l_descriptors_up);
	// Reserve and store descriptors
	l_descriptors.reserve(l_descriptors.size() * 3);
	l_descriptors.insert(l_descriptors.end(), l_descriptors_up.begin(), l_descriptors_up.end());
	// Calculate descriptor for pyramid down image
	hog_down.compute(l_down_image, l_descriptors_down);
	l_descriptors.insert(l_descriptors.end(), l_descriptors_down.begin(), l_descriptors_down.end());
	// Push back into gradient list -> 3 descriptors (normal, upImage and downImage)
	f_gradient_lst.push_back(cv::Mat(l_descriptors).clone());
}

/* @brief Convert training/testing set to be used by OpenCV Machine Learning algorithms.
		  Transposition of samples are made if needed.
   @param TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
   @return Void.
 */
void convertToML(const std::vector<cv::Mat> &f_train_samples, cv::Mat &f_trainData)
{
	// Variables
	const int l_rows = static_cast<int>(f_train_samples.size());
	const int l_cols = static_cast<int>(std::max(f_train_samples[0].cols, f_train_samples[0].rows));
	cv::Mat l_tmp(1, l_cols, CV_32FC1); //< used for transposition if needed

	f_trainData = cv::Mat(l_rows, l_cols, CV_32FC1);

	for (size_t i = 0; i < f_train_samples.size(); ++i)
	{
		CV_Assert(f_train_samples[i].cols == 1 || f_train_samples[i].rows == 1);
		if (f_train_samples[i].cols == 1)
		{
			cv::transpose(f_train_samples[i], l_tmp);
			l_tmp.copyTo(f_trainData.row(static_cast<int>(i)));
		}
		else if (f_train_samples[i].rows == 1)
		{
			f_train_samples[i].copyTo(f_trainData.row(static_cast<int>(i)));
		}
	}
}

//---------------------------------------------------------
// Training detectors module
//---------------------------------------------------------
void trainDetectors(CFile &f_file_object, const std::string &f_model_path)
{
	// Variables
	CFile l_file_object;
	cv::Mat l_positive_gradients, l_negative_gradients;
	std::vector<cv::Mat> l_gradients; // pos - direction: 0 - right, 1 - top, 2 - left, 3 - bottom.
	std::vector<cv::Mat> l_labels;
	cv::Mat l_error_mat, l_test_idx_mat;
	float l_error;

	// Train detectors. 4 differents detector are trained (right, up, left, down markpoints directions).
	//---------------------------------------------------------
	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Train step:" << std::endl;
	std::cout << "    --> Reading datasets" << std::endl;

	// Read each feature csv file
	//---------------------------------------------------------
	for (int i = 0; i <= m_number_detectors; i++)
	{
		// Prepare different paths for each detector
		l_file_object.setPath(f_file_object.getPath() + "\\positive_samples\\dataset\\" + std::to_string(i) +
							  "\\features\\features_" + std::to_string(i) + ".csv");
		l_file_object.readCSV(l_positive_gradients);
		l_gradients.push_back(l_positive_gradients);

		// Prepare labels
		cv::Mat l_positive_label(cv::Size(1, l_positive_gradients.rows), CV_32SC1);
		l_positive_label = 1; // 1 for positives
		l_labels.push_back(l_positive_label);
	}

	// Read csv file for negative features
	l_file_object.setPath(f_file_object.getPath() + "\\negative_samples\\dataset\\0\\features\\features_0.csv");
	l_file_object.readCSV(l_negative_gradients);
	// Prepare negative labels
	cv::Mat l_negative_label(cv::Size(1, l_negative_gradients.rows), CV_32SC1);
	l_negative_label = 0; // 0 for negatives

	// Add to gradients the negative part
	for (cv::Mat &l_mat : l_gradients)
	{
		l_mat.push_back(l_negative_gradients);
		cv::flip(l_mat, l_mat, 0);
	}

	// Add to labels the negative part
	for (cv::Mat &l_mat : l_labels)
	{
		l_mat.push_back(l_negative_label);
		cv::flip(l_mat, l_mat, 0);
	}

	// Prepare boost object parameters
	cv::Ptr<cv::ml::Boost> boost_ = cv::ml::Boost::create();
	boost_->setBoostType(cv::ml::Boost::GENTLE);
	boost_->setWeakCount(100);
	boost_->setWeightTrimRate(0.99);
	boost_->setMaxDepth(6);
	boost_->setUseSurrogates(false);

	// Prepare SVM object parameters
	cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::create();
	svm_->setType(cv::ml::SVM::C_SVC);
	svm_->setKernel(cv::ml::SVM::RBF);
	svm_->setC(2.5);
	svm_->setGamma(0.50625);

	std::cout << "    --> Number of detectors to train: " << m_number_detectors + 1 << std::endl;
	// Same steps for each detector
	for (int i = 0; i <= m_number_detectors; i++)
	{
		// Prepare train data
		cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(l_gradients[i], cv::ml::ROW_SAMPLE, l_labels[i]);

		// Select percentage for the training
		data->setTrainTestSplitRatio(0.9, true);
		std::cout << "    --> Number of train samples: " << data->getNTrainSamples() << std::endl;
		std::cout << "    --> Number of test samples: " << data->getNTestSamples() << std::endl;

		// Write csv train data test
		// l_test_idx_mat = data->getTestResponses();
		// l_file_object.writeCSV( f_model_path + "\\resources\\" + "Test_" + std::to_string(i) + ".csv", l_test_idx_mat);

		std::cout << "    --> Training models" << std::endl;

		// Boost train data
		try
		{
			boost_->train(data);

			if (boost_->isTrained())
			{
				std::cout << "    --> Boost model trained" << std::endl;

				// Calculate error over the split test data  for Boost model
				l_error = boost_->calcError(data, true, l_error_mat);
				std::cout << "    --> Boost error percentage over test: " << l_error << std::endl;
				// Write csv error model
				// l_file_object.writeCSV(f_model_path + "\\resources\\" + "ErrorModel_" + std::to_string(i) + ".csv", l_error_mat);

				// Calculate error over the split train data  for Boost model
				l_error = boost_->calcError(data, false, l_error_mat);
				std::cout << "    --> Boost error percentage over train: " << l_error << std::endl;

				boost_->save(f_model_path + "\\resources\\Boost_MarkPoint_Model_" + std::to_string(i) + ".yml");
				std::cout << "    --> Models saved" << std::endl;

				// Clear boost object
				boost_->clear();
			}
		}
		catch (const std::exception &e)
		{ // referencia a base de un objeto polimorfo
			std::cout << e.what();
		}

		// SVM train data
		try
		{
			// svm_->trainAuto(data);
			if (svm_->isTrained())
			{
				std::cout << "    --> SVM model trained" << std::endl;

				// Calculate error over the split test data for SVM model
				l_error = svm_->calcError(data, true, l_error_mat);
				std::cout << "    --> SVM error percentage over test: " << l_error << std::endl;

				// Calculate error over the split train data  for SVM model
				l_error = svm_->calcError(data, false, l_error_mat);
				std::cout << "    --> SVM error percentage over train: " << l_error << std::endl;

				svm_->save(f_model_path + "\\resources\\SVM_MarkPoint_Model_" + std::to_string(i) + ".yml");
			}
		}
		catch (const std::exception &e)
		{ // referencia a base de un objeto polimorfo
			std::cout << e.what();
		}
	}
}

//---------------------------------------------------------
// Predict module
//---------------------------------------------------------
void predictImages(const std::string &f_model_path, CFile &f_file_object, uint8_t f_width, uint8_t f_height)
{
	// Variables
	std::vector<cv::Ptr<cv::ml::Boost>> l_boost_detectors;
	std::vector<cv::Ptr<cv::ml::SVM>> l_svm_detectors;
	CFile l_file_object;
	std::list<cv::Mat> l_image_list;
	std::vector<std::string> l_file_names;
	cv::Mat l_crop_image, l_painted_boost_img, l_painted_svm_img, l_aux, l_aux_lines;
	std::vector<cv::Mat> l_gradient;
	std::vector<float> l_boost_responses, l_svm_responses;
	cv::Rect2d l_rect_selected;
	bool l_is_roi_selected = false;
	float l_distance;
	uint32_t l_nFrames = 0U;
	std::vector<float> l_confidence_vector;
	std::vector<cv::Rect> l_up_rectangles, l_down_rectangles, l_right_rectangles, l_left_rectangles;
	std::list<CSlot> l_slot_list;

	std::cout << "---------------------------------------------------------" << std::endl;
	std::cout << "--> Test step:" << std::endl;

#ifdef READ_VIDEO
	// Reading video
	cv::VideoCapture cap(f_file_object.getPath() + "\\..\\TestSet\\predictions\\output.mp4");
#endif // READ_VIDEO

#ifdef READ_IMAGES
	// Make a list of images for test
	l_file_object.setPath(f_file_object.getPath() + "\\..\\TestSet\\predictions"); // Training path -> \\positive_samples
	l_file_object.fileNamesByExtension("bmp", l_file_names);
	l_file_object.readBmp(l_file_names, l_image_list);
#endif // READ_IMAGES

	// Read and load ML boost models
	for (uint8_t i = 0; i <= m_number_detectors; i++)
	{
		l_boost_detectors.push_back(cv::ml::StatModel::load<cv::ml::Boost>(f_model_path + "\\resources\\Boost_MarkPoint_Model_" + std::to_string(i) + ".yml"));
	}

	// Read and load ML SVM models
	// for (uint8_t i = 0; i <= m_number_detectors; i++)
	//{
	//	l_svm_detectors.push_back(cv::ml::StatModel::load<cv::ml::SVM>(f_model_path + "\\resources\\SVM_MarkPoint_Model_" + std::to_string(i) + ".yml"));
	// }

#ifdef READ_VIDEO
	// Check if video opened successfully
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream or file" << std::endl;
		return;
	}
	while (1)
	{
		cv::Mat l_image;
		// Capture frame-by-frame
		cap >> l_image;
		// Increase the number of frames
		l_nFrames++;
#endif // READ_VIDEO

#ifdef READ_IMAGES
		for (cv::Mat l_image : l_image_list)
		{
			// Increase the number of frames
			l_nFrames++;
#endif // READ_IMAGES

#ifdef TIME_PROCESS
			// Process time counter initialize
			int64_t l_time1, l_time2;
			l_time1 = cv::getTickCount();
#endif // TIME_PROCESS

			l_image.copyTo(l_painted_boost_img);
			// l_image.copyTo(l_painted_svm_img);
			l_image.copyTo(l_aux);
			l_image.copyTo(l_aux_lines);

			// Select the roi that we don't need to process ( black box car in surround images )
			if (!l_is_roi_selected)
			{
				l_rect_selected = cv::selectROI(l_image);
				l_is_roi_selected = true;
				cv::destroyAllWindows();
			}

			// Create grid over the original image to get the differents square pieces for process and predict
			for (uint16_t y = 0; y < l_image.rows; y += f_height / m_number_divisions)
			{
				for (uint16_t x = 0; x < l_image.cols; x += f_width / m_number_divisions)
				{
					// Size restriction
					if (x + f_width > l_image.cols || y + f_height > l_image.rows)
						continue;

					// Car restrictions
					if (x >= l_rect_selected.x &&
						y >= l_rect_selected.y &&
						x <= l_rect_selected.width + l_rect_selected.x &&
						y <= l_rect_selected.height + l_rect_selected.y)
						continue;
					if (x + f_width >= static_cast<uint16_t>(l_rect_selected.x) &&
						y + f_height >= static_cast<uint16_t>(l_rect_selected.y) &&
						x + f_width <= static_cast<uint16_t>(l_rect_selected.width + l_rect_selected.x) &&
						y + f_height <= static_cast<uint16_t>(l_rect_selected.height + l_rect_selected.y))
						continue;
					if (x >= l_rect_selected.x &&
						y + f_height >= static_cast<uint16_t>(l_rect_selected.y) &&
						x <= l_rect_selected.width + l_rect_selected.x &&
						y + f_height <= static_cast<uint16_t>(l_rect_selected.height + l_rect_selected.y))
						continue;
					if (x + f_width >= static_cast<uint16_t>(l_rect_selected.x) &&
						y >= l_rect_selected.y &&
						x + f_width <= static_cast<uint16_t>(l_rect_selected.width + l_rect_selected.x) &&
						y <= l_rect_selected.height + l_rect_selected.y)
						continue;

					// Crop image
					cv::Rect l_rect = cv::Rect(x, y, f_width, f_height);
					l_crop_image = l_image(l_rect);

					// Convert to grayscale
					cv::cvtColor(l_crop_image, l_crop_image, cv::COLOR_RGB2GRAY);

					// cv::imshow("Gray Image", l_crop_image);
					// cv::waitKey(0);

					// Down-scale and upscale the image to filter out the noise
					cv::GaussianBlur(l_crop_image, l_crop_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

					// cv::imshow("Filtering noise", l_crop_image);
					// cv::waitKey(0);

					//#ifdef TIME_PROCESS
					////Process time counter initialize
					//				int64_t l_time3, l_time4;
					//				l_time3 = cv::getTickCount();
					//#endif	// TIME_PROCESS

					// Improve contrast stretching grayscale histogram
					stretchHistogram(l_crop_image);

					//#ifdef TIME_PROCESS
					//				//Obtaining time of process
					//				l_time4 = cv::getTickCount();
					//				l_time4 = l_time3 - l_time4;
					//				//Convert to miliseconds
					//				l_time4 = 1000 * l_time4 / cv::getTickFrequency();
					//				l_time3 = 1000 * l_time4;
					//
					//				std::cout << "---------------------------------------------------------" << std::endl;
					//				std::cout << "\n\n--> Strecht time:" << l_time4 << " ms" << std::endl;
					//#endif // TIME_PROCESS

					// cv::imshow("Stretched histogram Image", l_crop_image);
					// cv::waitKey(0);

					// Calculate hog for that piece of image
					if (hogs_sel)
					{
						computeHOGs(l_crop_image, l_gradient, f_width, f_height);
					}
					else
					{
						computeHOG(l_crop_image, l_gradient, f_width, f_height);
					}

					// Traspose features matrix
					l_gradient.back() = l_gradient.back().t();

					////Predict with SVM detectors
					// for (uint8_t i = 0; i <= m_number_detectors; i++)
					//{
					//	l_svm_responses.push_back( l_svm_detectors[i]->predict(l_gradient.back()) );
					//	if (l_svm_responses[i] == 1)
					//	{
					//		l_distance = distanceSample(l_gradient.back(), l_svm_detectors[i]);
					//		if (l_distance != 0)
					//			l_confidence_vector.push_back( 1.0 / ( 1.0 + std::exp(-l_distance) ) );
					//	}
					//	else {
					//		l_confidence_vector.push_back(0);
					//	}
					//
					// }

					////Paint SVM results
					// for (uint8_t i = 0; i <= m_number_detectors; i++)
					//{
					//	if (l_confidence_vector[i] >= m_confidence)
					//	{
					//		switch (i)
					//		{
					//		case 0:
					//			cv::rectangle(l_painted_svm_img, l_rect, colors[RED]);
					//			break;
					//		case 1:
					//			cv::rectangle(l_painted_svm_img, l_rect, colors[ORANGE]);
					//			break;
					//		case 2:
					//			cv::rectangle(l_painted_svm_img, l_rect, colors[YELLOW]);
					//			break;
					//		case 3:
					//			cv::rectangle(l_painted_svm_img, l_rect, colors[GREEN]);
					//			break;
					//		}
					//	}
					// }

					// Predict with Boost detectors
					for (uint8_t i = 0; i <= m_number_detectors; i++)
					{
						l_boost_responses.push_back(l_boost_detectors[i]->predict(l_gradient.back()));
					}

					// Paint boost results
					for (uint8_t i = 0; i <= m_number_detectors; i++)
					{
						if (l_boost_responses[i] >= 1)
						{
							switch (i)
							{
							case 0:
#ifdef DEBUG_VISUAL_TRACE
								cv::rectangle(l_painted_boost_img, l_rect, colors[RED]);
#endif // DEBUG_VISUAL_TRACE

								// Store rectangles to correct overlapping
								l_right_rectangles.push_back(l_rect);
								break;
							case 1:
#ifdef DEBUG_VISUAL_TRACE
								cv::rectangle(l_painted_boost_img, l_rect, colors[ORANGE]);
#endif // DEBUG_VISUAL_TRACE

								// Store rectangles to correct overlapping
								l_up_rectangles.push_back(l_rect);
								break;
							case 2:
#ifdef DEBUG_VISUAL_TRACE
								cv::rectangle(l_painted_boost_img, l_rect, colors[YELLOW]);
#endif // DEBUG_VISUAL_TRACE

								// Store rectangles to correct overlapping
								l_left_rectangles.push_back(l_rect);
								break;
							case 3:
#ifdef DEBUG_VISUAL_TRACE
								cv::rectangle(l_painted_boost_img, l_rect, colors[GREEN]);
#endif // DEBUG_VISUAL_TRACE

								// Store rectangles to correct overlapping
								l_down_rectangles.push_back(l_rect);
								break;
							}
						}
					}

					// Clean variables
					if (!l_gradient.empty())
					{
						l_gradient.pop_back();
						l_boost_responses.clear();
						l_svm_responses.clear();
						l_confidence_vector.clear();
					}
				}
			}

			// Correct overlap in detections
			cv::groupRectangles(l_right_rectangles, 2, m_group_rectangles_scale);
			cv::groupRectangles(l_up_rectangles, 2, m_group_rectangles_scale);
			cv::groupRectangles(l_left_rectangles, 2, m_group_rectangles_scale);
			cv::groupRectangles(l_down_rectangles, 2, m_group_rectangles_scale);

			uint8_t size = static_cast<uint8_t>(l_right_rectangles.size());
			for (int i = 0; i < size; i++)
			{
				l_right_rectangles.push_back(cv::Rect(l_right_rectangles[i]));
			}
			size = static_cast<uint8_t>(l_left_rectangles.size());
			for (int i = 0; i < size; i++)
			{
				l_left_rectangles.push_back(cv::Rect(l_left_rectangles[i]));
			}
			size = static_cast<uint8_t>(l_up_rectangles.size());
			for (int i = 0; i < size; i++)
			{
				l_up_rectangles.push_back(cv::Rect(l_up_rectangles[i]));
			}
			size = static_cast<uint8_t>(l_down_rectangles.size());
			for (int i = 0; i < size; i++)
			{
				l_down_rectangles.push_back(cv::Rect(l_down_rectangles[i]));
			}

			cv::groupRectangles(l_right_rectangles, 1, 0.2);
			cv::groupRectangles(l_up_rectangles, 1, 0.2);
			cv::groupRectangles(l_left_rectangles, 1, 0.2);
			cv::groupRectangles(l_down_rectangles, 1, 0.2);

			// Here develop json detections
			std::vector<cv::Rect> l_all_rectangles;
			l_all_rectangles.insert(std::end(l_all_rectangles), std::begin(l_right_rectangles), std::end(l_right_rectangles));
			l_all_rectangles.insert(std::end(l_all_rectangles), std::begin(l_left_rectangles), std::end(l_left_rectangles));
			l_all_rectangles.insert(std::end(l_all_rectangles), std::begin(l_down_rectangles), std::end(l_down_rectangles));
			l_all_rectangles.insert(std::end(l_all_rectangles), std::begin(l_up_rectangles), std::end(l_up_rectangles));

			// Group rectangles of all clasifiers
			groupRectanglesModified(l_all_rectangles, 1, 0.7, 0, 0);

#ifdef LOG_RESULTS
			// Log results
			l_file_object.logResults(l_nFrames, f_file_object.getPath() + "\\..\\TestSet\\predictions\\annotations\\annotations.json", l_all_rectangles);
#endif

#ifdef DEBUG_VISUAL_TRACE

			for (size_t i = 0; i < l_all_rectangles.size(); i++)
			{
				cv::rectangle(l_aux, l_all_rectangles[i], cv::Scalar(255, 128, 255));
			}

			cv::namedWindow("Boost Results for all detectors mixed", cv::WINDOW_AUTOSIZE);
			cv::imshow("Boost Results for all detectors mixed", l_aux);
			cv::waitKey(10);
#endif

			///////////////////////////////////

			for (size_t i = 0; i < l_right_rectangles.size(); i++)
			{
				cv::rectangle(l_aux, l_right_rectangles[i], colors[RED]);
			}
			for (size_t i = 0; i < l_up_rectangles.size(); i++)
			{
				cv::rectangle(l_aux, l_up_rectangles[i], colors[ORANGE]);
			}
			for (size_t i = 0; i < l_left_rectangles.size(); i++)
			{
				cv::rectangle(l_aux, l_left_rectangles[i], colors[YELLOW]);
			}
			for (size_t i = 0; i < l_down_rectangles.size(); i++)
			{
				cv::rectangle(l_aux, l_down_rectangles[i], colors[GREEN]);
			}

#ifdef DEBUG_VISUAL_TRACE
			// Text info
			cv::putText(l_aux, "RED -> Right", cv::Point2d(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[RED], 1, cv::LINE_AA);
			cv::putText(l_aux, "ORANGE -> Up", cv::Point2d(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[ORANGE], 1, cv::LINE_AA);
			cv::putText(l_aux, "YELLOW -> Left", cv::Point2d(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[YELLOW], 1, cv::LINE_AA);
			cv::putText(l_aux, "GREEN -> Down", cv::Point2d(10, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[GREEN], 1, cv::LINE_AA);

			cv::imshow("Boost Results with overlapping corrected", l_aux);
			// cv::imshow("Boost Results", l_painted_boost_img);
			// cv::imshow("SVM Results", l_painted_svm_img);
			cv::waitKey(10);
#endif

			// Slot inference
			slotInference(l_slot_list, l_right_rectangles, l_up_rectangles, l_left_rectangles, l_down_rectangles, l_aux_lines);

			// Clear variables
			l_right_rectangles.clear();
			l_up_rectangles.clear();
			l_down_rectangles.clear();
			l_left_rectangles.clear();

#ifdef TIME_PROCESS
			// Obtaining time of process
			l_time2 = cv::getTickCount();
			l_time2 = l_time2 - l_time1;
			// Convert to miliseconds
			l_time2 = 1000 * l_time2 / static_cast<int64_t>(cv::getTickFrequency());
			l_time1 = 1000 * l_time2;

			std::cout << "---------------------------------------------------------" << std::endl;
			std::cout << "\n\n--> Total process time:" << l_time2 << " ms" << std::endl;
#endif // TIME_PROCESS
		}
		cap.release();
		cv::destroyAllWindows();
	}

	float distanceSample(cv::Mat & f_sample, const cv::Ptr<cv::ml::SVM> &f_svm)
	{
		// Variables
		cv::Mat l_result;
		float l_distance;

		f_svm->predict(f_sample, l_result, cv::ml::StatModel::Flags::RAW_OUTPUT);
		l_distance = l_result.at<float>(0, 0);
		return l_distance;
	}

	//---------------------------------------------------------
	// Slot Inference
	// This function detect slots, then visualize and save them in a list
	// output: List with slots detected
	//---------------------------------------------------------
	void slotInference(std::list<CSlot> & f_slot_list, std::vector<cv::Rect> & f_right_rectangles, std::vector<cv::Rect> & f_up_rectangles,
					   std::vector<cv::Rect> & f_left_rectangles, std::vector<cv::Rect> & f_down_rectangles, cv::Mat & f_image)
	{
#ifdef TIME_PROCESS
		// Process time counter initialize
		int64_t l_time11, l_time22;
		l_time11 = cv::getTickCount();
#endif // TIME_PROCESS

		// Make a copy of input image
		cv::Mat l_image;
		f_image.copyTo(l_image);

		// Correct and draw every slot with all orientations
		correctAndDrawLeftSlots(f_left_rectangles, f_image, f_slot_list);
		correctAndDrawRightSlots(f_right_rectangles, f_image, f_slot_list);
		correctAndDrawUpSlots(f_up_rectangles, f_image, f_slot_list);
		correctAndDrawDownSlots(f_down_rectangles, f_image, f_slot_list);

// Visualize slots detected
#ifdef DEBUG_VISUAL_TRACE
#ifdef READ_VIDEO
		cv::imshow("Slots Inference", f_image);
		cv::waitKey(10);
#endif // READ_VIDEO
#ifdef READ_IMAGES
		cv::imshow("Slots Inference", f_image);
		cv::waitKey(0);
#endif // READ_IMAGES
#endif // DEBUG_LEVEL_TRACE

#ifdef TIME_PROCESS
		// Obtaining time of process
		l_time22 = cv::getTickCount();
		l_time22 = l_time22 - l_time11;
		// Convert to miliseconds
		l_time22 = 1000 * l_time22 / cv::getTickFrequency();
		l_time11 = 1000 * l_time22;

		std::cout << "---------------------------------------------------------" << std::endl;
		std::cout << "\n\n--> Process time of Slot Inference:" << l_time22 << " ms" << std::endl;
#endif // TIME_PROCESS
	}

	void correctAndDrawLeftSlots(const std::vector<cv::Rect> &f_left_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list)
	{
		// Variable
		cv::Point2i l_point_1, l_point_2, l_point_3, l_point_4, l_point_max, l_point_min;
		float l_distance_p1_p2, l_distance_pmax_pmid, l_distance_pmin_pmid;
		float l_perpendicular_slope;
		float l_angle;
		int8_t l_id = 0;
		cv::Point2f l_mid_point;
		CSlot l_slot;
		std::list<CSlot> l_slot_list;

		// Loop between every pair of LEFT rectangles
		for (size_t i = 0; i < f_left_rectangles.size(); i++)
		{
			for (size_t j = 0; j < f_left_rectangles.size(); j++)
			{
				if (i == j || j < i)
					continue;

				// Calculate mass center in each rectangle
				l_point_1.x = f_left_rectangles[i].x + f_left_rectangles[i].width / 2;
				l_point_1.y = f_left_rectangles[i].y + f_left_rectangles[i].height / 2;
				l_point_2.x = f_left_rectangles[j].x + f_left_rectangles[j].width / 2;
				l_point_2.y = f_left_rectangles[j].y + f_left_rectangles[j].height / 2;

				// Calculate distance between points
				l_distance_p1_p2 = std::pow(l_point_2.x - l_point_1.x, 2) + std::pow(l_point_2.y - l_point_1.y, 2);
				l_distance_p1_p2 = std::sqrt(l_distance_p1_p2);

				// Slot compare
				if (l_distance_p1_p2 < m_max_dist_slot_short && l_distance_p1_p2 > m_min_dist_slot_short || l_distance_p1_p2 < m_max_dist_slot_long && l_distance_p1_p2 > m_min_dist_slot_long)
				{
					// Perpendicular slope calculation
					l_perpendicular_slope = static_cast<float>((l_point_2.x - l_point_1.x)) /
											static_cast<float>((l_point_2.y - l_point_1.y));

					// Middle point between point_1 and point_2
					l_mid_point.x = (static_cast<float>(l_point_1.x) + static_cast<float>(l_point_2.x)) / 2.0f;
					l_mid_point.y = (static_cast<float>(l_point_1.y) + static_cast<float>(l_point_2.y)) / 2.0f;

					// Solve perpendicular lines
					if (std::abs(l_point_1.x - l_point_2.x) < std::abs(l_point_1.y - l_point_2.y)) // then perpendicular variance in x
					{
						l_point_max.y = -((l_perpendicular_slope * (f_image.cols - l_mid_point.x)) - l_mid_point.y);
						l_point_max.x = f_image.cols;

						l_point_min.y = -((l_perpendicular_slope * (0 - l_mid_point.x)) - l_mid_point.y);
						l_point_min.x = 0;

						// Solve polygon point to slot
						l_point_3.x = 0;
						l_point_3.y = l_point_min.y - (l_mid_point.y - l_point_1.y);
						l_point_4.x = 0;
						l_point_4.y = l_point_min.y + (l_point_2.y - l_mid_point.y);
					}
					else
					{ // then perpendicular variance in y so, it�s not left slot
						continue;
					}

					// Calculate distance between generated points and the middle point
					l_distance_pmax_pmid = std::pow(l_point_max.x - l_mid_point.x, 2) + std::pow(l_point_max.y - l_mid_point.y, 2);
					l_distance_pmax_pmid = std::sqrt(l_distance_pmax_pmid);
					l_distance_pmin_pmid = std::pow(l_point_min.x - l_mid_point.x, 2) + std::pow(l_point_min.y - l_mid_point.y, 2);
					l_distance_pmin_pmid = std::sqrt(l_distance_pmin_pmid);

					// Distance restriction
					if (l_distance_pmax_pmid < l_distance_pmin_pmid)
					{
						continue;
					}
					else
					{
						l_angle = angleOf(l_mid_point, l_point_max);

						// Compare angles restrictions
						if (l_angle < 45.0 || l_angle > 315.0) // then we have a slot candidate
						{
							l_slot.setHeading(l_angle);
							l_id++; // Increase ID
							l_slot.setId(l_id);
							l_slot.setWidth(static_cast<int16_t>(l_distance_p1_p2));
							l_slot.setPoint1(cv::Point2i(l_point_1.x, l_point_1.y));
							l_slot.setPoint2(cv::Point2i(l_point_2.x, l_point_2.y));
							l_slot.setPoint3(cv::Point2i(l_point_3.x, l_point_3.y));
							l_slot.setPoint4(cv::Point2i(l_point_4.x, l_point_4.y));
							l_slot.setOrientation("left");

							// Save slot candidate
							l_slot_list.push_back(l_slot);
						}
					}
				}
			}
		}

		std::list<CSlot> l_slot_candidates_list;
		bool l_not_candidate = false;

		// Correct slot overlapping
		for (CSlot l_slot_object1 : l_slot_list)
		{
			for (CSlot l_slot_object2 : l_slot_list)
			{

				if (l_slot_object1.getId() == l_slot_object2.getId())
				{
					continue;
				}

				else if (l_slot_object2.getPoint1().y > l_slot_object1.getPoint1().y &&
						 l_slot_object2.getPoint1().y < l_slot_object1.getPoint2().y)
				{
					l_not_candidate = true;
					continue;
				}

				else if (l_slot_object2.getPoint2().y > l_slot_object1.getPoint1().y &&
						 l_slot_object2.getPoint2().y < l_slot_object1.getPoint2().y)
				{
					l_not_candidate = true;
					continue;
				}
			}

			if (l_not_candidate)
			{
				// Update boolean
				l_not_candidate = false;
				continue;
			}
			else
			{
				l_slot_candidates_list.push_back(l_slot_object1);
				// Update boolean
				l_not_candidate = false;
			}
		}

		for (CSlot l_slot_object : l_slot_candidates_list)
		{

#ifdef DEBUG_VISUAL_TRACE
			// Draw slots candidates
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint2(), colors[YELLOW], 2);
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint3(), colors[YELLOW], 2);
			cv::line(f_image, l_slot_object.getPoint2(), l_slot_object.getPoint4(), colors[YELLOW], 2);
#endif // DEBUG_VISUAL_TRACE

#ifdef DEBUG_PROMPT_TRACE
			// Console debug
			std::cout << "---------------" << std::endl;
			std::cout << "Slot Point with ID -> " << std::to_string(l_slot_object.getId()) << ":\n\n"
					  << std::endl;
			std::cout << "Point 1: " << l_slot_object.getPoint1() << "\n"
					  << std::endl;
			std::cout << "Point 2: " << l_slot_object.getPoint2() << "\n"
					  << std::endl;
			std::cout << "Point 3: " << l_slot_object.getPoint3() << "\n"
					  << std::endl;
			std::cout << "Point 4: " << l_slot_object.getPoint4() << "\n"
					  << std::endl;
#endif

			// Save slot candidate in function list
			f_slot_list.push_back(l_slot_object);
		}
	}

	void correctAndDrawRightSlots(const std::vector<cv::Rect> &f_right_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list)
	{
		// Variable
		cv::Point2i l_point_1, l_point_2, l_point_3, l_point_4, l_point_max, l_point_min;
		float l_distance_p1_p2, l_distance_pmax_pmid, l_distance_pmin_pmid;
		float l_perpendicular_slope;
		float l_angle;
		int8_t l_id = 0;
		cv::Point2f l_mid_point;
		CSlot l_slot;
		std::list<CSlot> l_slot_list;

		// Loop between every pair of LEFT rectangles
		for (size_t i = 0; i < f_right_rectangles.size(); i++)
		{
			for (size_t j = 0; j < f_right_rectangles.size(); j++)
			{
				if (i == j || j < i)
					continue;

				// Calculate mass center in each rectangle
				l_point_1.x = f_right_rectangles[i].x + f_right_rectangles[i].width / 2;
				l_point_1.y = f_right_rectangles[i].y + f_right_rectangles[i].height / 2;
				l_point_2.x = f_right_rectangles[j].x + f_right_rectangles[j].width / 2;
				l_point_2.y = f_right_rectangles[j].y + f_right_rectangles[j].height / 2;

				// Calculate distance between points
				l_distance_p1_p2 = std::pow(l_point_2.x - l_point_1.x, 2) + std::pow(l_point_2.y - l_point_1.y, 2);
				l_distance_p1_p2 = std::sqrt(l_distance_p1_p2);

				// Slot compare
				if (l_distance_p1_p2 < m_max_dist_slot_short && l_distance_p1_p2 > m_min_dist_slot_short || l_distance_p1_p2 < m_max_dist_slot_long && l_distance_p1_p2 > m_min_dist_slot_long)
				{
					// Perpendicular slope calculation
					l_perpendicular_slope = static_cast<float>((l_point_2.x - l_point_1.x)) /
											static_cast<float>((l_point_2.y - l_point_1.y));

					// Middle point between point_1 and point_2
					l_mid_point.x = (static_cast<float>(l_point_1.x) + static_cast<float>(l_point_2.x)) / 2.0f;
					l_mid_point.y = (static_cast<float>(l_point_1.y) + static_cast<float>(l_point_2.y)) / 2.0f;

					// Solve perpendicular lines
					if (std::abs(l_point_1.x - l_point_2.x) < std::abs(l_point_1.y - l_point_2.y)) // then perpendicular variance in x
					{
						l_point_max.y = -((l_perpendicular_slope * (f_image.cols - l_mid_point.x)) - l_mid_point.y);
						l_point_max.x = f_image.cols;

						l_point_min.y = -((l_perpendicular_slope * (0 - l_mid_point.x)) - l_mid_point.y);
						l_point_min.x = 0;

						// Solve polygon point to slot
						l_point_3.x = f_image.cols;
						l_point_3.y = l_point_max.y - (l_mid_point.y - l_point_1.y);
						l_point_4.x = f_image.cols;
						l_point_4.y = l_point_max.y + (l_point_2.y - l_mid_point.y);
					}
					else
					{ // then perpendicular variance in y so, it�s not right slot
						continue;
					}

					// Calculate distance between generated points and the middle point
					l_distance_pmax_pmid = std::pow(l_point_max.x - l_mid_point.x, 2) + std::pow(l_point_max.y - l_mid_point.y, 2);
					l_distance_pmax_pmid = std::sqrt(l_distance_pmax_pmid);
					l_distance_pmin_pmid = std::pow(l_point_min.x - l_mid_point.x, 2) + std::pow(l_point_min.y - l_mid_point.y, 2);
					l_distance_pmin_pmid = std::sqrt(l_distance_pmin_pmid);

					// Distance restriction
					if (l_distance_pmax_pmid > l_distance_pmin_pmid)
					{
						continue;
					}
					else
					{
						l_angle = angleOf(l_mid_point, l_point_max);

						// Compare angles restrictions
						if (l_angle < 235.0 || l_angle > 135.0) // then we have a slot candidate
						{
							l_slot.setHeading(l_angle);
							l_id++; // Increase ID
							l_slot.setId(l_id);
							l_slot.setWidth(static_cast<int16_t>(l_distance_p1_p2));
							l_slot.setPoint1(cv::Point2i(l_point_1.x, l_point_1.y));
							l_slot.setPoint2(cv::Point2i(l_point_2.x, l_point_2.y));
							l_slot.setPoint3(cv::Point2i(l_point_3.x, l_point_3.y));
							l_slot.setPoint4(cv::Point2i(l_point_4.x, l_point_4.y));
							l_slot.setOrientation("right");

							// Save slot candidate
							l_slot_list.push_back(l_slot);
						}
					}
				}
			}
		}

		std::list<CSlot> l_slot_candidates_list;
		bool l_not_candidate = false;

		// Correct slot overlapping
		for (CSlot l_slot_object1 : l_slot_list)
		{
			for (CSlot l_slot_object2 : l_slot_list)
			{

				if (l_slot_object1.getId() == l_slot_object2.getId())
				{
					continue;
				}

				else if (l_slot_object2.getPoint1().y > l_slot_object1.getPoint1().y &&
						 l_slot_object2.getPoint1().y < l_slot_object1.getPoint2().y)
				{
					l_not_candidate = true;
					continue;
				}

				else if (l_slot_object2.getPoint2().y > l_slot_object1.getPoint1().y &&
						 l_slot_object2.getPoint2().y < l_slot_object1.getPoint2().y)
				{
					l_not_candidate = true;
					continue;
				}
			}

			if (l_not_candidate)
			{
				// Update boolean
				l_not_candidate = false;
				continue;
			}
			else
			{
				l_slot_candidates_list.push_back(l_slot_object1);
				// Update boolean
				l_not_candidate = false;
			}
		}

		for (CSlot l_slot_object : l_slot_candidates_list)
		{
#ifdef DEBUG_VISUAL_TRACE
			// Draw slots candidates
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint2(), colors[RED], 2);
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint3(), colors[RED], 2);
			cv::line(f_image, l_slot_object.getPoint2(), l_slot_object.getPoint4(), colors[RED], 2);
#endif // DEBUG_VISUAL_TRACE

#ifdef DEBUG_PROMPT_TRACE
			// Console debug
			std::cout << "---------------" << std::endl;
			std::cout << "Slot Point with ID -> " << std::to_string(l_slot_object.getId()) << ":\n\n"
					  << std::endl;
			std::cout << "Point 1: " << l_slot_object.getPoint1() << "\n"
					  << std::endl;
			std::cout << "Point 2: " << l_slot_object.getPoint2() << "\n"
					  << std::endl;
			std::cout << "Point 3: " << l_slot_object.getPoint3() << "\n"
					  << std::endl;
			std::cout << "Point 4: " << l_slot_object.getPoint4() << "\n"
					  << std::endl;
#endif

			// Save slot candidate in function list
			f_slot_list.push_back(l_slot_object);
		}
	}

	void correctAndDrawDownSlots(const std::vector<cv::Rect> &f_down_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list)
	{
		// Variable
		cv::Point2i l_point_1, l_point_2, l_point_3, l_point_4, l_point_max, l_point_min;
		float l_distance_p1_p2, l_distance_pmax_pmid, l_distance_pmin_pmid;
		float l_perpendicular_slope;
		float l_angle;
		int8_t l_id = 0;
		cv::Point2f l_mid_point;
		CSlot l_slot;
		std::list<CSlot> l_slot_list;

		// Loop between every pair of DOWN rectangles
		for (size_t i = 0; i < f_down_rectangles.size(); i++)
		{
			for (size_t j = 0; j < f_down_rectangles.size(); j++)
			{
				if (i == j || j < i)
					continue;

				// Calculate mass center in each rectangle
				l_point_1.x = f_down_rectangles[i].x + f_down_rectangles[i].width / 2;
				l_point_1.y = f_down_rectangles[i].y + f_down_rectangles[i].height / 2;
				l_point_2.x = f_down_rectangles[j].x + f_down_rectangles[j].width / 2;
				l_point_2.y = f_down_rectangles[j].y + f_down_rectangles[j].height / 2;

				// Calculate distance between points
				l_distance_p1_p2 = std::pow(l_point_2.x - l_point_1.x, 2) + std::pow(l_point_2.y - l_point_1.y, 2);
				l_distance_p1_p2 = std::sqrt(l_distance_p1_p2);

				// Slot compare
				if (l_distance_p1_p2 < m_max_dist_slot_short && l_distance_p1_p2 > m_min_dist_slot_short) // Not comparison in long slots to this orientation
				{
					// Perpendicular slope calculation
					l_perpendicular_slope = static_cast<float>(l_point_1.x - l_point_2.x) /
											static_cast<float>(l_point_1.y - l_point_2.y);
					l_perpendicular_slope = -l_perpendicular_slope;

					// Middle point between point_1 and point_2
					l_mid_point.x = (static_cast<float>(l_point_1.x) + static_cast<float>(l_point_2.x)) / 2.0f;
					l_mid_point.y = (static_cast<float>(l_point_1.y) + static_cast<float>(l_point_2.y)) / 2.0f;

					// Solve perpendicular lines
					if (std::abs(l_point_1.x - l_point_2.x) > std::abs(l_point_1.y - l_point_2.y))
					{
						l_point_max.y = 0;
						l_point_max.x = ((l_point_max.y - l_mid_point.y) / l_perpendicular_slope) + l_mid_point.x;

						l_point_min.y = f_image.rows;
						l_point_min.x = ((l_point_min.y - l_mid_point.y) / l_perpendicular_slope) + l_mid_point.x;

						////Draw perpendicular line to slot
						// cv::line(f_image, l_mid_point, l_point_max, colors[GREEN], 2);
						// cv::imshow("as", f_image);
						// cv::waitKey(0);
						// cv::line(f_image, l_mid_point, l_point_min, colors[GREEN], 2);
						// cv::imshow("as", f_image);
						// cv::waitKey(20);

						// Solve polygon point to slot
						l_point_3.x = l_point_min.x - (l_mid_point.x - l_point_1.x);
						l_point_3.y = f_image.rows;
						l_point_4.x = l_point_min.x + (l_point_2.x - l_mid_point.x);
						l_point_4.y = f_image.rows;
					}
					else
					{ // then perpendicular variance in 'x' so, it�s not down slot
						continue;
					}

					// Calculate distance between generated points and the middle point
					l_distance_pmax_pmid = std::pow(l_point_max.x - l_mid_point.x, 2) + std::pow(l_point_max.y - l_mid_point.y, 2);
					l_distance_pmax_pmid = std::sqrt(l_distance_pmax_pmid);
					l_distance_pmin_pmid = std::pow(l_point_min.x - l_mid_point.x, 2) + std::pow(l_point_min.y - l_mid_point.y, 2);
					l_distance_pmin_pmid = std::sqrt(l_distance_pmin_pmid);

					// Distance restriction
					if (l_distance_pmax_pmid < l_distance_pmin_pmid)
					{
						continue;
					}
					else
					{
						l_angle = angleOf(l_mid_point, l_point_max);

						// Compare angles restrictions
						if (l_angle < 135.0 || l_angle > 45.0) // then we have a slot candidate
						{
							l_slot.setHeading(l_angle);
							l_id++; // Increase ID
							l_slot.setId(l_id);
							l_slot.setWidth(static_cast<int16_t>(l_distance_p1_p2));
							l_slot.setPoint1(cv::Point2i(l_point_1.x, l_point_1.y));
							l_slot.setPoint2(cv::Point2i(l_point_2.x, l_point_2.y));
							l_slot.setPoint3(cv::Point2i(l_point_3.x, l_point_3.y));
							l_slot.setPoint4(cv::Point2i(l_point_4.x, l_point_4.y));
							l_slot.setOrientation("down");

							// Save slot candidate
							l_slot_list.push_back(l_slot);
						}
					}
				}
			}
		}

		std::list<CSlot> l_slot_candidates_list;
		bool l_not_candidate = false;

		// Correct slot overlapping
		for (CSlot l_slot_object1 : l_slot_list)
		{
			for (CSlot l_slot_object2 : l_slot_list)
			{

				if (l_slot_object1.getId() == l_slot_object2.getId())
				{
					continue;
				}

				else if (l_slot_object2.getPoint1().x > l_slot_object1.getPoint1().x &&
						 l_slot_object2.getPoint1().x < l_slot_object1.getPoint2().x)
				{
					l_not_candidate = true;
					continue;
				}

				else if (l_slot_object2.getPoint2().x > l_slot_object1.getPoint1().x &&
						 l_slot_object2.getPoint2().x < l_slot_object1.getPoint2().x)
				{
					l_not_candidate = true;
					continue;
				}
			}

			if (l_not_candidate)
			{
				// Update boolean
				l_not_candidate = false;
				continue;
			}
			else
			{
				l_slot_candidates_list.push_back(l_slot_object1);
				// Update boolean
				l_not_candidate = false;
			}
		}

		for (CSlot l_slot_object : l_slot_candidates_list)
		{
#ifdef DEBUG_VISUAL_TRACE
			// Draw slots candidates
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint2(), colors[GREEN], 2);
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint3(), colors[GREEN], 2);
			cv::line(f_image, l_slot_object.getPoint2(), l_slot_object.getPoint4(), colors[GREEN], 2);
#endif // DEBUG_VISUAL_TRACE

#ifdef DEBUG_PROMPT_TRACE
			// Console debug
			std::cout << "---------------" << std::endl;
			std::cout << "Slot Point with ID -> " << std::to_string(l_slot_object.getId()) << ":\n\n"
					  << std::endl;
			std::cout << "Point 1: " << l_slot_object.getPoint1() << "\n"
					  << std::endl;
			std::cout << "Point 2: " << l_slot_object.getPoint2() << "\n"
					  << std::endl;
			std::cout << "Point 3: " << l_slot_object.getPoint3() << "\n"
					  << std::endl;
			std::cout << "Point 4: " << l_slot_object.getPoint4() << "\n"
					  << std::endl;
#endif

			// Save slot candidate in function list
			f_slot_list.push_back(l_slot_object);
		}
	}

	void correctAndDrawUpSlots(const std::vector<cv::Rect> &f_up_rectangles, cv::Mat &f_image, std::list<CSlot> &f_slot_list)
	{
		// Variable
		cv::Point2i l_point_1, l_point_2, l_point_3, l_point_4, l_point_max, l_point_min;
		float l_distance_p1_p2, l_distance_pmax_pmid, l_distance_pmin_pmid;
		float l_perpendicular_slope;
		float l_angle;
		int8_t l_id = 0;
		cv::Point2f l_mid_point;
		CSlot l_slot;
		std::list<CSlot> l_slot_list;

		// Loop between every pair of DOWN rectangles
		for (size_t i = 0; i < f_up_rectangles.size(); i++)
		{
			for (size_t j = 0; j < f_up_rectangles.size(); j++)
			{
				if (i == j || j < i)
					continue;

				// Calculate mass center in each rectangle
				l_point_1.x = f_up_rectangles[i].x + f_up_rectangles[i].width / 2;
				l_point_1.y = f_up_rectangles[i].y + f_up_rectangles[i].height / 2;
				l_point_2.x = f_up_rectangles[j].x + f_up_rectangles[j].width / 2;
				l_point_2.y = f_up_rectangles[j].y + f_up_rectangles[j].height / 2;

				// Calculate distance between points
				l_distance_p1_p2 = std::pow(l_point_2.x - l_point_1.x, 2) + std::pow(l_point_2.y - l_point_1.y, 2);
				l_distance_p1_p2 = std::sqrt(l_distance_p1_p2);

				// Slot compare
				if (l_distance_p1_p2 < m_max_dist_slot_short && l_distance_p1_p2 > m_min_dist_slot_short) // Not comparison in long slots to this orientation
				{
					// Perpendicular slope calculation
					l_perpendicular_slope = static_cast<float>(l_point_1.x - l_point_2.x) /
											static_cast<float>(l_point_1.y - l_point_2.y);
					l_perpendicular_slope = -l_perpendicular_slope;

					// Middle point between point_1 and point_2
					l_mid_point.x = (static_cast<float>(l_point_1.x) + static_cast<float>(l_point_2.x)) / 2.0f;
					l_mid_point.y = (static_cast<float>(l_point_1.y) + static_cast<float>(l_point_2.y)) / 2.0f;

					// Solve perpendicular lines
					if (std::abs(l_point_1.x - l_point_2.x) > std::abs(l_point_1.y - l_point_2.y))
					{
						l_point_max.y = 0;
						l_point_max.x = ((l_point_max.y - l_mid_point.y) / l_perpendicular_slope) + l_mid_point.x;

						l_point_min.y = f_image.rows;
						l_point_min.x = ((l_point_min.y - l_mid_point.y) / l_perpendicular_slope) + l_mid_point.x;

						////Draw perpendicular line to slot
						// cv::line(f_image, l_mid_point, l_point_max, colors[GREEN], 2);
						// cv::imshow("as", f_image);
						// cv::waitKey(0);
						// cv::line(f_image, l_mid_point, l_point_min, colors[GREEN], 2);
						// cv::imshow("as", f_image);
						// cv::waitKey(20);

						// Solve polygon point to slot
						l_point_3.x = l_point_max.x - (l_mid_point.x - l_point_1.x);
						l_point_3.y = 0;
						l_point_4.x = l_point_max.x + (l_point_2.x - l_mid_point.x);
						l_point_4.y = 0;
					}
					else
					{ // then perpendicular variance in 'x' so, it�s not down slot
						continue;
					}

					// Calculate distance between generated points and the middle point
					l_distance_pmax_pmid = std::pow(l_point_max.x - l_mid_point.x, 2) + std::pow(l_point_max.y - l_mid_point.y, 2);
					l_distance_pmax_pmid = std::sqrt(l_distance_pmax_pmid);
					l_distance_pmin_pmid = std::pow(l_point_min.x - l_mid_point.x, 2) + std::pow(l_point_min.y - l_mid_point.y, 2);
					l_distance_pmin_pmid = std::sqrt(l_distance_pmin_pmid);

					// Distance restriction
					if (l_distance_pmax_pmid > l_distance_pmin_pmid)
					{
						continue;
					}
					else
					{
						l_angle = angleOf(l_mid_point, l_point_max);

						// Compare angles restrictions
						if (l_angle < 315.0 || l_angle > 225.0) // then we have a slot candidate
						{
							l_slot.setHeading(l_angle);
							l_id++; // Increase ID
							l_slot.setId(l_id);
							l_slot.setWidth(static_cast<int16_t>(l_distance_p1_p2));
							l_slot.setPoint1(cv::Point2i(l_point_1.x, l_point_1.y));
							l_slot.setPoint2(cv::Point2i(l_point_2.x, l_point_2.y));
							l_slot.setPoint3(cv::Point2i(l_point_3.x, l_point_3.y));
							l_slot.setPoint4(cv::Point2i(l_point_4.x, l_point_4.y));
							l_slot.setOrientation("down");

							// Save slot candidate
							l_slot_list.push_back(l_slot);
						}
					}
				}
			}
		}

		std::list<CSlot> l_slot_candidates_list;
		bool l_not_candidate = false;

		// Correct slot overlapping
		for (CSlot l_slot_object1 : l_slot_list)
		{
			for (CSlot l_slot_object2 : l_slot_list)
			{

				if (l_slot_object1.getId() == l_slot_object2.getId())
				{
					continue;
				}

				else if (l_slot_object2.getPoint1().x > l_slot_object1.getPoint1().x &&
						 l_slot_object2.getPoint1().x < l_slot_object1.getPoint2().x)
				{
					l_not_candidate = true;
					continue;
				}

				else if (l_slot_object2.getPoint2().x > l_slot_object1.getPoint1().x &&
						 l_slot_object2.getPoint2().x < l_slot_object1.getPoint2().x)
				{
					l_not_candidate = true;
					continue;
				}
			}

			if (l_not_candidate)
			{
				// Update boolean
				l_not_candidate = false;
				continue;
			}
			else
			{
				l_slot_candidates_list.push_back(l_slot_object1);
				// Update boolean
				l_not_candidate = false;
			}
		}

		for (CSlot l_slot_object : l_slot_candidates_list)
		{
#ifdef DEBUG_VISUAL_TRACE
			// Draw slots candidates
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint2(), colors[ORANGE], 2);
			cv::line(f_image, l_slot_object.getPoint1(), l_slot_object.getPoint3(), colors[ORANGE], 2);
			cv::line(f_image, l_slot_object.getPoint2(), l_slot_object.getPoint4(), colors[ORANGE], 2);
#endif // DEBUG_VISUAL_TRACE

#ifdef DEBUG_PROMPT_TRACE
			// Console debug
			std::cout << "---------------" << std::endl;
			std::cout << "Slot Point with ID -> " << std::to_string(l_slot_object.getId()) << ":\n\n"
					  << std::endl;
			std::cout << "Point 1: " << l_slot_object.getPoint1() << "\n"
					  << std::endl;
			std::cout << "Point 2: " << l_slot_object.getPoint2() << "\n"
					  << std::endl;
			std::cout << "Point 3: " << l_slot_object.getPoint3() << "\n"
					  << std::endl;
			std::cout << "Point 4: " << l_slot_object.getPoint4() << "\n"
					  << std::endl;
#endif

			// Save slot candidate in function list
			f_slot_list.push_back(l_slot_object);
		}
	}

	/**
	 * Work out the angle from the x horizontal winding anti-clockwise
	 * in screen space.
	 *
	 * The value returned from the following should be 315.
	 * <pre>
	 * x,y -------------
	 *     |  1,1
	 *     |    \
	 *     |     \
	 *     |     2,2
	 * </pre>
	 * @param p1
	 * @param p2
	 * @return - a float from 0 to 360
	 */
	float angleOf(const cv::Point2f &f_p1, const cv::Point2f &f_p2)
	{
		// NOTE: Remember that most math has the Y axis as positive above the X.
		// However, for screens we have Y as positive below. For this reason,
		// the Y values are inverted to get the expected results.

		// Local variables
		float l_deltaY = (f_p1.y - f_p2.y);
		float l_deltaX = (f_p2.x - f_p1.x);

		float l_result = static_cast<float>(atan2(l_deltaY, l_deltaX) * 180.0 / PI);

		return (l_result < 0) ? (360.0f + l_result) : l_result;
	}

	void groupRectanglesModified(std::vector<cv::Rect> & rectList, int groupThreshold, double eps,
								 std::vector<int> *weights, std::vector<double> *levelWeights)
	{
		// cv::CV_INSTRUMENT_REGION();

		if (groupThreshold <= 0 || rectList.empty())
		{
			if (weights)
			{
				size_t i, sz = rectList.size();
				weights->resize(sz);
				for (i = 0; i < sz; i++)
					(*weights)[i] = 1;
			}
			return;
		}

		std::vector<int> labels;
		int nclasses = cv::partition(rectList, labels, cv::SimilarRects(eps));

		std::vector<cv::Rect> rrects(nclasses);
		std::vector<int> rweights(nclasses, 0);
		std::vector<int> rejectLevels(nclasses, 0);
		std::vector<double> rejectWeights(nclasses, DBL_MIN);
		int i, j, nlabels = (int)labels.size();
		for (i = 0; i < nlabels; i++)
		{
			int cls = labels[i];
			rrects[cls].x += rectList[i].x;
			rrects[cls].y += rectList[i].y;
			rrects[cls].width += rectList[i].width;
			rrects[cls].height += rectList[i].height;
			rweights[cls]++;
		}

		bool useDefaultWeights = false;

		if (levelWeights && weights && !weights->empty() && !levelWeights->empty())
		{
			for (i = 0; i < nlabels; i++)
			{
				int cls = labels[i];
				if ((*weights)[i] > rejectLevels[cls])
				{
					rejectLevels[cls] = (*weights)[i];
					rejectWeights[cls] = (*levelWeights)[i];
				}
				else if (((*weights)[i] == rejectLevels[cls]) && ((*levelWeights)[i] > rejectWeights[cls]))
					rejectWeights[cls] = (*levelWeights)[i];
			}
		}
		else
			useDefaultWeights = true;

		for (i = 0; i < nclasses; i++)
		{
			cv::Rect r = rrects[i];
			float s = 1.f / rweights[i];
			rrects[i] = cv::Rect(cv::saturate_cast<int>(r.x * s),
								 cv::saturate_cast<int>(r.y * s),
								 cv::saturate_cast<int>(r.width * s),
								 cv::saturate_cast<int>(r.height * s));
		}

		rectList.clear();
		if (weights)
			weights->clear();
		if (levelWeights)
			levelWeights->clear();

		for (i = 0; i < nclasses; i++)
		{
			cv::Rect r1 = rrects[i];
			int n1 = rweights[i];
			double w1 = rejectWeights[i];
			int l1 = rejectLevels[i];

			// filter out rectangles which don't have enough similar rectangles
			// if (n1 <= groupThreshold)
			// 	continue;
			// filter out small face rectangles inside large rectangles
			for (j = 0; j < nclasses; j++)
			{
				int n2 = rweights[j];

				if (j == i || n2 <= groupThreshold)
					continue;
				cv::Rect r2 = rrects[j];

				int dx = cv::saturate_cast<int>(r2.width * eps);
				int dy = cv::saturate_cast<int>(r2.height * eps);

				if (i != j &&
					r1.x >= r2.x - dx &&
					r1.y >= r2.y - dy &&
					r1.x + r1.width <= r2.x + r2.width + dx &&
					r1.y + r1.height <= r2.y + r2.height + dy &&
					(n2 > std::max(3, n1) || n1 < 3))
					break;
			}

			if (j == nclasses)
			{
				rectList.push_back(r1);
				if (weights)
					weights->push_back(useDefaultWeights ? n1 : l1);
				if (levelWeights)
					levelWeights->push_back(w1);
			}
		}
	}

	//---------------------------------------------------------