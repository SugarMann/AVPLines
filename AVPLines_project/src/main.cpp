#include <stdlib.h>
#include <stdio.h>

#include <main.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//Define to write the markpoint dataset
//#define MAKE_TRAINING_SET

int main(int argc, const char* argv[])
{
	//Variables
	std::vector<std::string> l_file_json_names;
	std::vector<std::string> l_file_bmp_names;
	std::list<CMarkPoint> l_mark_point_list;
	CFile l_file_object;

	if (argc != 2) // argc must be 2 for correct execution
	{
		std::cout << "usage: " << argv[0] << " <path>\n"; // We assume that argv[0] is the program name
		return 1;
	}
	else
	{
		//Set the path in CFile object
		l_file_object.setPath(argv[1]);

		#ifdef MAKE_TRAINING_SET
		writeTrainingSet(l_file_json_names, l_file_bmp_names, l_mark_point_list, l_file_object);
		#endif // MAKE_TRAINING_SET

		featuresExtractor(l_file_object, l_file_bmp_names);

	}

	return 0;
}

void writeTrainingSet(std::vector<std::string>& f_file_json_names, std::vector<std::string>& f_file_bmp_names, 
	std::list<CMarkPoint>& f_mark_point_list, CFile& f_file_object)
{
	//Make a vector with all json file names in "positive samples"
	f_file_object.fileNamesByExtension("json", f_file_json_names);
	//Make a vector with all bmp file names in "positive samples"
	f_file_object.fileNamesByExtension("bmp", f_file_bmp_names);

	//Make a JSON MarkPoints object list, where each object have all MarkPoints in one image
	f_file_object.readJson(f_file_json_names, f_mark_point_list);
	f_file_json_names.clear();
	//Set related image for every MarkPoint object
	f_file_object.readBmp(f_file_bmp_names, f_mark_point_list);
	f_file_bmp_names.clear();
	//Make the training set images with MarkPoint list information
	f_file_object.makeTrainingSet(f_mark_point_list);
}

void featuresExtractor(CFile& f_file_object, std::vector<std::string>& f_file_bmp_names)
{
	//4 differents detector are trained (right, up, left, down markpoints directions)
	for (int i = 0; i < 3; i++)
	{
		//Prepare different paths for each detector
		f_file_object.setPath(f_file_object.getPath() + "\\dataset\\" + std::to_string(i));

		//Read images and save them in a markpoint list 
		f_file_object.fileNamesByExtension("bmp", f_file_bmp_names);
		std::list<CMarkPoint> l_mark_point_list((int)f_file_bmp_names.size(), CMarkPoint());
		f_file_object.readBmp(f_file_bmp_names, l_mark_point_list);

		////////////////////// HERE

	}

}