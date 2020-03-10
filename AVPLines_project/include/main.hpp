
#include <CFile.hpp>
#include <CMarkPoint.hpp>

#include <string>
#include <iostream>
#include <list>
#include <vector>

//Main
int main(int argc, const char* argv[]); 

//Methods
void writeTrainingSet(std::vector<std::string>& l_file_json_names, std::vector<std::string>& l_file_bmp_names,
	std::list<CMarkPoint>& l_mark_point_list, CFile& l_file_object);

void featuresExtractor(CFile& f_file_object, std::vector<std::string>& f_file_bmp_names);