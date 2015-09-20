#ifndef Seg_by_Complementarity_array_to_matrix_h
#define Seg_by_Complementarity_array_to_matrix_h

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int write_array_to_text( int *labels, int XSize, int YSize, int frameno, std::string filename );

int write_array_to_text_double( double *labels, int XSize, int YSize, int frameno, std::string filename );

int get_array_from_text( int *labels, int &XSize, int &YSize, int frameno, std::string filename );

void FindBlobs( const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs );

#endif