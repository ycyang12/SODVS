#ifndef _MULTI_OBJECT_TRACKING
#define _MULTI_OBJECT_TRACKING
#include "region_solver.h"

void read_parameters( );

int get_initial_labels_from_mask( int frameno, std::string maskname );

int get_next_frame( );

int render_images_using_labels( cv::Mat img1, int *labels, int frameno );

void multi_object_tracking_and_segmentation( std::string imgname0, std::string imgname1, std::string imgname2, std::string maskname, int frameno );

int initialize_labelmap( int *labelmap, int GSize, int XSize, int YSize );

void regularize_labelmap( int *labelmap, int GSize, int XSize, int YSize );//

void dissocclusion_classification( cv::Mat &img1, cv::Mat &img2, int *labelmap, int frameno );

void hole_detection( int *labelmap, int frameno );

void classify_disocclusion_new_no_blob( region_solver *regionPlus, int *labelmap, int frameno );

void markimages_using_label( cv::Mat img0, int *labelmap, std::string imname, int num );//

void combined_warp( region_solver *myregions, int *labelmap, std::string imgname, int num );

void update_label_new( region_solver *regionPlus, region_solver *regionMinus, int *newlabel, int *oldlabel, bool onlyintensity=false );

void save_labelmap( int *labelmap, cv::Mat maskimg, std::string imname, int num );//

void extract_the_largest_component( int *labelmap, int GSize, int XSize, int YSize );

void smoothing_labelmap( int *labelmap, int windowsize, float threshold, region_solver *regplus );

#endif