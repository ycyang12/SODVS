#ifndef _NARROWBAND
#define _NARROWBAND

#include "cvsettings.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define    X 1
#define    Y XSize

#define BANDPOINTS(p) int *ptr=band,    p=*ptr; ptr!=tail;     p=*++ptr
#define EDGEPOINTS(p) int *ptr=edgeband,p=*ptr; ptr!=edgetail; p=*--ptr
#define NHBRPOINTS(p) int *ptr=nhbrband,p=*ptr; ptr!=nhbrtail; p=*++ptr
#define INTEPOINTS(p) int *ptr=interior,p=*ptr; ptr!=intetail; p=*--ptr
#define INTEPOINTS1(p,k) int *ptr=interior,p=*ptr,k=0; ptr!=intetail; p=*--ptr,++k
#define CONTPOINTS(p) int *ptr=contour, p=*ptr; ptr!=contail;  p=*++ptr 

class narrowband {

 public:

  int GridSize,XSize,YSize; // GridSize = XSize*YSize, XSize is width,and YSize is the height of the picture

  double *Psi;               //Level set function
  double *initial_Psi;
  double *newPsi;
  double *distance;          //Distance function (unsigned)
  double *norm_x;
  double *norm_y;
  
  int *band1, *band2,*band3;
  int *band, *tail;          //Stores band points in interior of physical grid
  int *edgeband, *edgetail;  //Stores band points at physical grid boundaries
  int *nhbrband, *nhbrtail;  //Stores neighbors of band points
  int *contour, *contail;
  int *interior, *intetail;

  unsigned char *inband;     //Grid used to mark which points are in narrowband
  unsigned char *edgebits;   //Bits describing edgepoint or not
  enum {XPOS=1, XNEG=2, YPOS=4, YNEG=8,
        XEDGES=3,       YEDGES=12};
  int *dirbits;

 public:

  //Constructor

  narrowband() {
    GridSize=XSize=YSize=0;
    Psi=0;
	initial_Psi=0;
	newPsi=0;
	distance=0;
    norm_x = 0;
	norm_y = 0;
	band1=band2=band3=0;
	inband=0; edgebits=0; dirbits=0; 
    band=0;       tail=0;
    edgeband=0;   edgetail=0;
    nhbrband=0;   nhbrtail=0;
	interior=0;   intetail=0;
	contour=0;    contail=0;
  }
  ~narrowband()
	{
		deallocate();
	}

  //Data allocation/deallocation
  int  allocate(int _XSize,int _YSize); //Allocate array space
  void deallocate();                               //Deallocate array space

  //Initialization routines
  void initialize();
  void markEdges();
  int  makeLSF(cv::Mat mask);
  int  createBand();                   //Create initial narrowband
  void createContour();         //Visualization of contour
  void get_norm();
  void extendBand();
  void updateBand();

};

#endif