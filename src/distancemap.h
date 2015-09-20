#ifndef DISTANCEMAP_H
#define DISTANCEMAP_H

/*
 * Class for calculating the solution to the eikonal equation in 
 * arbitrary dimension.  It can also calculate a minimal path from a
 * starting point.
 *
 * Note that the potential must satisfy: P>0
 * The fast marching algorithm works if P is zero in some places,
 * but the minimal path will not be unique.
 * Unpredictable results if P<0 (indeed the program may crash).
 */
#include <cstdlib>
#include <math.h>
#include <float.h>
#include <iostream>
#include "minheap.h"

const static int MAXDIM=5;

#define MAXDIST 100.0

enum {FA, TRIAL, ALIVE};

class distancemap {
 public:
  int dim;                   //dimension of distance map
  int size[MAXDIM];         //size of each dimension
  int increments[MAXDIM];   //increment to move in x,y,... directions
  int coords[MAXDIM];       //coordinates of a certain pixel
  int GridSize;              //total pixels of grid
  
  float *P;                  //Potential defined on grid
  float *u;                  //Distance map defined on grid

  unsigned char *label;      //status of pixel (FAR, TRIAL, ALIVE)  
  minheap triallist;         //heap structure for fast marching

  int *minpath;              //list of pixels on a minimal path
  int size_minpath;          //number of pixels on minimal path

 public:
  distancemap() {P=u=0; size_minpath=dim=0; label=0; minpath=0;}
  int  allocate(int dim, int *size, float *P);
  void deallocate();
  void findDistanceMap(double *psi, int *Loc, int &Length, int *back, bool *indis);
  void getcoords(int p, int *coords) {
    for (int d=dim-1; d>=0; d--) {
      coords[d]=p/increments[d];
      p%=increments[d];
    }
  }
  int   inGrid(int dim, int coord)  {return coord>=0 && coord<size[dim];}
  float computeDistance(int p);
  int   findMinPath(int p);
  int   findMinPath2(int p);
};

#endif