#ifndef _MYPARAMS
#define _MYPARAMS
//********** region solver **********//
#define WSIZE_INC 5
#define WEIGHTONSHAPE 0.0//1.50
//********** region solver end **********//

//********** multi obj tracking **********//
#define TRANSLATION 10
#define INITIAL_ITERATIONS 20
#define REFINE_ITERATIONS 1

#define MAX_REFINEMENTS 24
#define MAX_STATISTICS 5
#define MAX_COMBO 3

#define OSCILATE 0.4
#define ISO_THRESH 0.05
#define BOTH_OCCLUSION 0.6
#define OUTPUTSEQ 0
#define SHADED_REGION 0
#define VIDEO_NAME "image"
#define MASKNAME "initial_mask"
#define VIDEO_SSD ".png"
#define LABELMAP_NAME "initial_labels"
//********** multi obj tracking **********//

//********** never used **********//
#define ROUND(x) ((int)(x+0.5))
#define STOP_SIGN 0.01
#define STOP_VALUE 1e-10
#define MAX_VELOCITY 0.5
#define PI 3.1415926
#define GAMA_OMEGA 1.0
//********** never used end **********//
#endif