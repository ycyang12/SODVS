#ifndef _REGION_SOLVER
#define _REGION_SOLVER
#include "myparams.h"
#include "cvsettings.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "narrowband.h"
#include "distancemap.h"
#include "omp.h"
#include "xfv_timmer.h"

#define PARALLEL_THE_FOR_LOOPS


class region_solver{

public:

    int GridSize, XSize, YSize, KthRegion;
    float REGION_WIDTH, REGION_HEIGHT;
    
    double *image0; // GridSizeX3
    double *image0_hsv;
    double *evolved_image0; // GridSizeX3
    double *image1; // GridSizeX3
    double *evolved_image1; // GridSizeX3
    
    float *F_ratio;
    float *F_error;
    int *lookback;
    float *F_shape;
	
    float *binscoreimg0;
 	
    double *velocity; // GridSizeX2
    double *velocity_outside; //GridSizeX2
    float *region_indicator; // GridSize
    char *initial_indicator;
	
    float *backward_map; // GridSizeX2
    float *forward_map; // GridSizeX2
    uchar *occlusion_map; // GridSize
	
    double *grad_image1; // GridSizeX6
    double *det_backward_map; // GridSize
    double *grad_backward_map; // GridSizeX4
	
    double *cgm_Ax; //GridSizeX2
    double *cgm_b; //GridSizeX2
    double *residue; //GridSizeX2
    double *pbase; //GridSizeX2
    double bnorm;
    double maxresidue;
    double avg_G[2];
    double my_max_velocity;
    
    int *blockhead;
    int *blocktail;
    int NumofBlocks;
    
    std::vector< std::vector<cv::Point2i> > Myblobs;
    int *blobindex;
    double *BlobTrans;
    int NumofBlobs;
    
    enum {XPOS=1, XNEG=2, YPOS=4, YNEG=8,
		XEDGES=3,       YEDGES=12};
    uchar *edgebits; // GridSize
    int ncolors;
    int colorwheel[60][3];
    cv::Mat moved_img;
    int *ConstRegion;

    static const float BinSize1;
    static const float BinSize2;
    static const float BinSize3;
    static const float BinWght1;
    static const float BinWght2;
    static const float BinWght3;
    static const int SHAPEWINE;
    static const int VARWINE;
    static const float VART;
    static const int LOCAL_WINDOW;
    static const int SAMPLE_SIZE;
    float BETA0;
    static const int SMOOTHNESS_OF_OCC;

    static const int BlurSize;
    static const float SIGX;
	
public:
	
	region_solver(){
        GridSize = XSize = YSize = KthRegion = 0;
		image0 = 0;         // GridSizeX3
		image0_hsv = 0;     // GridSizeX3
		evolved_image0 = 0; // GridSizeX3
		image1 = 0;         // GridSizeX3
		evolved_image1 = 0; // GridSizeX3

        F_ratio = F_error = F_shape = 0;
		lookback = 0;

		binscoreimg0 = 0;

        velocity = velocity_outside = 0; //GridSizeX2
		region_indicator = 0; // GridSize
		initial_indicator = 0;

        backward_map = forward_map = 0; // GridSizeX2
		occlusion_map = 0; // GridSize

		grad_image1 = 0;
		det_backward_map = 0; // GridSize
		grad_backward_map = 0; // GridSizeX4

		cgm_Ax = cgm_b = residue = pbase = 0; //GridSizeX2
        
        blockhead = 0;
        blocktail = 0;
        NumofBlocks = 0;
        
        blobindex = 0;;
        BlobTrans = 0;
        NumofBlobs = 0;
        
		bnorm = 0;
        maxresidue = 0;
		my_max_velocity = 0;
		
        edgebits = 0; // GridSize

		ncolors = 0;
		ConstRegion = 0;
	}
	~region_solver()
	{
		deallocate();
	}
	
	int allocate( cv::Mat img0, cv::Mat img1, int *labelmap, int label, float &regionLen, float &regionSize );      //
	void deallocate();                                                                                              //
	void read_parameters( );                                                                                        //
	void initialize( cv::Mat img0, cv::Mat img1, int *labelmap, int label, float &regionLen, float &regionSize );   //
	void initialize2( int *labelmap, int label, float &regionLen, float &regionSize );
	void markEdges();                                                                                               //
	void imgTo1darray( cv::Mat img, double *image );                                                                //
	void compute_gradients( double *img, double *grad_img );                                                        //
	void compute_region_indicator( int *labelmap, int label, float &regionLen, float &regionSize );                 //
	void initialize_warp( float *warp );                                                                            //
	void compute_gradients_of_warp( float *warp, double *grad_warp );                                               //
	void compute_det_of_gradient_of_warp( double *grad_warp, double *det_grad_warp );                               //
	void get_evolved_image( double *img, float *warp, double *evolved_img, int pass=0 );                            //
    
    void BlocksDivision()
    {
        NumofBlocks = omp_get_num_procs();
        // Now set the number of threads
        omp_set_num_threads(NumofBlocks);
        std::cout<<"# of cores on my station: "<<NumofBlocks<<std::endl;
        if( NumofBlocks < 2 )
        {
            std::cout<<"cores less than 2, please turn off the parallel micro!"<<std::endl; while(1);
        }
        
        blockhead = new int[NumofBlocks];
        blocktail = new int[NumofBlocks];
        int blocklength = (int)(GridSize/NumofBlocks)-2;
        blockhead[0] = 0;
        blocktail[0] = 0 + blocklength;
        for( int i=1; i<NumofBlocks-1; ++i )
        {
            blockhead[i] = blocktail[i-1] + 1;
            blocktail[i] = blockhead[i] + blocklength;
        }
        blockhead[NumofBlocks-1] = blocktail[NumofBlocks-2] + 1;
        blocktail[NumofBlocks-1] = GridSize - 1;
    }
    
	void get_cgm_b( double *b, int pass=0 );
	void get_cgm_b_using_region_based_metric( double *b, bool *evolved_regicator, bool *evolved_occmap, int pass=0 );
	void get_cgm_Ax( double *x, double *Ax, int pass=0 );
	void get_cgm_Ax_using_region_based_metric( double *x, double *Ax, bool *evolved_regicator, int pass=0 );
	cv::Mat solve_for_velocity( int num );
	cv::Mat solve_for_velocity_using_region_based_metric( int num );
	void solve_for_translation( );                                                                                  //
	int vector_plus( bool *evolved_regicator, double *v1, double al ,double *v2, double *v3, int test );
    int vector_plus_2_args( bool *evolved_regicator, double *v1, double al ,double *v2, double *v3, \
                                                    double *x1, double bl, double *x2, double *x3);
	double vector_dotx( bool *evolved_regicator, double *v1, double *v2 );
	void normalize_velocity( );
	void update_backward_map( );
	void update_forward_map( );
	int update_occlusion_map( double &energy );
		
	void calculate_F_for_yanchaoscheme( int ori_win_size, int minimumsupport )
	{
		if( ori_win_size < 15 ) ori_win_size = 15;
		narrowband bandfordisplay;
		bandfordisplay.allocate(XSize, YSize);
		for( int p = 0; p < GridSize; ++p ){ bandfordisplay.Psi[p] = region_indicator[p]>0.5? -1.0 : 1.0; }
		bandfordisplay.createBand();
		//calculate ConstRegion
		for( int p = 0; p < GridSize; ++p ){ ConstRegion[p] = 0; F_shape[p] = -100; }
		std::vector<double> Vecofdiff;
		for( int *ptr = bandfordisplay.band, p = *ptr; ptr != bandfordisplay.tail; p = *++ptr )
		{
			if( lookback[p] > 900*900 )
			{
				std::cout<<"lookback "<<lookback[p]<<std::endl;
				std::ofstream myfilepr("indcator.txt");
				std::ofstream myfilepg("lookback.txt");
				for( int inp = 0 , i = 1 ; inp < GridSize ; ++inp , ++i )
				{
					myfilepr<<(int)initial_indicator[inp]<<std::ends<<std::ends<<std::ends;
					myfilepg<<(int)lookback[inp]<<std::ends<<std::ends<<std::ends;
					if(i%XSize==0)
					{
						myfilepr<<std::endl;
						myfilepg<<std::endl;
					}
				}
				myfilepr.close();
				myfilepg.close();
				std::cout<<"rrrrrrrrrrrrrrr"<<std::endl;
				std::cout<<"candidate pixel is out of band"<<std::endl;
				std::cout<<"region: "<<KthRegion<<std::endl;
				std::cout<<"p: "<<p<<" x: "<<p%XSize<<" y: "<<p/XSize<<std::endl;
				while(1);
			}
			int m = p/XSize;
			int n = p%XSize;
			int delta_y = (int)(lookback[p]/XSize) - m;
			int delta_x = lookback[p]%XSize - n;
			double sum_sqr_diff = 0, shape_diff = 0;
			double pixelcounted = 0, shape_area = 0;
			double pb = 0;
			double pg = 0;
			double pr = 0;
			Vecofdiff.clear();
			double temp_counted = 0;
			for( int i = m - VARWINE; i <= m + VARWINE; ++i )
				for( int j = n - VARWINE; j <= n + VARWINE; ++j )
				{
					int pp = i*XSize + j;
					if( (i-m)*(i-m)+(j-n)*(j-n) <= VARWINE*VARWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{
							temp_counted += 1.0;
							double ppb = image0[pp*CHANNELS+0];
							double ppg = image0[pp*CHANNELS+1];
							double ppr = image0[pp*CHANNELS+2];
							pb += ppb;
							pg += ppg;
							pr += ppr;
						}
					}
			 	}
			if( temp_counted == 0 ){ std::cout<<"something wrong calculate_F_for_yanchaoscheme"; while(1); }
			pb /= temp_counted;
			pg /= temp_counted;
			pr /= temp_counted;
			
			for( int i = m - MAX(VARWINE, SHAPEWINE); i <= m + MAX(VARWINE, SHAPEWINE); ++i )
			{
				for( int j = n - MAX(VARWINE, SHAPEWINE); j <= n + MAX(VARWINE, SHAPEWINE); ++j )
				{
					int pp = i*XSize + j;
					if( (i-m)*(i-m)+(j-n)*(j-n) <= VARWINE*VARWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{
							pixelcounted += 1.0;
							double ppb = image0[pp*CHANNELS+0];
							double ppg = image0[pp*CHANNELS+1];
							double ppr = image0[pp*CHANNELS+2];
							double df = (ppb-pb)*(ppb-pb) + (ppg-pg)*(ppg-pg) + (ppr-pr)*(ppr-pr);
							sum_sqr_diff += df;
							Vecofdiff.push_back(-df);
						}
					}
					if( (i-m)*(i-m)+(j-n)*(j-n) <= SHAPEWINE*SHAPEWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 )
						{
							shape_area += 1.0;
							int ppp = (i+delta_y)*XSize + j+delta_x;
							if( i+delta_y >=0 && i+delta_y <= YSize-1 && j+delta_x >= 0 && j+delta_x <= XSize-1 )
								shape_diff += ( region_indicator[pp] - (float)initial_indicator[ppp] );
							else
								shape_diff += ( region_indicator[pp] - 0 );
						}
						else
						{
							shape_area += 1.0;
							int ppp = (i+delta_y)*XSize + j+delta_x;
							if( i+delta_y >=0 && i+delta_y <= YSize-1 && j+delta_x >= 0 && j+delta_x <= XSize-1 )
								shape_diff += ( 0 - (float)initial_indicator[ppp] );
							else
								shape_diff += ( 0 - 0 );
						}
					}
				}
			}
			if( shape_area < 1.0 ){ std::cout<<" something wrong with shape area "; while(1); }
			F_shape[p] = shape_diff/shape_area;
			
			int PixNotUsed = (int)( pixelcounted*0.2 ); // 
			std::nth_element( Vecofdiff.begin(), Vecofdiff.begin() + PixNotUsed, Vecofdiff.end() );
			for( int k = 0; k < PixNotUsed; ++k ) sum_sqr_diff += Vecofdiff[k];
			
			if( sum_sqr_diff/( 1e-10 + pixelcounted - PixNotUsed ) < VART ) ConstRegion[p] = 1;
			Vecofdiff.clear();
		}
		for( int *ptr = bandfordisplay.edgeband, p = *ptr; ptr != bandfordisplay.edgetail; p = *--ptr )
		{
			if( lookback[p] > 900*900 )
			{
				std::cout<<"lookback "<<lookback[p]<<std::endl;
				std::ofstream myfilepr("indcator.txt");
				std::ofstream myfilepg("lookback.txt");
				for( int inp = 0 , i = 1 ; inp < GridSize ; ++inp , ++i )
				{
					myfilepr<<(int)initial_indicator[inp]<<std::ends<<std::ends<<std::ends;
					myfilepg<<(int)lookback[inp]<<std::ends<<std::ends<<std::ends;
					if(i%XSize==0)
					{
						myfilepr<<std::endl;
						myfilepg<<std::endl;
					}
				}
				myfilepr.close();
				myfilepg.close();
				std::cout<<"rrrrrrrrrrrrrrr"<<std::endl;
				std::cout<<"candidate pixel is out of band"<<std::endl;
				std::cout<<"region: "<<KthRegion<<std::endl;
				std::cout<<"p: "<<p<<" x: "<<p%XSize<<" y: "<<p/XSize<<std::endl;
				while(1);
			}
			int m = p/XSize;
			int n = p%XSize;
			int delta_y = (int)(lookback[p]/XSize) - m;
			int delta_x = lookback[p]%XSize - n;
			double sum_sqr_diff = 0, shape_diff = 0;
			double pixelcounted = 0, shape_area = 0;
			double pb = 0;
			double pg = 0;
			double pr = 0;
			Vecofdiff.clear();
			double temp_counted = 0;
			for( int i = m - VARWINE; i <= m + VARWINE; ++i )
				for( int j = n - VARWINE; j <= n + VARWINE; ++j )
				{
					int pp = i*XSize + j;
					if( (i-m)*(i-m)+(j-n)*(j-n) <= VARWINE*VARWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{
							temp_counted += 1.0;
							double ppb = image0[pp*CHANNELS+0];
							double ppg = image0[pp*CHANNELS+1];
							double ppr = image0[pp*CHANNELS+2];
							pb += ppb;
							pg += ppg;
			 				pr += ppr;
						}
					}
				}
			if( temp_counted == 0 ){ std::cout<<"something wrong calculate_F_for_yanchaoscheme"; while(1); }
			pb /= temp_counted;
			pg /= temp_counted;
			pr /= temp_counted;
			
			for( int i = m - MAX(VARWINE, SHAPEWINE); i <= m + MAX(VARWINE, SHAPEWINE); ++i )
			{
				for( int j = n - MAX(VARWINE, SHAPEWINE); j <= n + MAX(VARWINE, SHAPEWINE); ++j )
				{
					int pp = i*XSize + j;
					if( (i-m)*(i-m)+(j-n)*(j-n) <= VARWINE*VARWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{
							pixelcounted += 1.0;
							double ppb = image0[pp*CHANNELS+0];
							double ppg = image0[pp*CHANNELS+1];
							double ppr = image0[pp*CHANNELS+2];
							double df = (ppb-pb)*(ppb-pb) + (ppg-pg)*(ppg-pg) + (ppr-pr)*(ppr-pr);
							sum_sqr_diff += df;
							Vecofdiff.push_back(-df);
						}
					}
					if( (i-m)*(i-m)+(j-n)*(j-n) <= SHAPEWINE*SHAPEWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 )
						{
							shape_area += 1.0;
							int ppp = (i+delta_y)*XSize + j+delta_x;
							if( i+delta_y >=0 && i+delta_y <= YSize-1 && j+delta_x >= 0 && j+delta_x <= XSize-1 )
								shape_diff += ( region_indicator[pp] - (float)initial_indicator[ppp] );
							else
								shape_diff += ( region_indicator[pp] - 0 );
						}
						else
						{
							shape_area += 1.0;
							int ppp = (i+delta_y)*XSize + j+delta_x;
							if( i+delta_y >=0 && i+delta_y <= YSize-1 && j+delta_x >= 0 && j+delta_x <= XSize-1 )
								shape_diff += ( 0 - (float)initial_indicator[ppp] );
							else
								shape_diff += ( 0 - 0 );
						}
					}
				}
			}
			if( shape_area < 1.0 ){ std::cout<<" something wrong with shape area "; while(1); }
			F_shape[p] = shape_diff/shape_area;
			
			int PixNotUsed = (int)( pixelcounted*0.2 ); // 
			std::nth_element( Vecofdiff.begin(), Vecofdiff.begin() + PixNotUsed, Vecofdiff.end() );
			for( int k = 0; k < PixNotUsed; ++k ) sum_sqr_diff += Vecofdiff[k];
			
			if( sum_sqr_diff/( 1e-10 + pixelcounted - PixNotUsed ) < VART ) ConstRegion[p] = 1;
			Vecofdiff.clear();
		}
        
		for( int p=0; p<GridSize; ++p ) F_ratio[p] = F_error[p] = -100;
		double t_1 = (1.0/BinSize1) + 0.00001;
		double t_2 = (1.0/BinSize2) + 0.00001;
		double t_3 = (1.0/BinSize3) + 0.00001;
		for( int *ptr=bandfordisplay.band, p=*ptr; ptr!=bandfordisplay.tail; p=*++ptr )
		{
			int m = p/XSize;
			int n = p%XSize;
			double pixelcounted = 0;
			double pixelinbin1 = 0;
			double pixelinbin2 = 0;
			double pixelinbin3 = 0;
			double pb = image0_hsv[p*CHANNELS+0];
			double pg = image0_hsv[p*CHANNELS+1];
			double pr = image0_hsv[p*CHANNELS+2];
			int win_size = ori_win_size;
			do{
				pixelcounted = 0;
				pixelinbin1 = 0;
				pixelinbin2 = 0;
				pixelinbin3 = 0;
				for( int i = m-win_size; i <= m+win_size; ++i )
				{
					for( int j = n-win_size; j <= n+win_size; ++j )
					{
						if( (i-m)*(i-m)+(j-n)*(j-n) > win_size*win_size )continue;
						int pp = i*XSize + j;
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{	
							pixelcounted += 1.0;
							double ppb = image0_hsv[pp*CHANNELS+0];
							double ppg = image0_hsv[pp*CHANNELS+1];
							double ppr = image0_hsv[pp*CHANNELS+2];
							if ( fabs(pb-ppb)<t_1 && fabs(pg-ppg)<t_1 && fabs(pr-ppr)<t_1  ) pixelinbin1 += 1.0;
							if ( fabs(pb-ppb)<t_2 && fabs(pg-ppg)<t_2 && fabs(pr-ppr)<t_2  ) pixelinbin2 += 1.0;
							if ( fabs(pb-ppb)<t_3 && fabs(pg-ppg)<t_3 && fabs(pr-ppr)<t_3  ) pixelinbin3 += 1.0;
						}
					}
				}
				if( pixelcounted < minimumsupport ) win_size += WSIZE_INC;
				else break;
				if( win_size > MAX(XSize,YSize) )break;
			}while(1);
			float like1 = pixelinbin1/( pixelcounted + 1.0e-10 );
			float like2 = pixelinbin2/( pixelcounted + 1.0e-10 );
			float like3 = pixelinbin3/( pixelcounted + 1.0e-10 );
			binscoreimg0[p] = BinWght1*like1 + BinWght2*like2 + BinWght3*like3;
			
			double diff = 0;
			for( int k=0; k<CHANNELS; ++k )
			{
				double d = evolved_image1[p*CHANNELS+k] - image0[p*CHANNELS+k];
				diff += d*d;
			}
			
			F_ratio[p] = binscoreimg0[p] + WEIGHTONSHAPE*F_shape[p];
			F_error[p] = diff;
		}
        
		for( int *ptr=bandfordisplay.edgeband, p=*ptr; ptr!=bandfordisplay.edgetail; p=*--ptr )
		{
			int m = p/XSize;
			int n = p%XSize;
			double pixelcounted = 0;
			double pixelinbin1 = 0;
			double pixelinbin2 = 0;
			double pixelinbin3 = 0;
			double pb = image0_hsv[p*CHANNELS+0];
			double pg = image0_hsv[p*CHANNELS+1];
			double pr = image0_hsv[p*CHANNELS+2];
			int win_size = ori_win_size;
			do{
				pixelcounted = 0;
				pixelinbin1 = 0;
				pixelinbin2 = 0;
				pixelinbin3 = 0;
				for( int i = m-win_size; i <= m+win_size; ++i )
				{
					for( int j = n-win_size; j <= n+win_size; ++j )
					{
						if( (i-m)*(i-m)+(j-n)*(j-n) > win_size*win_size )continue;
						int pp = i*XSize + j;
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{	
							pixelcounted += 1.0;
							double ppb = image0_hsv[pp*CHANNELS+0];
							double ppg = image0_hsv[pp*CHANNELS+1];
							double ppr = image0_hsv[pp*CHANNELS+2];
							if ( fabs(pb-ppb)<t_1 && fabs(pg-ppg)<t_1 && fabs(pr-ppr)<t_1  ) pixelinbin1 += 1.0;
							if ( fabs(pb-ppb)<t_2 && fabs(pg-ppg)<t_2 && fabs(pr-ppr)<t_2  ) pixelinbin2 += 1.0;
							if ( fabs(pb-ppb)<t_3 && fabs(pg-ppg)<t_3 && fabs(pr-ppr)<t_3  ) pixelinbin3 += 1.0;
						}
					}
				}
				if( pixelcounted < minimumsupport ) win_size += WSIZE_INC;
				else break;
				if( win_size > MAX(XSize,YSize) )break;
			}while(1);
			float like1 = pixelinbin1/( pixelcounted + 1.0e-10 );
			float like2 = pixelinbin2/( pixelcounted + 1.0e-10 );
			float like3 = pixelinbin3/( pixelcounted + 1.0e-10 );
			binscoreimg0[p] = BinWght1*like1 + BinWght2*like2 + BinWght3*like3;
			
			double diff = 0;
			for( int k=0; k<CHANNELS; ++k )
			{
				double d = evolved_image1[p*CHANNELS+k] - image0[p*CHANNELS+k];
				diff += d*d;
			}
			
			F_ratio[p] = binscoreimg0[p] + WEIGHTONSHAPE*F_shape[p];
			F_error[p] = diff;
		}
		
		bandfordisplay.deallocate();
	}
	
	void calculate_F_for_disocclusion( int *labelmap, int ori_wsize, int minimumsupport )
	{
		if( ori_wsize < 15 ) ori_wsize = 15;
		// initialize ConstRegion
		std::vector<double> Vecofdiff;
		for( int p = 0; p < GridSize; ++p )
		{
			if( labelmap[p] != -1 ){ ConstRegion[p] = 0; continue; }
			int m = p/XSize;
			int n = p%XSize;
			double sum_sqr_diff = 0;
			double pixelcounted = 0;
			double pb = 0;
			double pg = 0;
			double pr = 0;
			Vecofdiff.clear();
			double temp_counted = 0;
			for( int i = m - VARWINE; i <= m + VARWINE; ++i )
				for( int j = n - VARWINE; j <= n + VARWINE; ++j )
				{
					int pp = i*XSize + j;
					if( (i-m)*(i-m)+(j-n)*(j-n) <= VARWINE*VARWINE )
					{
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && ( region_indicator[pp]>0.5 || labelmap[pp]==-1 ) )
						{
							temp_counted += 1.0;
							double ppb = image0[pp*CHANNELS+0];
							double ppg = image0[pp*CHANNELS+1];
							double ppr = image0[pp*CHANNELS+2];
							pb += ppb;
							pg += ppg;
							pr += ppr;
						}
					}
				}
			if( temp_counted == 0 ){ std::cout<<"something wrong calculate_F_for_yanchaoscheme"; while(1); }
			pb /= temp_counted;
			pg /= temp_counted;
			pr /= temp_counted;
			
			for( int i = m - VARWINE; i <= m + VARWINE; ++i )
			{
				for( int j = n - VARWINE; j <= n + VARWINE; ++j )
				{
					if( (i-m)*(i-m)+(j-n)*(j-n) > VARWINE*VARWINE )continue;
					int pp = i*XSize + j;
					if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && ( region_indicator[pp]>0.5 || labelmap[pp]==-1 ) )
					{
						pixelcounted += 1.0;
						double ppb = image0[pp*CHANNELS+0];
						double ppg = image0[pp*CHANNELS+1];
						double ppr = image0[pp*CHANNELS+2];
						double df = (ppb-pb)*(ppb-pb) + (ppg-pg)*(ppg-pg) + (ppr-pr)*(ppr-pr);
						sum_sqr_diff += df;
						Vecofdiff.push_back(-df);
					}
				}
			}
			
			int PixNotUsed = (int)( pixelcounted*0.2 ); // 
			std::nth_element( Vecofdiff.begin(), Vecofdiff.begin() + PixNotUsed, Vecofdiff.end() );
			for( int k = 0; k < PixNotUsed; ++k ) sum_sqr_diff += Vecofdiff[k];
			
			if( sum_sqr_diff/( 1e-10 + pixelcounted - PixNotUsed ) < VART ) ConstRegion[p] = 1;
			else ConstRegion[p] = 0;
			Vecofdiff.clear();
		}
        
		for( int p = 0; p < GridSize; ++p ) F_ratio[p] = F_error[p] = -100;
		double t_1 = (1.0/BinSize1) + 0.00001;
		double t_2 = (1.0/BinSize2) + 0.00001;
		double t_3 = (1.0/BinSize3) + 0.00001;
		for( int p = 0; p < GridSize; ++p )
		{
			if( labelmap[p] != -1 )continue;
			int m = p/XSize;
			int n = p%XSize;
			double pixelcounted = 0;
			double pixelinbin1 = 0;
			double pixelinbin2 = 0;
			double pixelinbin3 = 0;
			double pb = image0_hsv[p*CHANNELS+0];
			double pg = image0_hsv[p*CHANNELS+1];
			double pr = image0_hsv[p*CHANNELS+2];
			int wsize = ori_wsize;
			do{
				pixelcounted = 0;
				pixelinbin1 = 0;
				pixelinbin2 = 0;
				pixelinbin3 = 0;
				for( int i = m - wsize; i <= m + wsize; ++i )
				{
					for( int j = n - wsize; j <= n + wsize; ++j )
					{
						if( (i-m)*(i-m) + (j-n)*(j-n) > wsize*wsize )continue;
						int pp = i*XSize + j;
						if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && region_indicator[pp]>0.5 )
						{	
							pixelcounted += 1.0;
							double ppb = image0_hsv[pp*CHANNELS+0];
							double ppg = image0_hsv[pp*CHANNELS+1];
							double ppr = image0_hsv[pp*CHANNELS+2];
							if ( fabs(pb-ppb)<t_1 && fabs(pg-ppg)<t_1 && fabs(pr-ppr)<t_1  ) pixelinbin1 += 1.0;
							if ( fabs(pb-ppb)<t_2 && fabs(pg-ppg)<t_2 && fabs(pr-ppr)<t_2  ) pixelinbin2 += 1.0;
							if ( fabs(pb-ppb)<t_3 && fabs(pg-ppg)<t_3 && fabs(pr-ppr)<t_3  ) pixelinbin3 += 1.0;
						}
					}
				}
				if( pixelcounted < minimumsupport ) wsize += WSIZE_INC;
				else break;
				if( wsize > MAX(XSize, YSize) )break;
			}while(1);
			float like1 = pixelinbin1/( pixelcounted + 1.0e-10 );
			float like2 = pixelinbin2/( pixelcounted + 1.0e-10 );
			float like3 = pixelinbin3/( pixelcounted + 1.0e-10 );
			binscoreimg0[p] = BinWght1*like1 + BinWght2*like2 + BinWght3*like3;
			
			double diff = 0;
			for( int k = 0; k < CHANNELS; ++k )
			{
 				double d = evolved_image1[p*CHANNELS+k] - image0[p*CHANNELS+k];
				diff += d*d;
			}
			
			F_ratio[p] = binscoreimg0[p];
			F_error[p] = diff;
		}
        
	}
	
	int estimate_warp_and_occlusion( bool getoutoflocal, int winsize, int minimumsupport, int a, int b, int c );
	
	cv::Mat ShowColorMap( int num )
	{
		cv::Mat colorimage = cv::Mat::zeros( YSize, XSize, CV_8UC3 );
		cv::MatIterator_<cv::Vec3b> it = colorimage.begin<cv::Vec3b>();
		double vmax=-1e10; double t;
		for( int p=0; p<GridSize; ++p )
		{
			t = velocity[2*p]*velocity[2*p]+velocity[2*p+1]*velocity[2*p+1];
			t = sqrt(t);
			vmax = MAX(t,vmax);
		}
		if( vmax<1e-10 ){std::cout<<" vmax = "<<vmax<<" is too small in Show_Color"<<std::endl;vmax += 1e-10;}
		for( int p=0; p<GridSize ; ++p, ++it )
		{
			computeColor( velocity[2*p]/vmax, velocity[2*p+1]/vmax,it );
		}
#ifdef DISPLAY_ON
		cv::imshow( "velocity", colorimage );
#endif

        int bbbb = num/1000, bbb = ( num - 1000*bbbb )/100, bb = ( num - 1000*bbbb - 100*bbb )/10, b = num%10;
        char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
		std::string imgname = "instant_velocity";
		imgname = imgname+aaaa+aaa+aa+a+".jpg";

		cv::imwrite( imgname, colorimage );
		cv::waitKey(1);
		
		return colorimage;
	}
	void computeColor(double fx, double fy, cv::MatIterator_<cv::Vec3b> it)
	{
		if (ncolors == 0)makecolorwheel();

		double rad = sqrt(fx * fx + fy * fy);
		double a = atan2(-fy, -fx) / (PI);
		double fk = (a + 1) / 2 * (ncolors-1);
		int k0 = (int)fk;
		int k1 = (k0 + 1) % ncolors;
		double f = fk - k0;
		//f = 0; // uncomment to see original color wheel
		for (int b = 0; b < 3; ++b )
		{
			double col0 = colorwheel[k0][b] / 255.0;
			double col1 = colorwheel[k1][b] / 255.0;
			double col = (1 - f) * col0 + f * col1;
			if (rad <= 1.0)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= 0.75; // out of range
			(*it)[2-b] = (int)(255.0 * col);
		}
	}
	void makecolorwheel()
	{
		// relative lengths of color transitions:
		// these are chosen based on perceptual similarity
		// (e.g. one can distinguish more shades between red and yellow 
		//  than between yellow and green)
		int RY = 15;
		int YG = 6;
		int GC = 4;
		int CB = 11;
		int BM = 13;
		int MR = 6;
		ncolors = RY + YG + GC + CB + BM + MR;
		//printf("ncols = %d\n", ncols);
		if (ncolors > 60)
		{
			std::cout<<"ncolors larger than MAXCOLS(makecolorwheel deformation)..";
			while(1);
		}
		int i;
		int k = 0;
		for (i = 0; i < RY; i++) setcols(255          ,	255*i/RY     , 0            , k++ );
		for (i = 0; i < YG; i++) setcols(255-255*i/YG , 255          , 0            , k++ );
		for (i = 0; i < GC; i++) setcols(0            ,	255          , 255*i/GC     , k++ );
		for (i = 0; i < CB; i++) setcols(0            ,	255-255*i/CB , 255          , k++ );
		for (i = 0; i < BM; i++) setcols(255*i/BM     ,	0            , 255          , k++ );
		for (i = 0; i < MR; i++) setcols(255          ,	0            , 255-255*i/MR , k++ );
	}
	void setcols( int r, int g, int b, int k )
	{
		colorwheel[k][0] = r;
		colorwheel[k][1] = g;
		colorwheel[k][2] = b;
	}
	void show_occlusion_map()
	{
		cv::Mat colorimage = cv::Mat::zeros( YSize, XSize, CV_8UC3 );
		cv::MatIterator_<cv::Vec3b> it = colorimage.begin<cv::Vec3b>();
		for( int p=0; p<GridSize; ++p, ++it )
		{
			if( occlusion_map[p] ) (*it)[0]=(*it)[1]=(*it)[2]=255;
			else (*it)[0]=(*it)[1]=(*it)[2]=0;
		}
#ifdef DISPLAY_ON
		cv::imshow( "occlusion", colorimage );
#endif

		char a = '0'+KthRegion;
		std::string imgname = "occlusion";
		imgname = imgname+a+".png";
		cv::imwrite(imgname, colorimage );
		cv::waitKey(1);
		colorimage.release();
	}
	cv::Mat showimage( double *img, std::string imname, int num )
	{
		cv::Mat colorimage = cv::Mat::zeros( YSize, XSize, CV_8UC3 );
		cv::MatIterator_<cv::Vec3b> it = colorimage.begin<cv::Vec3b>();
		for( int p=0; p<GridSize; ++p, ++it )
		{
			for( int k=0; k<CHANNELS; ++k )
			{
				float a = img[p*CHANNELS+k];
				(*it)[k] = a<0? 0: ( a>1? 255: (uchar)(a*255.0) );
			}
		}
#ifdef DISPLAY_ON
		cv::imshow( imname, colorimage );
#endif

        int bbbb = num/1000, bbb = ( num - 1000*bbbb )/100, bb = ( num - 1000*bbbb - 100*bbb )/10, b = num%10;
        char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
		imname = imname+aaaa+aaa+aa+a+".png";

		cv::imwrite( imname, colorimage );
		cv::waitKey(1);
		return colorimage;
	}
	cv::Mat showwarp( float *warp, std::string wpname, int num )
	{
		double *tempv = new double[GridSize*2];

		for( int p=0; p<GridSize; ++p )
		{
			int x_ori = p%XSize;
			int y_ori = p/XSize;
			tempv[2*p] = warp[2*p]-x_ori;
			tempv[2*p+1] = warp[2*p+1]-y_ori;
		}
		cv::Mat colorimage = cv::Mat::zeros( YSize, XSize, CV_8UC3 );
		cv::MatIterator_<cv::Vec3b> it = colorimage.begin<cv::Vec3b>();
		double vmax=-1e10; double t;
		for( int p=0; p<GridSize; ++p )
		{
			t = tempv[2*p]*tempv[2*p]+tempv[2*p+1]*tempv[2*p+1];
			t = sqrt(t);
			vmax = MAX(t,vmax);
		}
		if( vmax<1e-10 ){std::cout<<" vmax = "<<vmax<<" is too small in Show_Color"<<std::endl; vmax += 1e-10; }
		for( int p=0; p<GridSize ; ++p, ++it )
		{
			computeColor( tempv[2*p]/vmax, tempv[2*p+1]/vmax, it );
		}
#ifdef DISPLAY_ON
		cv::imshow( wpname, colorimage );
#endif

        int bbbb = num/1000, bbb = ( num - 1000*bbbb )/100, bb = ( num - 1000*bbbb - 100*bbb )/10, b = num%10;
        char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
		char rrr = '0'+KthRegion;
		wpname = wpname+"R"+rrr+'_'+aaaa+aaa+aa+a+".jpg";

		cv::imwrite( wpname, colorimage );
		cv::waitKey(1);

		delete[] tempv; tempv=0;

		return colorimage;
	}
	void showContour( int num, std::string imgname, cv::Mat background_image, uchar b=0, uchar g=255, uchar r=0 )
	{
		narrowband bandfordisplay;
		bandfordisplay.allocate(XSize, YSize);
		
		double *reg1 = new double[GridSize];
		double *reg2 = new double[GridSize];
		for( int p=0; p<GridSize; ++p ){ reg1[p]=region_indicator[p]; reg2[p]=0; }
		get_evolved_image( reg1, backward_map, reg2, 1 );
		for( int p=0; p<GridSize; ++p ){ bandfordisplay.Psi[p] = reg2[p]>0.5? -1.0 : 1.0; }
		delete[] reg1; reg1=0;
		delete[] reg2; reg2=0;

		bandfordisplay.createBand();
		cv::Mat tempimage;
		background_image.copyTo(tempimage);
		cv::MatIterator_<cv::Vec3b> it = tempimage.begin<cv::Vec3b>();
		for( int *ptr=bandfordisplay.band, p=*ptr; ptr!=bandfordisplay.tail; p=*++ptr )
		{
			(*(it+p))[0]=b;
			(*(it+p))[1]=g;
			(*(it+p))[2]=r;
		}
		for( int *ptr=bandfordisplay.nhbrband, p=*ptr; ptr!=bandfordisplay.nhbrtail; p=*++ptr )
		{
			(*(it+p))[0]=b;
			(*(it+p))[1]=g;
			(*(it+p))[2]=r;
		}
#ifdef DISPLAY_ON		
		cv::imshow(imgname,tempimage);
#endif
		cv::imwrite("tempcontour.png",tempimage);
		cv::waitKey(1);
		tempimage.release();
		
		bandfordisplay.deallocate();
	}
	void residueblur( float *src, float *des, int wsize, float *regicator, bool outlier_detection = false )
	{
		#define GWIN(i,j) gwin[(i)*wsize+(j)]
	
		if( (wsize%2) != 1 ){ std::cout<<"window size should be odd number"; while(1); }
		int hsize = wsize/2;
		double *g=new double[wsize], *gwin=new double[wsize*wsize];
		double sig = 2.0/(double)(wsize*wsize);
		double sum=0;
		for( int k=0, diff=-hsize; k<wsize; ++k, ++diff ) g[k] = exp(-sig*diff*diff );
		for( int i=0; i<wsize; ++i )
		{
			for( int j=0; j<wsize; ++j )
			{
				GWIN(i,j) = g[i]*g[j];			
				sum += GWIN(i,j);
			}
		}
		sum = 1.0/sum;
		for( int k=0; k<wsize*wsize; ++k )gwin[k] *= sum;
		
        int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
		{
            int row,col,pos;
            double total;
            double g_sum;
			if( regicator[p]<0.5 ){ des[p]=0; continue; }
			if( outlier_detection )if( src[p]>90 ){ des[p]=src[p]; continue; }
			row=(int)(p/XSize)-hsize; col=p%XSize-hsize;
			total=0;
			g_sum=0;
			for( int m=row,i=0; i<wsize; ++m,++i )
			{
				for( int n=col,j=0; j<wsize; ++n,++j )
				{
					pos = m*XSize+n;
					if( (m>=0) && (m<YSize) && (n>=0) && (n<XSize) && regicator[pos]>0.5 )
					{
						if( src[pos]>90 ) continue;
						total += GWIN(i,j)*src[pos];
						g_sum += GWIN(i,j);
					}
				}
			}
			des[p] = total/g_sum;
		}
			
		#undef GWIW
		delete[] g; g=0;
		delete[] gwin; gwin=0;
	}
	void get_out_of_local( )
	{
		int optimal_x = -100000;
		int optimal_y = -100000;
		double minimum_error = FLT_MAX;
        
        int NumofSamples = 0;
        for( int i = - LOCAL_WINDOW; i <= LOCAL_WINDOW; i += SAMPLE_SIZE ) NumofSamples += 1;
        double *myerrors = new double[NumofSamples*NumofSamples];
        int *opt_x = new int[NumofSamples*NumofSamples];
        int *opt_y = new int[NumofSamples*NumofSamples];
        for( int k = 0; k < NumofSamples*NumofSamples; ++k ){ myerrors[k] = FLT_MAX; opt_x[k] = -100000; opt_y[k] = -100000; }
        
        int j=0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(j)
#endif
		for( int i = - LOCAL_WINDOW; i <= LOCAL_WINDOW; i += SAMPLE_SIZE )
		{
			for( j = - LOCAL_WINDOW; j <= LOCAL_WINDOW; j += SAMPLE_SIZE )
			{
				double average_error = 0;
				double pixel_counted = 0;
				for( int p = 0; p < GridSize; ++p )
				{
					if( region_indicator[p] < 0.5 ) continue;
					int m = p/XSize; // row
					int n = p%XSize; // col
					m += i;
					n += j;
					if( m >= 0 && m <= YSize-1 && n >= 0 && n <= XSize-1 )
					{
						pixel_counted += 1.0;
						int pp = m*XSize + n;
						for( int k = 0; k < CHANNELS; ++k )
						{
							double diff = image0[p*CHANNELS+k] - image1[pp*CHANNELS+k];
							average_error += diff*diff;
						}
					}
				}
				if( pixel_counted < 1 )continue;
                average_error /= pixel_counted;
                
                int ii = (i+LOCAL_WINDOW)/SAMPLE_SIZE;
                int jj = (j+LOCAL_WINDOW)/SAMPLE_SIZE;
                
                myerrors[ ii*NumofSamples + jj ] = average_error;
                opt_y[ ii*NumofSamples + jj ] = i;
                opt_x[ ii*NumofSamples + jj ] = j;
			}
		}
        
        for( int k = 0; k < NumofSamples*NumofSamples; ++k )
            if( myerrors[k] < minimum_error ){ minimum_error = myerrors[k]; optimal_y = opt_y[k]; optimal_x = opt_x[k]; }
        
        if( abs(optimal_x) > 0.5*REGION_WIDTH ) optimal_x = optimal_x/abs(optimal_x)*0.5*REGION_WIDTH;
        if( abs(optimal_y) > 0.5*REGION_HEIGHT ) optimal_y = optimal_y/abs(optimal_y)*0.5*REGION_HEIGHT;
        
        std::cout<<"optx: "<<optimal_x<<" opty: "<<optimal_y<<std::endl;
        
        if( optimal_x == -100000 || optimal_y == -100000 ){ std::cout<<"wrong! get_out_of_local"<<std::endl; while(1); }
		for( int p = 0; p < GridSize; ++p )
		{
			velocity[2*p+0] = -optimal_x;
			velocity[2*p+1] = -optimal_y;
		}
        
        update_backward_map( );
        update_forward_map( );
        get_evolved_image( image0, backward_map, evolved_image0 );
		get_evolved_image( image1, forward_map, evolved_image1 );
        update_occlusion_map( minimum_error );
        
        delete[] myerrors; myerrors = 0;
        delete[] opt_x; opt_x = 0;
        delete[] opt_y; opt_y = 0;
	}	
};

#endif