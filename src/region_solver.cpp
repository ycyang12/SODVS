#include "region_solver.h"
#define SAFE_BORDER 4
bool MOTION_ZERO;

const float region_solver::BinSize1 = 10;
const float region_solver::BinSize2 = 20;
const float region_solver::BinSize3 = 30;
const float region_solver::BinWght1 = 0.3;
const float region_solver::BinWght2 = 0.4;
const float region_solver::BinWght3 = 0.3;
const int region_solver::SHAPEWINE = 12;
const int region_solver::VARWINE = 12;
const float region_solver::VART = 0.002;
const int region_solver::LOCAL_WINDOW = 30;
const int region_solver::SAMPLE_SIZE = 2;
const int region_solver::SMOOTHNESS_OF_OCC = 5;

const int region_solver::BlurSize = 3;
const float region_solver::SIGX = 1.0;

int region_solver::allocate( cv::Mat img0, cv::Mat img1, int *labelmap, int label, float &regionLen, float &regionSize )
{
	if( img0.empty() || img1.empty() ) return 0;
	read_parameters();
	XSize = img0.cols;
	YSize = img0.rows;
	GridSize = XSize*YSize;
	KthRegion = label;
	moved_img = img1;

	if( !( image0			= new double[GridSize*CHANNELS] ) ||
		!( image0_hsv		= new double[GridSize*CHANNELS] ) ||
		!( evolved_image0	= new double[GridSize*CHANNELS] ) ||
		!( image1			= new double[GridSize*CHANNELS] ) ||
		!( evolved_image1	= new double[GridSize*CHANNELS] ) ||
		!( F_ratio			= new float[GridSize]			) ||
		!( F_error			= new float[GridSize]			) ||
		!( lookback			= new int[GridSize]				) ||
		!( F_shape			= new float[GridSize]			) ||
		!( binscoreimg0		= new float[GridSize]			) ||
		!( velocity			= new double[GridSize*2]		) ||
		!( velocity_outside = new double[GridSize*2]		) ||
		!( region_indicator = new float[GridSize]			) ||
		!( initial_indicator= new char[GridSize]			) ||
		!( backward_map		= new float[GridSize*2]			) ||
		!( forward_map		= new float[GridSize*2]			) ||
		!( occlusion_map	= new uchar[GridSize]			) ||
		!( grad_image1		= new double[GridSize*CHANNELS*2] ) ||
		!( det_backward_map = new double[GridSize]			) ||
		!( grad_backward_map = new double[GridSize*4]		) ||
		!( cgm_Ax			= new double[GridSize*2]		) ||
		!( cgm_b			= new double[GridSize*2]		) ||
		!( residue			= new double[GridSize*2]		) ||
		!( pbase			= new double[GridSize*2]		) ||
		!( ConstRegion		= new int[GridSize]				) ||
		!( edgebits			= new uchar[GridSize] )  )
	{
		std::cout<<"allocate failed...";
		deallocate();
		return 0;
	}
    
    BlocksDivision();
    
	initialize( img0, img1, labelmap, label, regionLen, regionSize );
	return 1;
}

void region_solver::deallocate()
{
	delete[] image0; image0 = 0;
	delete[] image0_hsv; image0_hsv = 0;
	delete[] evolved_image0; evolved_image0 = 0;
	delete[] image1; image1 = 0;
	delete[] evolved_image1; evolved_image1 = 0;
	delete[] F_ratio; F_ratio = 0;
	delete[] F_error; F_error = 0;
	delete[] lookback; lookback = 0;
	delete[] F_shape; F_shape = 0;
	delete[] binscoreimg0; binscoreimg0=0;
	delete[] velocity; velocity = 0;
	delete[] velocity_outside; velocity_outside = 0;
	delete[] region_indicator; region_indicator = 0;
	delete[] initial_indicator; initial_indicator = 0;
	delete[] backward_map; backward_map = 0;
	delete[] forward_map; forward_map = 0;
	delete[] occlusion_map; occlusion_map = 0;
	delete[] grad_image1; grad_image1 = 0;
	delete[] det_backward_map; det_backward_map = 0;
	delete[] grad_backward_map; grad_backward_map = 0;
	delete[] cgm_Ax; cgm_Ax = 0;
	delete[] cgm_b; cgm_b = 0;
	delete[] residue; residue = 0;
	delete[] pbase; pbase = 0;
	delete[] ConstRegion; ConstRegion = 0;
	delete[] edgebits; edgebits = 0;
    
    delete[] blockhead; blockhead = 0;
    delete[] blocktail; blocktail = 0;
    
	moved_img.release();
}

void region_solver::read_parameters( )
{
	std::ifstream myFile( "a_list_of_params.txt", std::ifstream::in );
	char inputline [100];
	if (myFile.good())
	{
		while (!myFile.eof()) 
		{
			myFile.getline(inputline,100);
			sscanf (inputline,"BETA0=%f", &BETA0);
		}
		std::cout<<"BinSize1= "<<BinSize1<<std::endl;
		std::cout<<"BinSize2= "<<BinSize2<<std::endl;
		std::cout<<"BinSize3= "<<BinSize3<<std::endl;
		std::cout<<"BinWght1= "<<BinWght1<<std::endl;
		std::cout<<"BinWght2= "<<BinWght2<<std::endl;
		std::cout<<"BinWght3= "<<BinWght3<<std::endl;
		std::cout<<"SHAPEWINE= "<<SHAPEWINE<<std::endl;
		std::cout<<"VARWINE= "<<VARWINE<<std::endl;
		std::cout<<"VART= "<<VART<<std::endl;
		std::cout<<"LOCAL_WINDOW= "<<LOCAL_WINDOW<<std::endl;
		std::cout<<"SAMPLE_SIZE= "<<SAMPLE_SIZE<<std::endl;
		std::cout<<"BETA0= "<<BETA0<<std::endl;
		std::cout<<"SMOOTHNESS_OF_OCC= "<<SMOOTHNESS_OF_OCC<<std::endl;
		std::cout<<"BlurSize= "<<BlurSize<<std::endl;
		std::cout<<"SIGX= "<<SIGX<<std::endl;
		
		myFile.close();
	}
	else
	{
		std::cout <<"ERROR: can't open file."<<std::endl;
		while(1);
	}
	return;
}

void region_solver::initialize( cv::Mat img0, cv::Mat img1, int *labelmap, int label, float &regionLen, float &regionSize  )
{
	markEdges( );
	imgTo1darray( img0, image0 );
	cv::MatIterator_<cv::Vec3b> itimg0 = img0.begin<cv::Vec3b>();

	cv::Mat HSVimg = cv::Mat::zeros( YSize, XSize, CV_32FC3 );
	cv::MatIterator_<cv::Vec3f> ithsv = HSVimg.begin<cv::Vec3f>();
	for( int p = 0; p < GridSize; ++p )
	{
		for( int k = 0; k < CHANNELS; ++k )
		{
			(*(ithsv+p))[k]	= (float)(*(itimg0+p))[k]/255.0;
		}
	}
	cv::cvtColor( HSVimg, HSVimg, CV_BGR2HSV );
	
	for( int p = 0; p < GridSize; ++p, ++ithsv )
	{
		int pixel = p*CHANNELS; 
		for( int k = 0; k < CHANNELS; ++k )
		{
			if( k==0 )image0_hsv[pixel+k] = (*ithsv)[k]/360.0;
			else image0_hsv[pixel+k] = (*ithsv)[k];
            if( image0_hsv[pixel+k]<0 || image0_hsv[pixel+k]>1 )
            {std::cout<<"wrong range, hsv"<<std::endl; while(1);}
		}
	}
	HSVimg.release();

	imgTo1darray( img1, image1 );
	compute_gradients( image1, grad_image1 );
	compute_region_indicator( labelmap, label, regionLen, regionSize );
	for( int p=0; p<GridSize; ++p )initial_indicator[p] = region_indicator[p]>0.5 ? 1 : 0;
	initialize_warp( backward_map );
	initialize_warp( forward_map );
	compute_gradients_of_warp( backward_map, grad_backward_map );
	compute_det_of_gradient_of_warp( grad_backward_map, det_backward_map );
	for( int p=0; p<GridSize; ++p ) occlusion_map[p]=0;
	for( int p=0; p<2*GridSize; ++p ) velocity[p]=velocity_outside[p]=0;
	get_evolved_image( image0, backward_map, evolved_image0 );
	get_evolved_image( image1, forward_map, evolved_image1 );
	
	for( int p=0; p<GridSize*2; ++p ) cgm_Ax[p]=cgm_b[p]=residue[p]=pbase[p]=0;

	for( int p=0; p<GridSize; ++p )binscoreimg0[p]=0;
    
	distancemap mydist;
	int size[2]= { XSize, YSize };
	float *potential = new float[GridSize];
	double *tempsi = new double[GridSize];
	int *LOC = new int[GridSize];
	int Length = 0;
	bool *indis = new bool[GridSize];
	for( int p=0; p<GridSize; ++p ){ potential[p] = 1.0; lookback[p]  = 1000*1000; tempsi[p] = 1.0; }
	mydist.allocate( 2, size, potential );
    
	narrowband bandfordisplay;
	bandfordisplay.allocate( XSize, YSize );
	for( int p = 0; p < GridSize; ++p ){ bandfordisplay.Psi[p] = region_indicator[p]>0.5? -1.0 : 1.0; }
    bandfordisplay.createBand();
	bandfordisplay.createContour();
//	for( int *ptr = bandfordisplay.contour,		p=*ptr;		ptr != bandfordisplay.contail;  p=*++ptr ) lookback[p] = -1;
	for( int *ptr = bandfordisplay.band,		p = *ptr;	ptr != bandfordisplay.tail;		p = *++ptr )lookback[p] = -1;
	for( int *ptr = bandfordisplay.edgeband,	p = *ptr;	ptr != bandfordisplay.edgetail; p = *--ptr )lookback[p] = -1;
	mydist.findDistanceMap(  tempsi, LOC, Length, lookback, indis );
	bandfordisplay.deallocate();

	delete[] tempsi; tempsi = 0;
	delete[] LOC; LOC = 0;
	delete[] indis; indis = 0;
	delete[] potential; potential = 0;
	mydist.deallocate();
}

void region_solver::initialize2( int *labelmap, int label, float &regionLen, float &regionSize )
{
	compute_region_indicator( labelmap, label, regionLen, regionSize );
	compute_gradients_of_warp( backward_map, grad_backward_map );
	compute_det_of_gradient_of_warp( grad_backward_map, det_backward_map );
	for( int p=0; p<2*GridSize; ++p ) velocity[p]=velocity_outside[p]=0;
	get_evolved_image( image0, backward_map, evolved_image0 );
	get_evolved_image( image1, forward_map, evolved_image1 );
	for( int p=0; p<GridSize*2; ++p ) cgm_Ax[p]=cgm_b[p]=residue[p]=pbase[p]=0;
}

void region_solver::markEdges()
{
	int p;
	for( p=0; p<GridSize; ++p )edgebits[p]=0;
	//Y faces
	for (p=0; p<XSize; p+=1) 
	{
		edgebits[p]|=YNEG;
		edgebits[GridSize-1-p]|=YPOS;
	}
	//X faces
	for (p=0; p<GridSize; p+=XSize) 
	{
		edgebits[p]|=XNEG;
		edgebits[XSize-1+p]|=XPOS;
	}
}

void region_solver::imgTo1darray( cv::Mat imgsrc, double *image )
{
	cv::Mat img;
	imgsrc.copyTo(img);
	cv::GaussianBlur( imgsrc, img, cv::Size(BlurSize,BlurSize), SIGX );

	if( img.channels()!=3 ){
		std::cout<<"channels of image is not 3...";
		while(1);
	}
	cv::MatIterator_<cv::Vec3b> itimg;
	itimg = img.begin<cv::Vec3b>();
	for ( int p=0; p<GridSize; ++p, ++itimg )
	{
		int pixel = p*CHANNELS; 
		for( int k=0; k<CHANNELS; ++k )
		{
			image[pixel+k] = (double)((*itimg)[k])/255.0;
		}
	}
	img.release();
}

void region_solver::compute_gradients( double *img, double *grad_img )
{
	#define X 1
	#define Y XSize
	#define K int k=0; k<CHANNELS; ++k
	#define IM(p,k) img[(p)*CHANNELS+k]
	#define GX(p,k) grad_img[(p)*CHANNELS*2+2*k]
	#define GY(p,k) grad_img[(p)*CHANNELS*2+2*k+1]

	for( int k=0; k<GridSize*CHANNELS*2; ++k )grad_img[k] = 0;
	for( int p=0; p<GridSize; ++p )
	{
		if( edgebits[p]&XNEG ) for(K) GX(p,k) = IM(p+X,k) - IM(p,k);
		else if( edgebits[p]&XPOS ) for(K) GX(p,k) = IM(p,k) - IM(p-X,k);
		else for(K) GX(p,k) = ( IM(p+X,k) - IM(p-X,k) )/2.0;

		if( edgebits[p]&YNEG ) for(K) GY(p,k) = IM(p+Y,k) - IM(p,k);
		else if( edgebits[p]&YPOS ) for(K) GY(p,k) = IM(p,k) - IM(p-Y,k);
		else for(K) GY(p,k) = ( IM(p+Y,k) - IM(p-Y,k) )/2.0;
	}

	#undef X
	#undef Y
	#undef K
	#undef IM
	#undef GX
	#undef GY
}

void region_solver::compute_region_indicator( int *labelmap, int label, float &regionLen, float &regionSize )
{
	float minX = FLT_MAX;
	float maxX = -1;
	float minY = FLT_MAX;
	float maxY = -1;
	regionSize = 0;
	for( int p = 0; p < GridSize; ++p )
	{
		region_indicator[p] = labelmap[p]==label? 1.0 : 0;
		if( region_indicator[p]>0.5 )
		{
			int x = p%XSize;
			int y = p/XSize;
			if( x<minX ) minX = x;
			if( x>maxX ) maxX = x;
			if( y<minY ) minY = y;
			if( y>maxY ) maxY = y;
			regionSize += 1.0;
		}
	}
	regionLen = sqrt( (maxX-minX)*(maxX-minX)+(maxY-minY)*(maxY-minY) );
    
    REGION_WIDTH = maxX-minX;
    REGION_HEIGHT = maxY-minY;
}

void region_solver::initialize_warp( float *warp )
{
	for( int p=0; p<GridSize; ++p )
	{
		int i=p/XSize;
		int j=p%XSize;
		warp[2*p] = j;
		warp[2*p+1] = i;
	}
}

void region_solver::compute_gradients_of_warp( float *warp, double *grad_warp )
{
	#define X 1
	#define Y XSize
	#define K int k=0; k<2; ++k
	#define WP(p,k) warp[(p)*2+k]
	#define GX(p,k) grad_warp[(p)*4+2*k]
	#define GY(p,k) grad_warp[(p)*4+2*k+1]

	//double *reg1 = new double[GridSize];
	//double *reg2 = new double[GridSize];
	//for( int p=0; p<GridSize; ++p ){ reg1[p]=region_indicator[p]; reg2[p]=0; }
	//get_evolved_image( reg1, backward_map, reg2, 1 );
	//delete[] reg1; reg1=0;
	
	for( int p=0; p<GridSize; ++p )
	{
//		for(K) GX(p,k)=GY(p,k)=0;
//		if( reg2[p]>0.5 )
//		{
//			if( edgebits[p]&XNEG )
//			{
//				if( reg2[p+X]>0.5 )for(K) GX(p,k) = WP(p+X,k) - WP(p,k);
//			}
//			else if( edgebits[p]&XPOS ) 
//			{
//				if( reg2[p-X]>0.5 )for(K) GX(p,k) = WP(p,k) - WP(p-X,k);
//			}
//			else
//			{
//				if( reg2[p+X]>0.5 )
//				{
//					if( reg2[p-X]>0.5 ) for(K) GX(p,k) = ( WP(p+X,k) - WP(p-X,k) )/2.0;
//					else for(K) GX(p,k) = ( WP(p+X,k) - WP(p,k) );
//				}
//				else if( reg2[p-X]>0.5 )
//				{
//					for(K) GX(p,k) = ( WP(p,k) - WP(p-X,k) );
//				}
//			}
//			
//			if( edgebits[p]&YNEG )
//			{
//				if( reg2[p+Y]>0.5 )for(K) GY(p,k) = WP(p+Y,k) - WP(p,k);
//			}
//			else if( edgebits[p]&YPOS )
//			{
//				if( reg2[p-Y]>0.5 )for(K) GY(p,k) = WP(p,k) - WP(p-Y,k);
//			}
//			else
//			{
//				if( reg2[p+Y]>0.5 )
//				{
//					if( reg2[p-Y]>0.5 ) for(K) GY(p,k) = ( WP(p+Y,k) - WP(p-Y,k) )/2.0;
//					else for(K) GY(p,k) = ( WP(p+Y,k) - WP(p,k) );
//				}
//				else if( reg2[p-Y]>0.5 )
//				{
//					for(K) GY(p,k) = ( WP(p,k) - WP(p-Y,k) );
//				}
//			}
//		}
//		else
//		{
			if( edgebits[p]&XNEG ) for(K) GX(p,k) = WP(p+X,k) - WP(p,k);
			else if( edgebits[p]&XPOS ) for(K) GX(p,k) = WP(p,k) - WP(p-X,k);
			else for(K) GX(p,k) = ( WP(p+X,k) - WP(p-X,k) )/2.0;

			if( edgebits[p]&YNEG ) for(K) GY(p,k) = WP(p+Y,k) - WP(p,k);
			else if( edgebits[p]&YPOS ) for(K) GY(p,k) = WP(p,k) - WP(p-Y,k);
			else for(K) GY(p,k) = ( WP(p+Y,k) - WP(p-Y,k) )/2.0;
//		}
	}

	//delete[] reg2; reg2=0;
	#undef X
	#undef Y
	#undef K
	#undef WP
	#undef GX
	#undef GY
}

void region_solver::compute_det_of_gradient_of_warp( double *grad_warp, double *det_grad_warp )
{
	#define GX(p,k) grad_warp[(p)*4+2*k]
	#define GY(p,k) grad_warp[(p)*4+2*k+1]
	
    int p=0;
    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
        for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p ) det_grad_warp[p] = ( GX(p,0)*GY(p,1) - GX(p,1)*GY(p,0) );
	
	#undef GX
	#undef GY
}
 
void region_solver::get_evolved_image( double *img, float *warp, double *evolved_img, int pass )
{
	if( pass==0 )
	{
        int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
		{
			double x0 = warp[2*p];
			double y0 = warp[2*p+1];
			if( x0 < -SAFE_BORDER || x0 > XSize+SAFE_BORDER || y0 < -SAFE_BORDER || y0 > YSize+SAFE_BORDER )
			{
				for( int k=0; k<CHANNELS; ++k )evolved_img[p*CHANNELS+k]=-10;
				continue;
			}
			if( x0 < 0 ) x0 = 0;
			if( x0 > XSize-1 ) x0 = XSize-1;
			if( y0 < 0 ) y0 = 0;
			if( y0 > YSize-1 ) y0 = YSize-1;
			int x_lw = (int)floor(x0);
			int y_lw = (int)floor(y0);
			int x_up = x_lw + 1; x_up = x_up>XSize-1 ? XSize-1:x_up;
			int y_up = y_lw + 1; y_up = y_up>YSize-1 ? YSize-1:y_up;

			if( x_lw <0 || x_lw > XSize -1 || y_lw <0 || y_lw > YSize -1 || x_up <0 || x_up > XSize -1 || y_up <0 || y_up > YSize -1 )
			{
				std::cout<<"p: "<<p<<" x0: "<<x0<<" y0: "<<y0<<std::endl;
				std::cout<<"x_lw : "<<x_lw<<std::endl;
				std::cout<<"y_lw : "<<y_lw<<std::endl;
				std::cout<<"x_up : "<<x_up<<std::endl;
				std::cout<<"y_up : "<<y_up<<std::endl;
				while(1);
			}

			double aa,bb,cc,dd,alpha,beta;
			alpha = x0-x_lw;
			beta  = y0-y_lw;
			double alpha_1 = (double)(1.0 - alpha);
			for( int k=0; k<CHANNELS; ++k )
			{
				aa = img[ (x_lw + y_lw*XSize)*CHANNELS + k ];
				bb = img[ (x_up + y_lw*XSize)*CHANNELS + k ];
				cc = img[ (x_lw + y_up*XSize)*CHANNELS + k ];
				dd = img[ (x_up + y_up*XSize)*CHANNELS + k ];
				evolved_img[p*CHANNELS+k] = alpha*(bb-aa) + aa + beta*( alpha_1*(cc-aa) + alpha*(dd-bb) );
			}
		}
	}
	else
	{
        int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
		{
			double x0 = warp[2*p];
			double y0 = warp[2*p+1];
			if( x0 < -SAFE_BORDER || x0 > XSize+SAFE_BORDER || y0 < -SAFE_BORDER || y0 > YSize+SAFE_BORDER )
			{
				evolved_img[p]=-10;
				continue;
			}
			if( x0 < 0 ) x0 = 0;
			if( x0 > XSize-1 ) x0 = XSize-1;
			if( y0 < 0 ) y0 = 0;
			if( y0 > YSize-1 ) y0 = YSize-1;
			int x_lw = (int)floor(x0);
			int y_lw = (int)floor(y0);
			int x_up = x_lw + 1; x_up = x_up>XSize-1 ? XSize-1:x_up;
			int y_up = y_lw + 1; y_up = y_up>YSize-1 ? YSize-1:y_up;
			double aa,bb,cc,dd,alpha,beta;
			alpha = x0-x_lw;
			beta  = y0-y_lw;
			double alpha_1 = (double)(1.0 - alpha);
			
			aa = img[ (x_lw + y_lw*XSize) ];
			bb = img[ (x_up + y_lw*XSize) ];
			cc = img[ (x_lw + y_up*XSize) ];
			dd = img[ (x_up + y_up*XSize) ];
			evolved_img[p] = alpha*(bb-aa) + aa + beta*( alpha_1*(cc-aa) + alpha*(dd-bb) );
		}
	}
}

void region_solver::get_cgm_b_using_region_based_metric( double *b, bool *evolved_regicator, bool *evolved_occmap, int pass )
{
	#define GX(p,k) grad_image1[(p)*CHANNELS*2+2*k]
	#define GY(p,k) grad_image1[(p)*CHANNELS*2+2*k+1]

    if( pass == -1 )
    {
        avg_G[0]=0;
        avg_G[1]=0;
        double pix_in = 0;
        for( int p = 0; p < GridSize; ++p )
        {
            b[2*p]   = 0;
            b[2*p+1] = 0;
            if( evolved_regicator[p] )
            {
                pix_in += 1.0;
                if( !evolved_occmap[p] )
                {
                    
                    int ppp = p*CHANNELS;
                    double d0 = ( image1[ppp+0] - evolved_image0[ppp+0]) * det_backward_map[p];
                    double d1 = ( image1[ppp+1] - evolved_image0[ppp+1]) * det_backward_map[p];
                    double d2 = ( image1[ppp+2] - evolved_image0[ppp+2]) * det_backward_map[p];

                    b[2*p]   = GX(p,0)*d0 + GX(p,1)*d1 + GX(p,2)*d2;
                    b[2*p+1] = GY(p,0)*d0 + GY(p,1)*d1 + GY(p,2)*d2;

                    avg_G[0] += b[2*p];
                    avg_G[1] += b[2*p+1];
                }
            }
        }
        avg_G[0] /= pix_in;
        avg_G[1] /= pix_in;
    }
	if( pass==0 )
	{
        double *pix_in = new double[NumofBlobs];
        for( int i=0; i<NumofBlobs; ++i )
        {
            BlobTrans[2*i+0] = 0;
            BlobTrans[2*i+1] = 0;
            pix_in[i] = 0;
        }
        for( int p=0; p<GridSize; ++p )
		{
			b[2*p+0] = 0;
			b[2*p+1] = 0;
			if( evolved_regicator[p] )
			{
				pix_in[blobindex[p]] += 1.0;
				if( !evolved_occmap[p] )
				{
                    int ppp = p*CHANNELS;
                    double d0 = (image1[ppp+0] - evolved_image0[ppp+0])*det_backward_map[p];
                    double d1 = (image1[ppp+1] - evolved_image0[ppp+1])*det_backward_map[p];
                    double d2 = (image1[ppp+2] - evolved_image0[ppp+2])*det_backward_map[p];

                    b[2*p+0] = GX(p,0)*d0 + GX(p,1)*d1 + GX(p,2)*d2;
                    b[2*p+1] = GY(p,0)*d0 + GY(p,1)*d1 + GY(p,2)*d2;
					
                    BlobTrans[2*blobindex[p]+0] += b[2*p+0];
					BlobTrans[2*blobindex[p]+1] += b[2*p+1];
				}
			}
		}
        for( int i=0; i<NumofBlobs; ++i )
        {
            BlobTrans[2*i+0] /= pix_in[i];
            BlobTrans[2*i+1] /= pix_in[i];
        }
		bnorm = 0;
		for( int p=0; p<GridSize; ++p )
		{
			if( evolved_regicator[p] )
			{	
				b[2*p+0] -= BlobTrans[2*blobindex[p]+0];
				b[2*p+1] -= BlobTrans[2*blobindex[p]+1];
				bnorm += b[2*p]*b[2*p] + b[2*p+1]*b[2*p+1];
			}
		}
		bnorm = sqrt(bnorm);
        
        delete[] pix_in; pix_in = 0;
	}
	if( pass == 1 )
	{
		bnorm = 0;
		for( int p=0; p<GridSize; ++p )
		{
			int	px = 2*p;
			int	py = 2*p+1;
			b[px] = 0;
			b[py] = 0;
			if( evolved_regicator[p] )
			{
				double xa = edgebits[p] & XNEG ? 0 : ( !evolved_regicator[p-1]     ? velocity[2*(p-1)]    :0 );
				double xb = edgebits[p] & YNEG ? 0 : ( !evolved_regicator[p-XSize] ? velocity[2*(p-XSize)]:0 );
				double xc = edgebits[p] & XPOS ? 0 : ( !evolved_regicator[p+1]     ? velocity[2*(p+1)]    :0 );
				double xd = edgebits[p] & YPOS ? 0 : ( !evolved_regicator[p+XSize] ? velocity[2*(p+XSize)]:0 );
				
				double ya = edgebits[p] & XNEG ? 0 : ( !evolved_regicator[p-1]     ? velocity[2*(p-1)+1]    :0 );
				double yb = edgebits[p] & YNEG ? 0 : ( !evolved_regicator[p-XSize] ? velocity[2*(p-XSize)+1]:0 );
				double yc = edgebits[p] & XPOS ? 0 : ( !evolved_regicator[p+1]     ? velocity[2*(p+1)+1]    :0 );
				double yd = edgebits[p] & YPOS ? 0 : ( !evolved_regicator[p+XSize] ? velocity[2*(p+XSize)+1]:0 );
	
				b[px] = xa+xb+xc+xd;
				b[py] = ya+yb+yc+yd;
			}
			bnorm += b[px]*b[px]+b[py]*b[py];
		}
		bnorm = sqrt(bnorm);
	}

	#undef GX
	#undef GY
}

void region_solver::get_cgm_Ax_using_region_based_metric( double *x, double *Ax, bool *evolved_regicator, int pass )
{
	int dx=2, dy=2*XSize;

	if( pass==0 )
	{
        int p = 0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
        {
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
            {
                int	px = 2*p;
                int	py = px+1;
                Ax[px] = 0;
                Ax[py] = 0;

                if( evolved_regicator[p] )
                {
/*
                    double xa = edgebits[p] & XNEG ? 0 : ( evolved_regicator[p-1]    <0? x[px-dx]-x[px] : 0 );
                    double xb = edgebits[p] & YNEG ? 0 : ( evolved_regicator[p-XSize]<0? x[px-dy]-x[px] : 0 );
                    double xc = edgebits[p] & XPOS ? 0 : ( evolved_regicator[p+1]    <0? x[px+dx]-x[px] : 0 );
                    double xd = edgebits[p] & YPOS ? 0 : ( evolved_regicator[p+XSize]<0? x[px+dy]-x[px] : 0 );
                    
                    double ya = edgebits[p] & XNEG ? 0 : ( evolved_regicator[p-1]    <0? x[py-dx]-x[py] : 0 );
                    double yb = edgebits[p] & YNEG ? 0 : ( evolved_regicator[p-XSize]<0? x[py-dy]-x[py] : 0 );
                    double yc = edgebits[p] & XPOS ? 0 : ( evolved_regicator[p+1]    <0? x[py+dx]-x[py] : 0 );
                    double yd = edgebits[p] & YPOS ? 0 : ( evolved_regicator[p+XSize]<0? x[py+dy]-x[py] : 0 );
 */
                    double xa, ya, xb, yb, xc, yc, xd, yd;
                    if( !(edgebits[p] & XNEG) && evolved_regicator[p-1]     ) { xa = x[px-dx]-x[px]; ya = x[py-dx]-x[py]; } else { xa = ya = 0; }
                    if( !(edgebits[p] & YNEG) && evolved_regicator[p-XSize] ) { xb = x[px-dy]-x[px]; yb = x[py-dy]-x[py]; } else { xb = yb = 0; }
                    if( !(edgebits[p] & XPOS) && evolved_regicator[p+1]     ) { xc = x[px+dx]-x[px]; yc = x[py+dx]-x[py]; } else { xc = yc = 0; }
                    if( !(edgebits[p] & YPOS) && evolved_regicator[p+XSize] ) { xd = x[px+dy]-x[px]; yd = x[py+dy]-x[py]; } else { xd = yd = 0; }
                    
                    Ax[px] = - 1.0*( xa+xb+xc+xd );
                    Ax[py] = - 1.0*( ya+yb+yc+yd );
                }
            }
        }
	}
	else if( pass==1 )
	{
        int p = 0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
            {
                int	px = 2*p;
                int	py = px + 1;
                Ax[px] = 0;
                Ax[py] = 0;
                if( evolved_regicator[p] )
                {
                    double xa = edgebits[p] & XNEG ? 0 : ( evolved_regicator[p-1]    ? x[px-dx]-x[px] : 0-x[px] );
                    double xb = edgebits[p] & YNEG ? 0 : ( evolved_regicator[p-XSize]? x[px-dy]-x[px] : 0-x[px] );
                    double xc = edgebits[p] & XPOS ? 0 : ( evolved_regicator[p+1]    ? x[px+dx]-x[px] : 0-x[px] );
                    double xd = edgebits[p] & YPOS ? 0 : ( evolved_regicator[p+XSize]? x[px+dy]-x[px] : 0-x[px] );
                    
                    double ya = edgebits[p] & XNEG ? 0 : ( evolved_regicator[p-1]    ? x[py-dx]-x[py] : 0-x[py] );
                    double yb = edgebits[p] & YNEG ? 0 : ( evolved_regicator[p-XSize]? x[py-dy]-x[py] : 0-x[py] );
                    double yc = edgebits[p] & XPOS ? 0 : ( evolved_regicator[p+1]    ? x[py+dx]-x[py] : 0-x[py] );
                    double yd = edgebits[p] & YPOS ? 0 : ( evolved_regicator[p+XSize]? x[py+dy]-x[py] : 0-x[py] );

                    Ax[px] = - 1.0*( xa+xb+xc+xd );
                    Ax[py] = - 1.0*( ya+yb+yc+yd );
                }
            }
	}
}

cv::Mat region_solver::solve_for_velocity_using_region_based_metric( int num )
{
	bool *evolved_region_indicator = new bool[GridSize];
	bool *evolved_occlusion_map = new bool[GridSize];
    
	double *reg1 = new double[GridSize];
	double *reg2 = new double[GridSize];
	for( int p=0; p<GridSize; ++p ){ reg1[p] = region_indicator[p]; reg2[p] = 0; }
	get_evolved_image( reg1, backward_map, reg2, 1 );
	for( int p=0; p<GridSize; ++p ){ evolved_region_indicator[p] = reg2[p]>0.5 ? true : false; }
	for( int p=0; p<GridSize; ++p )
	{
		reg1[p] = 0;
		if( evolved_region_indicator[p] )
        {
            int ppp = p*CHANNELS;
            double d0 = evolved_image0[ppp+0]-image1[ppp+0];
            double d1 = evolved_image0[ppp+1]-image1[ppp+1];
            double d2 = evolved_image0[ppp+2]-image1[ppp+2];
            reg1[p] = d0*d0 + d1*d1 + d2*d2;
        }
	}
	for( int p=0; p<GridSize; ++p ){ evolved_occlusion_map[p] = reg1[p]*GAMA_OMEGA>BETA0? true : false; }
	delete[] reg1; reg1=0;
	delete[] reg2; reg2=0;
    
    cv::Mat blobimg = cv::Mat::zeros( YSize, XSize, CV_8UC1 );//grayscale
    cv::MatIterator_<uchar> itblob = blobimg.begin<uchar>();
    for( int p=0; p<GridSize; ++p, ++itblob ) if( evolved_region_indicator[p] ) *itblob = 1;
    FindBlobs( blobimg, Myblobs );
    blobimg.release();
    NumofBlobs = (int)Myblobs.size();
    blobindex = new int[GridSize];
    BlobTrans = new double[2*NumofBlobs];
    for(  size_t i=0; i<Myblobs.size(); ++i )
        for( size_t j=0; j<Myblobs[i].size(); ++j )
        {
            int x = Myblobs[i][j].x;
            int y = Myblobs[i][j].y;
            int p = y*XSize + x;
            blobindex[p] = (int)i;
        }

	//std::cout<<"1L : get velocity in region :::::::"<<std::endl;

    
    
	double al = 0, beta = 0, rtr = 0, rtr1 = 0;
	get_cgm_b_using_region_based_metric( cgm_b, evolved_region_indicator, evolved_occlusion_map, 0 );
	get_cgm_Ax_using_region_based_metric( velocity, cgm_Ax, evolved_region_indicator, 0 );
	vector_plus( evolved_region_indicator, cgm_b, -1, cgm_Ax, residue, 0 );
	for( int i=0; i<GridSize*2; ++i ) pbase[i] = residue[i];
	rtr = vector_dotx( evolved_region_indicator, residue, residue );
	double old_maxresidue = FLT_MAX;
	int first_maxresidue=0;
    while(1)
	{
        get_cgm_Ax_using_region_based_metric( pbase, cgm_Ax, evolved_region_indicator, 0 );
		al = rtr/vector_dotx( evolved_region_indicator, pbase, cgm_Ax );
		//vector_plus( evolved_region_indicator, velocity, al, pbase, velocity, 0 );
		//if( vector_plus( evolved_region_indicator, residue, -al, cgm_Ax, residue, 1 ) )break;
        if( vector_plus_2_args( evolved_region_indicator, velocity, al, pbase, velocity, residue, -al, cgm_Ax, residue ) ) break;

        if( first_maxresidue==0 ){ ++first_maxresidue; old_maxresidue = maxresidue; }
        if( maxresidue > 3.0*old_maxresidue ){ std::cout<<"Do not CONVERGE!!!!"<<std::endl; break; }
        
		rtr1 = vector_dotx( evolved_region_indicator, residue, residue );
		beta = rtr1/rtr;
		rtr  = rtr1;
		vector_plus( evolved_region_indicator, residue, beta, pbase, pbase, 0 );
	}
    

	//std::cout<<"2L : extend the velocity :::::::"<<std::endl;
	for( int p=0; p<GridSize; ++p ) evolved_region_indicator[p] = !evolved_region_indicator[p];

	al = 0, beta = 0, rtr = 0, rtr1 = 0;
	get_cgm_b_using_region_based_metric( cgm_b, evolved_region_indicator, evolved_occlusion_map, 1 );
	get_cgm_Ax_using_region_based_metric( velocity_outside, cgm_Ax, evolved_region_indicator, 1 );
	vector_plus( evolved_region_indicator, cgm_b, -1, cgm_Ax, residue, 0 );
	for( int i=0; i<GridSize*2; ++i ) pbase[i] = residue[i];
	rtr = vector_dotx( evolved_region_indicator, residue, residue );
	while(1)
	{
		get_cgm_Ax_using_region_based_metric( pbase, cgm_Ax, evolved_region_indicator, 1 );
		al = rtr/vector_dotx( evolved_region_indicator, pbase, cgm_Ax );
		//vector_plus( evolved_region_indicator, velocity_outside, al, pbase, velocity_outside, 0 );
		//if( vector_plus( evolved_region_indicator, residue, -al, cgm_Ax, residue, 1 ) )break;
        if( vector_plus_2_args( evolved_region_indicator, velocity_outside, al, pbase, velocity_outside, residue, -al, cgm_Ax, residue) ) break;
        
        rtr1 = vector_dotx( evolved_region_indicator, residue, residue );
		beta = rtr1/rtr;
		rtr  = rtr1;
		vector_plus( evolved_region_indicator, residue, beta, pbase, pbase, 0 );
	}

	for( int p=0; p<GridSize; ++p )
	{
		if( evolved_region_indicator[p] )
		{
			velocity[2*p]   = velocity_outside[2*p];
			velocity[2*p+1] = velocity_outside[2*p+1];
		}
	}
	normalize_velocity();
    
    cv::Mat deformation_field;
#ifdef GENERATE_INTERMEDIATE_RESULT
	deformation_field = ShowColorMap(num);
#endif

    delete[] evolved_region_indicator; evolved_region_indicator = 0;
    delete[] evolved_occlusion_map; evolved_occlusion_map = 0;
    Myblobs.clear();
    delete[] blobindex; blobindex = 0;
    delete[] BlobTrans; BlobTrans = 0;

	return deformation_field;
}

void region_solver::solve_for_translation()
{
	bool *evolved_region_indicator = new bool[GridSize];
	bool *evolved_occlusion_map = new bool[GridSize];

	double *reg1 = new double[GridSize];
	double *reg2 = new double[GridSize];
	for( int p=0; p<GridSize; ++p ){ reg1[p] = region_indicator[p]; reg2[p] = 0; }
	get_evolved_image( reg1, backward_map, reg2, 1 );
	for( int p=0; p<GridSize; ++p ){ evolved_region_indicator[p] = reg2[p]>0.5? true : false; }
	for( int p=0; p<GridSize; ++p ){ reg1[p] = occlusion_map[p]; reg2[p] = 0; }
	get_evolved_image( reg1, backward_map, reg2, 1 );
	for( int p=0; p<GridSize; ++p ){ evolved_occlusion_map[p] = reg2[p]>0.5? true : false; }
	delete[] reg1; reg1=0;
	delete[] reg2; reg2=0;
    
    get_cgm_b_using_region_based_metric( cgm_b, evolved_region_indicator, evolved_occlusion_map, -1 );
	for( int p = 0; p < GridSize; ++p )
	{
		velocity[2*p+0] = avg_G[0];
		velocity[2*p+1] = avg_G[1];
	}
	normalize_velocity();
	
    delete[] evolved_region_indicator; evolved_region_indicator=0;
    delete[] evolved_occlusion_map; evolved_occlusion_map=0;
}

int region_solver::vector_plus( bool *evolved_regicator, double *v1, double al ,double *v2, double *v3, int test )
{
	if( test == 1 )
	{
		maxresidue = 0;
		if( bnorm < STOP_VALUE ){ bnorm = STOP_VALUE; std::cout<<"bnorm is too small!"<<std::endl; MOTION_ZERO = true; }
		double t = STOP_SIGN * bnorm;

        for( int p=0; p<GridSize; ++p )
		{
			if( evolved_regicator[p] )
            {
                int pp = p + p;
                v3[pp] = v1[pp] + al*v2[pp];
                v3[pp+1] = v1[pp+1] + al*v2[pp+1];
                
                double absv3 = MAX( fabs(v3[pp]), fabs(v3[pp+1]) );
                maxresidue = MAX( maxresidue, absv3 );
            }
		}
        
        int small_enough = 0;
        if( maxresidue < t ) small_enough = 1;
        
		return small_enough;
	}
	if( test == 0 )
	{
        int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
        #pragma omp parallel for private(p)
#endif
        for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
            for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
            {
                if( evolved_regicator[p] )
                {
                    int pp = p + p;
                    v3[pp] = v1[pp] + al*v2[pp];
                    v3[pp+1] = v1[pp+1] + al*v2[pp+1];
                }
            }
		return 0;
	}
	std::cout<<"illegal test value";while(1);
}

int region_solver::vector_plus_2_args( bool *evolved_regicator, double *v1, double al ,double *v2, double *v3, \
                                                                double *x1, double bl, double *x2, double *x3 )
{
    maxresidue = 0;
    if( bnorm < STOP_VALUE ){ bnorm = STOP_VALUE; std::cout<<"bnorm is too small!"<<std::endl; MOTION_ZERO = true; }
    double t = STOP_SIGN * bnorm;

    double *res = new double[NumofBlocks];
    for( int k=0; k<NumofBlocks; ++k ) res[k] = 0;
    
    int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
    #pragma omp parallel for private(p)
#endif
    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
    {
        double blockmaxres = 0;
        for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
        {
            if( evolved_regicator[p] )
            {
                int px = p+p;
                int py = px+1;
                
                v3[px] = v1[px] + al*v2[px];
                v3[py] = v1[py] + al*v2[py];
                
                x3[px] = x1[px] + bl*x2[px];
                x3[py] = x1[py] + bl*x2[py];
                
                blockmaxres = MAX( blockmaxres, MAX( fabs(x3[px]), fabs(x3[py]) ) );
            }
        }
        res[nthblock] = blockmaxres;
    }
    
    for( int k=0; k<NumofBlocks; ++k ) maxresidue = MAX( maxresidue , res[k] );
    
    int small_enough = 0;
    if( maxresidue < t ) small_enough = 1;
    
    delete[] res; res = 0;
    return small_enough;
}

double region_solver::vector_dotx( bool *evolved_regicator, double *v1, double *v2 )
{
	double *sum = new double[NumofBlocks];
    for( int k=0; k<NumofBlocks; ++k ) sum[k] = 0;

    int p=0;
#ifdef PARALLEL_THE_FOR_LOOPS
    #pragma omp parallel for private(p)
#endif
    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
    {
        double blocksum = 0;
        for( p = blockhead[nthblock]; p <= blocktail[nthblock] ; ++p )
        {
            if( evolved_regicator[p] )
            {
                int pp = p+p;
                blocksum += ( v1[pp]*v2[pp] + v1[pp+1]*v2[pp+1] );
            }
        }
        sum[nthblock] = blocksum;
    }
    
    double total = 0;
    for( int k=0; k<NumofBlocks; ++k ) total += sum[k];
    
	if( fabs(total) < 1e-20 )
	{ 
		std::cout<<"the result of dotx is too small"<<std::endl;
		total += 2e-20;
	}
    delete[] sum; sum = 0;
	return total;
}

void region_solver::normalize_velocity( )
{
	double maxv = 0;
	for( int p=0; p<GridSize; ++p )
	{
        int px = 2*p;
        int py = px + 1;
		double v = velocity[px]*velocity[px]+velocity[py]*velocity[py];
		v = sqrt(v);
		maxv = MAX(maxv,v);
	}
	maxv = my_max_velocity/(maxv+1e-10);
	for( int p=0; p<GridSize; ++p )
	{
        int px = 2*p;
		velocity[px+0] *= maxv;
		velocity[px+1] *= maxv;
	}
}

void region_solver::update_backward_map( )
{	
	#define GX(p,k) grad_backward_map[(p)*4+2*k]
	#define GY(p,k) grad_backward_map[(p)*4+2*k+1]
	#define WP(p,k) backward_map[(p)*2+k]
	#define X 1
	#define Y XSize

    int p=0;
    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
    {
        for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
        {
            for( int k=0; k<2; ++k )
            {
                GX(p,k) = velocity[2*p]  <0? ( edgebits[p]&XNEG ? WP(p+X,k)-WP(p,k) : WP(p,k)-WP(p-X,k) ) : ( edgebits[p]&XPOS ? WP(p,k)-WP(p-X,k) : WP(p+X,k)-WP(p,k) );
                GY(p,k) = velocity[2*p+1]<0? ( edgebits[p]&YNEG ? WP(p+Y,k)-WP(p,k) : WP(p,k)-WP(p-Y,k) ) : ( edgebits[p]&YPOS ? WP(p,k)-WP(p-Y,k) : WP(p+Y,k)-WP(p,k) );
            }
        }
    }

    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
    {
        for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
        {
            WP(p,0) += (GX(p,0)*velocity[2*p]+GY(p,0)*velocity[2*p+1]);
            WP(p,1) += (GX(p,1)*velocity[2*p]+GY(p,1)*velocity[2*p+1]);
        }
    }

	#undef GX
	#undef GY
	#undef WP	
	#undef X
	#undef Y

	compute_det_of_gradient_of_warp( grad_backward_map, det_backward_map );
}

void region_solver::update_forward_map( )
{
	#define WP(p,k) forward_map[(p)*2+k]
	
    int p=0;
    for( int nthblock = 0; nthblock < NumofBlocks; ++nthblock )
    {
        for( p = blockhead[nthblock]; p <= blocktail[nthblock]; ++p )
        {
            float j = WP(p,0); j = j<0? 0: ( j>XSize-1? XSize-1 : ROUND(j) );
            float i = WP(p,1); i = i<0? 0: ( i>YSize-1? YSize-1 : ROUND(i) );
            int pp = i*XSize+j;
            WP(p,0) -= velocity[2*pp];
            WP(p,1) -= velocity[2*pp+1];
        }
    }
	
	#undef WP
}

int region_solver::update_occlusion_map( double &energy )
{
	float *residue_map		= new float[GridSize];
	float *smoothed_residue = new float[GridSize];
	for( int p=0; p<GridSize; ++p )
	{
		if( region_indicator[p]<0.5 ){ residue_map[p]=0; continue; }
        int ppp = p*CHANNELS;
        float d0 = evolved_image1[ppp+0]-image0[ppp+0];
        float d1 = evolved_image1[ppp+1]-image0[ppp+1];
        float d2 = evolved_image1[ppp+2]-image0[ppp+2];
        residue_map[p] = d0*d0 + d1*d1 + d2*d2;
	}
	residueblur( residue_map, smoothed_residue, SMOOTHNESS_OF_OCC, region_indicator, true );
	
	int label_changed = 0;
	for( int p=0; p<GridSize; ++p )
	{
		if( region_indicator[p]<0.5 ){ occlusion_map[p]=0; continue; }
		uchar old_label = occlusion_map[p];
		occlusion_map[p] = smoothed_residue[p]*GAMA_OMEGA>BETA0? 1:0;
		if( occlusion_map[p] != old_label ) ++label_changed;
	}
	
	energy = 0;
	int unoccluded_pix = 0;
	for( int p=0; p<GridSize; ++p )
	{
		if( region_indicator[p]>0.5 ){ energy += occlusion_map[p]==1? BETA0: residue_map[p]; ++unoccluded_pix; }
	}
	energy /= (double)unoccluded_pix;

	delete[] residue_map; residue_map=0;
	delete[] smoothed_residue; smoothed_residue=0;

	return label_changed;
}
 
int region_solver::estimate_warp_and_occlusion( bool getoutoflocal, int winsize, int minimumsupport, int a, int b, int c )
{
	MOTION_ZERO = false;
	int k = 0, combo_shoot = 0;

#ifdef GENERATE_INTERMEDIATE_RESULT
	showContour( k, "contour_on_img1_", moved_img );
#endif
	double global_old_energy = FLT_MAX;
	if( getoutoflocal )get_out_of_local();
	if( a>0 )my_max_velocity = MAX_VELOCITY;
    
	do
	{
		++k;
		double old_energy = FLT_MAX, new_energy = 0;
		int changes = GridSize;
		int max_subtranslation = 0;
		do 
		{
			++max_subtranslation;
			if( k > a )break;
            solve_for_translation();
            update_backward_map();
			update_forward_map();
			get_evolved_image( image0, backward_map, evolved_image0 );
			get_evolved_image( image1, forward_map, evolved_image1 );
			changes = update_occlusion_map( new_energy );
			if( new_energy>old_energy )break;
			if( max_subtranslation > 30 )break;
			old_energy = new_energy;
		}while(1);
     
        old_energy = FLT_MAX; new_energy = 0;
		int wocao = 0;
		do 
		{
			++wocao;
            
//            std::cout<<"estimate flow..."<<std::endl;
			cv::Mat deformation_field = solve_for_velocity_using_region_based_metric(k);
//            std::cout<<"flow estimated..."<<std::endl;
            
			update_backward_map();

#ifdef GENERATE_INTERMEDIATE_RESULT
            cv::Mat cumulative_warp = showwarp( backward_map, "backward_map", k );
#endif
			
            update_forward_map();
			get_evolved_image( image0, backward_map, evolved_image0 );
            
#ifdef GENERATE_INTERMEDIATE_RESULT
            cv::Mat ed_i0_mat = showimage( evolved_image0, "evolved_img0_", k );
#endif

            get_evolved_image( image1, forward_map, evolved_image1 );
			changes = update_occlusion_map( new_energy );
            
#ifdef GENERATE_INTERMEDIATE_RESULT
            show_occlusion_map();
#endif
            
            std::cout<<"pixels that changed labels: "<<changes<<"  energy is: "<<new_energy<<std::endl;
            
#ifdef GENERATE_INTERMEDIATE_RESULT
			showContour( k, "contour_on_img1_", moved_img );
			showContour( k, "contour_on_evolved_img0_", ed_i0_mat ); ed_i0_mat.release();
			showContour( k, "contour_on_deformation_field", deformation_field );
			showContour( k, "contour_on_backward_map", cumulative_warp ); cumulative_warp.release();
#endif
            deformation_field.release();
            
			if( new_energy>old_energy ){ break; };
			old_energy = new_energy;
		}while( wocao < 1);
        
		
		if( new_energy > global_old_energy ) my_max_velocity *=0.968;
		global_old_energy = new_energy;
		
		if( changes==0 )combo_shoot += 1;
		else { combo_shoot=0; }
    } while( k<b && combo_shoot<c );
	calculate_F_for_yanchaoscheme( winsize, minimumsupport );
	if( MOTION_ZERO ) return 1;
	else return 0;
}