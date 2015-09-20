#include "multi_object_tracking.h"

#define OLD_COLOR_SCHEME

#ifdef OLD_COLOR_SCHEME
int Region_Color[13][3] = { {255,255,255},  \
                            {255,0,0},      \
                            {0,255,0},      \
                            {0,0,255},      \
                            {0,255,255},    \
                            {255,0,255},    \
                            {255,255,0},    \
                            {128,0,0},      \
                            {0,128,0},      \
                            {0,0,128},      \
                            {0,128,128},    \
                            {128,0,128},    \
                            {128,128,0} };
#else
int Region_Color[13][3] = { {255,255,255}, \
                            {128,0,0}, \
                            {0,128,0}, \
                            {0,0,128}, \
                            {255,0,0}, \
                            {0,255,0}, \
                            {0,0,255}, \
                            {0,128,128}, \
                            {128,0,128}, \
                            {128,128,0}, \
                            {0,255,255}, \
                            {255,0,255}, \
                            {255,255,0} };
#endif

int Shaded_Color[22][3] = { {126,158,130},  \
							{188,58,58},    \
							{58,35,174},    \
							{58,75,240},    \
							{35,97,174},    \
							{46,151,140},   \
							{171,169,24},   \
							{112,11,120},   \
							{83,165,241},   \
							{83,165,241},   \
                            {128,0,0},      \
                            {0,128,0},      \
                            {0,0,128},      \
                            {255,0,0},      \
                            {0,255,0},      \
                            {0,0,255},      \
                            {0,128,128},    \
                            {128,0,128},    \
                            {128,128,0},    \
                            {0,255,255},    \
                            {255,0,255},    \
                            {255,255,0} };
cv::Mat hole_indicator;
cv::Mat disocc_indicator;
std::vector<int> region_index;

int ColorWantToShow = -1;   //

int REGIONS;                //
int NEXT_FRAME;             //

float HISTOGRAM_SUPPORT0 = -1;   //
const float HISTOGRAM_SUPPORT1 = 0.28;   //

const float STABLE_factor = 0.5;
const float STABLE_LABELMAP = 12;
const int BLOB_SIZE = 9;
const float ERROR_MARGIN = 0.04;
const float ERROR_MARGIN_DIS = 0.01;
const float DIS_OCC_RATIO = 0.1;
const float DISCRIMINATE_RATIO = 0.35;
const float DISCRIMINATE_RATIO2 = 0.25;

const float MAX_SPDIFF = 0.1;

const float Threshold_for_classifition = 0.5;
const int Smoothing_contour = 5;

const float OBJECT_RATIO = 0.49;         //
const int OBJECT_MIN_SIZE = 121;        //

void read_parameters( )
{
	std::ifstream myFile( "a_list_of_params.txt", std::ifstream::in );
	char inputline[200];
	if( myFile.good() )
	{
		while( !myFile.eof() )
		{
			myFile.getline(inputline,100);
			sscanf (inputline,"REGIONS=%d", &REGIONS);
			sscanf (inputline,"NEXT_FRAME=%d", &NEXT_FRAME);
			
			sscanf (inputline,"HISTOGRAM_SUPPORT0=%f", &HISTOGRAM_SUPPORT0);
		}
		std::cout<<"REGIONS= "<<REGIONS<<std::endl;
		std::cout<<"NEXT_FRAME= "<<NEXT_FRAME<<std::endl;
		
		std::cout<<"HISTOGRAM_SUPPORT0= "<<HISTOGRAM_SUPPORT0<<std::endl;
		std::cout<<"HISTOGRAM_SUPPORT1= "<<HISTOGRAM_SUPPORT1<<std::endl;
		
		std::cout<<"STABLE_factor= "<<STABLE_factor<<std::endl;
		std::cout<<"STABLE_LABELMAP= "<<STABLE_LABELMAP<<std::endl;
		std::cout<<"BLOB_SIZE= "<<BLOB_SIZE<<std::endl;
		std::cout<<"ERROR_MARGIN= "<<ERROR_MARGIN<<std::endl;
		std::cout<<"ERROR_MARGIN_DIS= "<<ERROR_MARGIN_DIS<<std::endl;
		std::cout<<"DIS_OCC_RATIO= "<<DIS_OCC_RATIO<<std::endl;
		std::cout<<"DISCRIMINATE_RATIO= "<<DISCRIMINATE_RATIO<<std::endl;
		std::cout<<"DISCRIMINATE_RATIO2= "<<DISCRIMINATE_RATIO2<<std::endl;
		std::cout<<"MAX_SPDIFF= "<<MAX_SPDIFF<<std::endl;
		std::cout<<"Threshold_for_classifition= "<<Threshold_for_classifition<<std::endl;
		std::cout<<"Smoothing_contour= "<<Smoothing_contour<<std::endl;
		std::cout<<"OBJECT_RATIO= "<<OBJECT_RATIO<<std::endl;
		std::cout<<"OBJECT_MIN_SIZE= "<<OBJECT_MIN_SIZE<<std::endl;
		
		myFile.close();
	}
	else
	{
		std::cout <<"ERROR: can't open file."<<std::endl;
		while(1);
	}
	return;
}

int get_initial_labels_from_mask( int frameno, std::string maskname )
{
    cv::Mat maskimg = cv::imread( maskname );
    int XSize = maskimg.cols;
    int YSize = maskimg.rows;
    int *labelmap = new int[XSize*YSize];
    cv::MatIterator_<cv::Vec3b> it = maskimg.begin<cv::Vec3b>();
    
    for( int p=0; p<XSize*YSize; ++p, ++it )
    {
        int label = 0;
        float min_dist = 0;
        for( int k = 0; k < CHANNELS; ++k )
            min_dist += ( (int)((*it)[k]) - Region_Color[0][k] )*( (int)((*it)[k]) - Region_Color[0][k] );
        for( int k = 1; k < REGIONS; ++k )
        {
            float dist = 0;
            for( int c = 0; c < CHANNELS; ++c )
                dist += ( (int)((*it)[c]) - Region_Color[k][c] )*( (int)((*it)[c]) - Region_Color[k][c] );
            if( dist < min_dist ){ min_dist = dist; label = k; }
        }
        labelmap[p] = label;
    }
    write_array_to_text(labelmap, XSize, YSize, frameno, LABELMAP_NAME);
    
    maskimg.release();
    delete[] labelmap; labelmap = 0;
    return 1;
}

int get_next_frame( )
{
	return NEXT_FRAME;
}

int render_images_using_labels( cv::Mat img1, int *labels, int frameno )
{
    if( SHADED_REGION )
    {
        cv::Mat img;
        img1.copyTo(img);
        cv::MatIterator_<cv::Vec3b> itimg = img.begin<cv::Vec3b>();
        for( int p=0; p<img.cols*img.rows; ++p )
        {
            for( int k=0; k<CHANNELS; ++k )
                (*(itimg+p))[k] = (uchar)( 0.5*((float)((*(itimg+p))[k])) + 0.5*((float)(Shaded_Color[labels[p]][k])) );
        }
        int bbbb = frameno/1000, bbb = ( frameno - 1000*bbbb )/100, bb = ( frameno - 1000*bbbb - 100*bbb )/10, b = frameno%10;
        char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
        std::string imgname = "segmentation";
        imgname = imgname+aaaa+aaa+aa+a+".png";
        cv::imwrite( imgname, img );
        img.release();
        return 1;
    }
    else
    {
        if( ColorWantToShow < 0 )
        {
            std::cout<<"Please input the color of the contour: "; std::cin>>ColorWantToShow;
        }
        for( int k=1; k<REGIONS; ++k )
        {
            region_solver fordisplay;
            float regionLen, regionSize;
            if( k==1 )
            {
                fordisplay.allocate( img1, img1, labels, region_index[k], regionLen, regionSize );
                fordisplay.showContour( frameno, "regionincontour", img1, Shaded_Color[region_index[k]+ColorWantToShow][0], Shaded_Color[region_index[k]+ColorWantToShow][1], Shaded_Color[region_index[k]+ColorWantToShow][2] );
            }
            else
            {
                cv::Mat img = cv::imread("tempcontour.png");
                fordisplay.allocate( img, img, labels, region_index[k], regionLen, regionSize );
                fordisplay.showContour( frameno, "regionincontour", img, Shaded_Color[region_index[k]+ColorWantToShow][0], Shaded_Color[region_index[k]+ColorWantToShow][1], Shaded_Color[region_index[k]+ColorWantToShow][2] );
                img.release();
            }
            fordisplay.deallocate();
        }
        int bbbb = frameno/1000, bbb = ( frameno - 1000*bbbb )/100, bb = ( frameno - 1000*bbbb - 100*bbb )/10, b = frameno%10;
        char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
        std::string imgname = "regions";
        imgname = imgname+aaaa+aaa+aa+a+".png";
        cv::Mat img = cv::imread("tempcontour.png");
        cv::imwrite( imgname, img );
        img.release();
        return 1;
    }
}

int initialize_labelmap( int *labelmap, int GSize, int XSize, int YSize )
{
    region_index.clear();
    region_index.push_back(0);
    for( int p = 0; p < GSize; ++p )
    {
        if( labelmap[p] == -1 ) continue;
        if( std::find(region_index.begin(), region_index.end(), labelmap[p]) == region_index.end() )
            region_index.push_back(labelmap[p]);
    }
    REGIONS = (int)region_index.size();
    int Max_Ind = *std::max_element( region_index.begin(), region_index.end() );
    
	int *templabel = new int[GSize];
	for( int p=0; p<GSize; ++p ) templabel[p] = 0;
	for( int k = 1; k < REGIONS; ++k )
	{
        cv::Mat img = cv::Mat::zeros( YSize, XSize, CV_8UC1 ); // force grayscale
        cv::MatIterator_<uchar> it = img.begin<uchar>();
        std::vector < std::vector<cv::Point2i > > blobs;
		
		for( int p = 0; p < GSize; ++p, ++it ) if( labelmap[p] == region_index[k] ) *it = 1;
		FindBlobs( img, blobs );
		
		int max_blobsize = 0;
		for(  size_t i = 0; i < blobs.size(); ++i ) if( blobs[i].size() > max_blobsize ) max_blobsize = (int)blobs[i].size();
		
        bool mainblobfound = false;
		for(  size_t i = 0; i < blobs.size(); ++i )
		{
			if( blobs[i].size() > OBJECT_MIN_SIZE && (float)blobs[i].size()/(float)max_blobsize > OBJECT_RATIO )
			{
                if( blobs[i].size() == max_blobsize && !mainblobfound )
                {
                    mainblobfound = true;
                    for( size_t j = 0; j < blobs[i].size(); ++j )
                    {
                        int x = blobs[i][j].x;
                        int y = blobs[i][j].y;
                        int p = y * XSize + x;
                        templabel[p] = region_index[k];
                    }
                }
                else
                {
                    Max_Ind += 1;
                    for( size_t j = 0; j < blobs[i].size(); ++j )
                    {
                        int x = blobs[i][j].x;
                        int y = blobs[i][j].y;
                        int p = y * XSize + x;
                        templabel[p] = Max_Ind;
                    }
                }
			}
			else
			{
				for( size_t j = 0; j < blobs[i].size(); ++j )
				{
					int x = blobs[i][j].x;
					int y = blobs[i][j].y;
					int p = y * XSize + x;
					templabel[p] = -1;
				}
			}
		}
        
        img.release();
        blobs.clear();
	}
    
    int NUMofDIS = 0;
    for( int p = 0; p < GSize; ++p ){
        if( labelmap[p] > 0 ) labelmap[p] = templabel[p];
        if( labelmap[p] == -1 ) NUMofDIS += 1;
    }
    
    region_index.clear();
    region_index.push_back(0);
    for( int p = 0; p < GSize; ++p )
    {
        if( labelmap[p] == -1 ) continue;
        if( std::find(region_index.begin(), region_index.end(), labelmap[p]) == region_index.end() )
            region_index.push_back(labelmap[p]);
    }
    REGIONS = (int)region_index.size();

    float MIN_BLOB_SIZE_FOR_RATIO = FLT_MAX;
    float BACKGROUND_SIZE = 0;
    for( int p = 0; p < GSize; ++p ) if( labelmap[p] == region_index[0] ) ++BACKGROUND_SIZE;
    for( int k = 1; k < REGIONS; ++k )
    {
        float sum = 0;
        for( int p = 0; p < GSize; ++p ) if( labelmap[p] == region_index[k] ) ++sum;
        MIN_BLOB_SIZE_FOR_RATIO = MIN( MIN_BLOB_SIZE_FOR_RATIO, sum );
    }
    if( BACKGROUND_SIZE < 10 ){ std::cout<<"background size is too small...(initialize_labelmap)"<<std::endl; while(1); }
    
    HISTOGRAM_SUPPORT0 = MIN_BLOB_SIZE_FOR_RATIO / BACKGROUND_SIZE;
    if( HISTOGRAM_SUPPORT0 < 0.02 )
        HISTOGRAM_SUPPORT0 = 0.02;
    else if ( HISTOGRAM_SUPPORT0 > 0.12 )
        HISTOGRAM_SUPPORT0 = 0.12;
    
    if( MIN_BLOB_SIZE_FOR_RATIO > 0.5*BACKGROUND_SIZE )
        HISTOGRAM_SUPPORT0 = HISTOGRAM_SUPPORT1*MIN_BLOB_SIZE_FOR_RATIO/BACKGROUND_SIZE;
    
    std::cout<<" HISTOGRAM_SUPPORT0 : "<<HISTOGRAM_SUPPORT0<<std::endl;
    if( HISTOGRAM_SUPPORT0 < 0 ){ std::cout<<"Negative Support! "<<std::endl; while(1); }
	
    delete[] templabel; templabel = 0;

	return NUMofDIS;
}

void regularize_labelmap( int *labelmap, int GSize, int XSize, int YSize )
{
	narrowband redband;
	redband.allocate( XSize, YSize );
	enum {XPOS=1, XNEG=2, YPOS=4, YNEG=8,
		XEDGES=3,       YEDGES=12};

	int *templabel = new int[GSize];
	int isolated_pix_found = 0;
	int iter=0;
	int *labelhist = new int[REGIONS];
	do{
		++iter;
		isolated_pix_found = 0;
		for( int p=0; p<GSize; ++p )templabel[p]=labelmap[p];
		for( int k=0; k<REGIONS; ++k )
		{
			for( int p=0; p<GSize; ++p )
			{
				if( labelmap[p] == region_index[k] )
				{
					int xa = redband.edgebits[p] & XNEG ? 0 : ( labelmap[p-1]		== region_index[k]? 1 : 0 );
					int xb = redband.edgebits[p] & YNEG ? 0 : ( labelmap[p-XSize]	== region_index[k]? 1 : 0 );
					int xc = redband.edgebits[p] & XPOS ? 0 : ( labelmap[p+1]		== region_index[k]? 1 : 0 );
					int xd = redband.edgebits[p] & YPOS ? 0 : ( labelmap[p+XSize]	== region_index[k]? 1 : 0 );
					if( xa+xb+xc+xd < 1 )
					{
						isolated_pix_found += 1;
						for( int lk=0; lk<REGIONS; ++lk ) labelhist[lk]=0;
						for( int i=0; i<REGIONS; ++i )
						{
							labelhist[i] += redband.edgebits[p] & XNEG ? 0 : ( labelmap[p-1]		== region_index[i]? 1 : 0 );
							labelhist[i] += redband.edgebits[p] & YNEG ? 0 : ( labelmap[p-XSize]	== region_index[i]? 1 : 0 );
							labelhist[i] += redband.edgebits[p] & XPOS ? 0 : ( labelmap[p+1]		== region_index[i]? 1 : 0 );
							labelhist[i] += redband.edgebits[p] & YPOS ? 0 : ( labelmap[p+XSize]	== region_index[i]? 1 : 0 );
						}
						int majoritylabel = -1;
						int numofmajoritylabel = 0;
						for( int i=0; i<REGIONS; ++i )if( labelhist[i]>numofmajoritylabel ){ majoritylabel=i; numofmajoritylabel=labelhist[i]; }
						labelmap[p] = region_index[ majoritylabel ];
					}
				}
			}
		}
		int labelchanged = 0;
		for( int p=0; p<GSize; ++p )if( labelmap[p] != templabel[p] )++labelchanged;
		std::cout<<"isolated_pix_found is "<<isolated_pix_found<<std::endl;
		
		if(isolated_pix_found==0||labelchanged==0)break;
		
	}while(iter<500);
	delete[] labelhist; labelhist=0;
	delete[] templabel; templabel=0;
	redband.deallocate();
}

void dissocclusion_classification( cv::Mat &img1, cv::Mat &img2, int *labelmap, int frameno )
{
	int GSize = img1.cols * img1.rows;
	int XSize = img1.cols, YSize = img1.rows;
	int NUMofDIS = 0;
    for( int p=0; p<GSize; ++p )if( labelmap[p] == -1 ) NUMofDIS += 1;
	if( NUMofDIS > 0 )
	{
		region_solver *myregionsinit;
		myregionsinit = new region_solver[REGIONS];
        
        
		for( int k=0; k<REGIONS; ++k )
		{
			float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
			float regionLen, regionSize;
			myregionsinit[k].allocate( img1, img2, labelmap, region_index[k], regionLen, regionSize );
			std::cout<<"**** Disocclusion Classification **** "<<k<<"th Region ***"<<std::endl;
			std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
			myregionsinit[k].estimate_warp_and_occlusion( true, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, TRANSLATION, INITIAL_ITERATIONS, MAX_COMBO );
			myregionsinit[k].calculate_F_for_disocclusion( labelmap, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT );
		}
        
		classify_disocclusion_new_no_blob( myregionsinit, labelmap, frameno );
		for( int p = 0; p < GSize; ++p ) if( labelmap[p] < 0 ){ std::cout<<"wrong in classify_disocclusion.."; while(1); }
        
		extract_the_largest_component( labelmap, GSize, XSize, YSize );// no -1
		for( int k = 0; k < REGIONS; ++k ) myregionsinit[k].deallocate( );
		delete[] myregionsinit; myregionsinit = 0;
        std::cout<<" Disocclusion Classified! "<<std::endl;
	}
}

void extract_the_largest_component( int *labelmap, int GSize, int XSize, int YSize )
{
	int *templabel = new int[GSize]; for( int p = 0; p < GSize; ++p )templabel[p] = 0;
	for( int k = 1; k < REGIONS; ++k )
	{
		cv::Mat img = cv::Mat::zeros( YSize, XSize, CV_8UC1 ); // force greyscale
		cv::MatIterator_<uchar> it = img.begin<uchar>();
		
		std::vector < std::vector<cv::Point2i > > blobs;
		
		for( int p = 0; p < GSize; ++p, ++it )if( labelmap[p] == region_index[k] ) *it = 1;
		FindBlobs( img, blobs );
		int max_blobsize = 0;
		int max_blobindex = -1;
		for(  size_t i = 0; i < blobs.size(); ++i ){
			if( (int)(blobs[i].size()) > max_blobsize ){
				max_blobsize = (int)( blobs[i].size() );
				max_blobindex = (int)i;
			}
		}
		for(  size_t i = 0; i < blobs.size(); ++i ){
			if( (float)blobs[i].size() / (float)max_blobsize < OBJECT_RATIO )continue;
			for( size_t j = 0; j < blobs[i].size(); ++j )
			{
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;
				int p = y * XSize + x;
				templabel[p] = region_index[k];
			}
		}
		img.release();
		blobs.clear();
	}
	for( int p = 0; p < GSize; ++p ) labelmap[p] = templabel[p];
	delete[] templabel; templabel = 0;
}

void hole_detection( int *labelmap, int frameno )
{
	std::string dddname = "transformed_occlusion";
    int bbbb = frameno/1000, bbb = ( frameno - 1000*bbbb )/100, bb = ( frameno - 1000*bbbb - 100*bbb )/10, b = frameno%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
	dddname = dddname+aaaa+aaa+aa+a+".png";
	cv::Mat transformed_occ = cv::imread( dddname, 0 );
	cv::MatIterator_<uchar> it = transformed_occ.begin<uchar>();
	cv::MatIterator_<uchar> ith = hole_indicator.begin<uchar>();
    
    int Max_Ind = *std::max_element( region_index.begin(), region_index.end() );
    Max_Ind += 1;

	int GSize = transformed_occ.cols * transformed_occ.rows;
	int XSize = transformed_occ.cols, YSize = transformed_occ.rows;

	for( int p = 0; p < GSize; ++p )
		if( labelmap[p] == -1 && *(it+p) > 100 ) *(ith+p) = 1;
		else *(ith+p) = 0;
	std::vector < std::vector<cv::Point2i > > blobs; blobs.clear();
    
    FindBlobs( hole_indicator, blobs );
    
	// check the blobs to find holes
	for( int p = 0; p < GSize; ++p ) *( ith + p ) = 0;
	float *labelhist = new float[Max_Ind];
 	for( size_t i = 0; i < blobs.size(); ++i )
	{
		for( int rk = 0; rk < Max_Ind; ++rk ) labelhist[rk] = 0;
		narrowband BandForHole;
		BandForHole.allocate( XSize, YSize );
		for( int p=0; p<GSize; ++p ) BandForHole.Psi[p] = 1.0;
		for( size_t j = 0; j < blobs[i].size(); ++j )
		{
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			int p = y * XSize + x;
			BandForHole.Psi[p] = -1.0;
		}
		BandForHole.createBand();
		for( int *ptr = BandForHole.band, p=*ptr; ptr!=BandForHole.tail; p=*++ptr )
		{
			if( labelmap[p] == -1 )continue;
			labelhist[labelmap[p]] += 1.0;
		}
		for( int *ptr = BandForHole.edgeband, p=*ptr; ptr!=BandForHole.edgetail; p=*--ptr )
		{
			if( labelmap[p] == -1 )continue;
			labelhist[labelmap[p]] += 1.0;
		}
		std::vector<float> Vecofdiff;
		Vecofdiff.clear();
		for( int rk = 0; rk < Max_Ind; ++rk ) Vecofdiff.push_back( labelhist[rk] );
		std::nth_element( Vecofdiff.begin(), Vecofdiff.begin()+1, Vecofdiff.end(), std::greater<float>());
        if( Vecofdiff[1]>Vecofdiff[0] ){ std::cout<<"wrong using std::nth_element..."<<std::endl; while(1); }
		if( Vecofdiff[1]/Vecofdiff[0] < ISO_THRESH )
		{
			for( size_t j = 0; j < blobs[i].size(); ++j )
			{
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;
				int p = y * XSize + x;
				*( ith + p ) = 255;
			}
		}
		Vecofdiff.clear();
		BandForHole.deallocate();
	}
	delete[] labelhist; labelhist = 0;
	transformed_occ.release();
	blobs.clear();
}

void classify_disocclusion_new_no_blob( region_solver *regionPlus, int *labelmap, int frameno )
{
	//here we need a function to detect holes
	hole_detection( labelmap, frameno );
    
	cv::imwrite( "holemap.png", hole_indicator );
    
	//
	int GridSize = regionPlus[0].GridSize;
	int XSize = regionPlus[0].XSize;
	int YSize = regionPlus[0].YSize;
	int *templabel = new int[GridSize];
	for( int p = 0; p < GridSize; ++p ) templabel[p] = labelmap[p];
	double *Qee = new double[REGIONS];
	double *Rar = new double[REGIONS];
	double *Err = new double[REGIONS];
	int *Kar = new int[REGIONS];

	for( int p = 0; p < GridSize; ++p )
	{
		if( labelmap[p] != -1 ) continue;
		for( int k = 0; k < REGIONS; ++k ){ Qee[k] = 0; Rar[k] = 0; Err[k] = 0; Kar[k] = 0; }
		
		int m = p/XSize;
		int n = p%XSize;
		float pixelcounted = 0;
		for( int i = m - BLOB_SIZE; i <= m + BLOB_SIZE; ++i )
			for( int j = n - BLOB_SIZE; j <= n + BLOB_SIZE; ++j )
			{
				if( (i-m)*(i-m)+(j-n)*(j-n) > BLOB_SIZE*BLOB_SIZE )continue;
				int pp = i*XSize + j;
				if( i>=0 && i<=YSize-1 && j>=0 && j<=XSize-1 && labelmap[pp] == -1 )
				{
					bool outofdomain = false;
					for( int k = 0; k < REGIONS; ++k ) if( regionPlus[k].F_error[pp] > 5 ) outofdomain = true;
					if( outofdomain ) continue;
					for( int k = 0; k < REGIONS; ++k ) Err[k] += regionPlus[k].F_error[pp];
					pixelcounted += 1.0;
				}
			}
		if( pixelcounted < 1 ) for( int k = 0; k < REGIONS; ++k ) Err[k] = 0;
		else for( int k = 0; k < REGIONS; ++k ) { Err[k] = Err[k]/pixelcounted; Err[k] = sqrt( Err[k] ); }
		for( int k = 0; k < REGIONS; ++k ) Rar[k] = regionPlus[k].F_ratio[p] + 1.0e-10;
		
		double minERROR = FLT_MAX;
		for( int k = 0; k < REGIONS; ++k ) minERROR = MIN( minERROR, Err[k] );
		for( int k = 0; k < REGIONS; ++k )
		{
			double step_p;
			double t = Err[k] - minERROR;
			if( t > 1.25*ERROR_MARGIN_DIS ) step_p = 0;
			else if( t >= 0.75*ERROR_MARGIN_DIS && t <= 1.25*ERROR_MARGIN_DIS )
				step_p = 1 - ( t - 0.75*ERROR_MARGIN_DIS ) / (0.5*ERROR_MARGIN_DIS);
			else step_p = 1.0;
			Qee[k] = Rar[k]*step_p;
			if( minERROR > BOTH_OCCLUSION ) Qee[k] = Rar[k];
		}
		
		 double maxQ = -FLT_MAX;
		 int ind_maxQ = -1;
		 for( int k = 0; k < REGIONS; ++k ) if( Qee[k] > maxQ ){ maxQ = Qee[k]; ind_maxQ = k; }
		 if( ind_maxQ == -1 ){ std::cout<<"somthing wrong in update_label"; while(1); }
		
		 double secondQ = -FLT_MAX;
		 for( int k = 0; k < REGIONS; ++k )
		 {
			 if( k == ind_maxQ ) continue;
			 if( Qee[k] > secondQ ){ secondQ = Qee[k]; }
		 }
		 if( secondQ / maxQ < DIS_OCC_RATIO ) templabel[p] = region_index[ind_maxQ];
		 else templabel[p] = 0;
	}
	
	cv::MatIterator_<uchar> ith = hole_indicator.begin<uchar>();
	//***** label assignment for each p *****//
	for( int p = 0; p < GridSize; ++p )
	{
		if( labelmap[p] != -1 ) continue;
		bool constreg = false;
		for( int k = 0; k < REGIONS; ++k ) if( regionPlus[k].ConstRegion[p] > 0 ) constreg = true;
		if( *(ith+p) > 100 || constreg  ){ }
		else continue;
		for( int k = 0; k < REGIONS; ++k ){ Rar[k]=-1; Kar[k]=-1; }
		int	  candidateNo = 0;
		for( int k = 0; k < REGIONS; ++k )
		{
			if( regionPlus[k].F_ratio[p] > -10 )
			{
				Rar[candidateNo] = regionPlus[k].F_ratio[p] + 1.0e-10;
				Kar[candidateNo] = k;
				candidateNo += 1;
			}
		}
		if( candidateNo != REGIONS ){ std::cout<<"something wrong in classify_disocclusion..."; while(1); } // p is not on the boundary, label is oldlabel
		
		double maxQ = -FLT_MAX;
		int ind_maxQ = -1;
		for( int j = 0; j < candidateNo; ++j ) if( Rar[j] > maxQ ){ maxQ = Rar[j]; ind_maxQ = j; }
		if( ind_maxQ == -1 ){ std::cout<<"somthing wrong in classify_disocclusion"; while(1); }
		
		double secondQ = -FLT_MAX;
		for( int j = 0; j < candidateNo; ++j )
		{
			if( j == ind_maxQ ) continue;
			if( Rar[j] > secondQ ){ secondQ = Rar[j]; }
		}
		if( secondQ / maxQ < DIS_OCC_RATIO ) templabel[p] = region_index[ Kar[ind_maxQ] ];
		else templabel[p] = 0;
	}
	delete[] Qee; Qee = 0;
	delete[] Rar; Rar=0;
	delete[] Kar; Kar=0;
	delete[] Err; Err=0;
	for( int p = 0; p < GridSize; ++p ) labelmap[p] = templabel[p];
	delete[] templabel; templabel = 0;
}

void markimages_using_label( cv::Mat img0, int *labelmap, std::string imname, int num )
{
    int Max_Ind = *std::max_element( region_index.begin(), region_index.end() );
	if( Max_Ind>21 ){ std::cout<<"only 22 regions are allowed currently..."; while(1); }
	cv::Mat temp;
	img0.copyTo(temp);
	cv::MatIterator_<cv::Vec3b> itt = temp.begin<cv::Vec3b>();
	for( int p=0; p<img0.cols*img0.rows; ++p )
	{
		if( labelmap[p]==0 || labelmap[p]==-1 ) continue;
        for( int k=0; k < CHANNELS; ++k )
            (*(itt+p))[k] = (uchar)(0.5*(float)(*(itt+p))[k] + 0.5*(float)Shaded_Color[labelmap[p]][k]);
	}
    int bbbb = num/1000, bbb = ( num - 1000*bbbb )/100, bb = ( num - 1000*bbbb - 100*bbb )/10, b = num%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
	imname = imname+aaaa+aaa+aa+a+".png";
	cv::imwrite( imname, temp ); cv::waitKey(1);
	temp.release();
}

void combined_warp( region_solver *myregions, int *labelmap, std::string imgname, int num )
{
	float *combined_warp = new float[2*myregions[0].GridSize];
	
	for( int p=0; p<myregions[0].GridSize; ++p )
	{
        int regionId=0;
        for( int k=0; k<REGIONS; ++k ) if( region_index[k]==labelmap[p] ){ regionId = k; break; }
		combined_warp[2*p]	= myregions[regionId].forward_map[2*p];
		combined_warp[2*p+1]= myregions[regionId].forward_map[2*p+1];
	}
	
	myregions[0].showwarp( combined_warp, imgname, num );
	
	delete[] combined_warp; combined_warp=0;
}

void save_labelmap( int *labelmap, cv::Mat maskimg, std::string imname, int num )
{
	cv::Mat temp = cv::Mat::zeros( maskimg.size(), maskimg.type() );
	cv::MatIterator_<cv::Vec3b> itt = temp.begin<cv::Vec3b>();
	for( int p=0; p<temp.cols*temp.rows; ++p, ++itt )
	{
		for( int k=0; k<REGIONS; ++k ){
			if( labelmap[p]==k ) for(int c=0; c<CHANNELS; ++c)(*itt)[c]=Region_Color[k][c];
		}
	}
    int bbbb = num/1000, bbb = ( num - 1000*bbbb )/100, bb = ( num - 1000*bbbb - 100*bbb )/10, b = num%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
	imname = imname+aaaa+aaa+aa+a+".png";
	
	cv::imwrite( imname, temp ); cv::waitKey(1);
	temp.release();
}

void multi_object_tracking_and_segmentation( std::string imgname0, std::string imgname1, std::string imgname2, std::string maskname, int frameno )
{
    std::cout<<"*** multi object tracking *** frame: "<<frameno<<std::endl;
	cv::Mat img0 = cv::imread( imgname0 );
	cv::Mat img1 = cv::imread( imgname1 );
	cv::Mat img2 = cv::imread( imgname2 );
	int GSize = img1.cols*img1.rows, XSize = img1.cols, YSize = img1.rows;
    
    hole_indicator.release();
    hole_indicator = cv::Mat::zeros( YSize, XSize, CV_8UC1 );
    disocc_indicator.release();
    hole_indicator.copyTo(disocc_indicator);
    
	int *labelmap = new int[GSize];
    int ncols, nrows;
    get_array_from_text( labelmap, ncols, nrows, frameno, maskname);
    if( ncols != XSize || nrows != YSize )
    {
        std::cout<<" Image dimensions do NOT match! (multi_object_tracking_and_segmentation) "<<std::endl;
        std::cout<<" ncols: "<<ncols<<" nrows: "<<nrows<<" XSize: "<<XSize<<" YSize: "<<YSize<<std::endl;
        while(1);
    }
    // disocc_indicator is the indicator of black region: 000, nonblack: 255
    cv::MatIterator_<uchar> itdis = disocc_indicator.begin<uchar>();
    for( int p=0; p<GSize; ++p )
        if( labelmap[p] == -1 ) *(itdis+p) = 0;
        else *(itdis+p) = 255;
    
    initialize_labelmap(labelmap, GSize, XSize, YSize);
//	write_array_to_text(labelmap, XSize, YSize, frameno, "templabels");
    
    if( OUTPUTSEQ ) { render_images_using_labels( img1, labelmap, frameno ); return; }
	
	// Dis-occlusion Classifition, Get rid of -1
	dissocclusion_classification( img1, img2, labelmap, frameno );
	markimages_using_label( img1, labelmap, "labelmap", 0 );
	///
	
	region_solver *myregionsminus;
	myregionsminus = new region_solver[REGIONS];
	for( int k = 0; k < REGIONS; ++k )
	{
		float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
		float regionLen, regionSize;
		myregionsminus[k].allocate( img1, img0, labelmap, region_index[k], regionLen, regionSize );
		std::cout<<"**** Estimate backward map and occlusion for **** "<<k<<"th Region ***"<<std::endl;
		std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
		myregionsminus[k].estimate_warp_and_occlusion( true, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, TRANSLATION, INITIAL_ITERATIONS, MAX_COMBO );
	}
	
	region_solver *myregionsplus;
	myregionsplus = new region_solver[REGIONS];
	for( int k = 0; k < REGIONS; ++k )
	{
		float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
		float regionLen, regionSize;
		myregionsplus[k].allocate( img1, img2, labelmap, region_index[k], regionLen, regionSize );
		std::cout<<"**** Estimate forward map and occlusion for **** "<<k<<"th Region ***"<<std::endl;
		std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
		myregionsplus[k].estimate_warp_and_occlusion( true, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, TRANSLATION, INITIAL_ITERATIONS, MAX_COMBO );
	}
	
	combined_warp( myregionsplus, labelmap, "combined_forward", frameno );
	combined_warp( myregionsminus, labelmap, "combined_backward", frameno );
	
	//*************** refinement start here ***************//
	int labelchanges;
	int iterno = 0;
	uchar *newchange = new uchar[GSize];
	uchar *oldchange = new uchar[GSize];
	for( int p = 0; p < GSize; ++p ) newchange[p] = oldchange[p] = 0; 
	do{
		std::cout<<"***** refinement start!! *****"<<std::endl;
		++iterno;
		labelchanges = 0;
		int *newlabel = new int[GSize];
		update_label_new( myregionsplus, myregionsminus, newlabel, labelmap );
		
		regularize_labelmap( newlabel, GSize, XSize, YSize );
		for( int p = 0; p < GSize; ++p )
		{
			newchange[p] = 0;
			if( labelmap[p] != newlabel[p] ) { labelchanges += 1; newchange[p]=1; }
			labelmap[p] = newlabel[p];
		}
		delete[] newlabel; newlabel=0;
		
		float oscillation_rate = 0;
		for( int p = 0; p < GSize; ++p )if( newchange[p]==1 && oldchange[p]==1 )oscillation_rate += 1.0;
		std::cout<<"**** iteration: "<<iterno<<" ****** label changes : "<<labelchanges<<" **** osscilation: "<<oscillation_rate<<std::endl;
 		oscillation_rate /= (float)labelchanges;
		if( oscillation_rate > OSCILATE )break;
		else for( int p = 0; p < GSize; ++p ) oldchange[p] = newchange[p];
		
		markimages_using_label( img1, labelmap, "labelmap", iterno );
		//save_labelmap( labelmap, maskimg, "tempmask", 0 );
		
		if( labelchanges < STABLE_LABELMAP )break;
		
		for( int k = 0; k < REGIONS; ++k )
		{
			float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
			float regionLen, regionSize;
			myregionsminus[k].initialize2( labelmap, region_index[k], regionLen, regionSize );
			std::cout<<"**** Refine backward map and occlusion for *** "<<k<<"th Region ***"<<std::endl;
			std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
			myregionsminus[k].estimate_warp_and_occlusion( false, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, -1, REFINE_ITERATIONS, MAX_COMBO );
		}
		for( int k = 0; k < REGIONS; ++k )
		{
			float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
			float regionLen, regionSize;
			myregionsplus[k].initialize2( labelmap, region_index[k], regionLen, regionSize );
			std::cout<<"**** Refine forward map and occlusion for *** "<<k<<"th Region ***"<<std::endl;
			std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
			myregionsplus[k].estimate_warp_and_occlusion( false, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, -1, REFINE_ITERATIONS, MAX_COMBO );
		}
	}while( iterno < MAX_REFINEMENTS );
	
	for( int k = 0; k < REGIONS; ++k )
	{
		float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
		float regionLen, regionSize;
		myregionsplus[k].initialize2( labelmap, region_index[k], regionLen, regionSize );
		std::cout<<"Refine forward map and occlusion for *** "<<k<<"th Region ***"<<std::endl;
		std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
		myregionsplus[k].estimate_warp_and_occlusion( false, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, 2, 5, MAX_COMBO );
	}
	markimages_using_label( img1, labelmap, "labelmap", iterno );
	
	//std::cout<<"after smoothing"; char bbbbbb; std::cin>>bbbbbb;
	int statistics_number = 0;
 	do{
		++statistics_number;
		std::cout<<"**** refinement using statistics ****"<<std::endl;
		labelchanges = 0;
		int *newlabel = new int[GSize];
		update_label_new( myregionsplus, myregionsminus, newlabel, labelmap, true );
		
		regularize_labelmap( newlabel, GSize, XSize, YSize );
		for( int p = 0; p < GSize; ++p )
		{
			newchange[p] = 0;
			if( labelmap[p] != newlabel[p] ) { labelchanges += 1; newchange[p]=1; }
			labelmap[p] = newlabel[p];
		}
		delete[] newlabel; newlabel=0;
		
		float oscillation_rate = 0;
		for( int p=0; p<GSize; ++p )if( newchange[p]==1 && oldchange[p]==1 )oscillation_rate += 1.0;
		std::cout<<"****iteration: "<<iterno<<" ******label changes : "<<labelchanges<<" **** osscilation: "<<oscillation_rate<<std::endl;
 		oscillation_rate /= (float)labelchanges;
		if( oscillation_rate > OSCILATE )break;
		else for( int p = 0; p < GSize; ++p ) oldchange[p] = newchange[p];
		
		markimages_using_label( img1, labelmap, "labelmap", iterno );
		//save_labelmap( labelmap, maskimg, "tempmask", 0 );
		
		if( labelchanges<STABLE_factor*STABLE_LABELMAP )break;
 		
		for( int k = 0; k < REGIONS; ++k ) 
		{
			float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
			float regionLen, regionSize;
			myregionsplus[k].initialize2( labelmap, region_index[k], regionLen, regionSize );
			std::cout<<"***** Refine forward map and occlusion for *** "<<k<<"th Region ***"<<std::endl;
			std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
			myregionsplus[k].estimate_warp_and_occlusion( false, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, -1, REFINE_ITERATIONS, MAX_COMBO );
		}
	}while( statistics_number < MAX_STATISTICS );
	
	//std::cout<<"before smoothing"; char ccccc; std::cin>>ccccc;
	extract_the_largest_component( labelmap, GSize, XSize, YSize );
	smoothing_labelmap( labelmap, Smoothing_contour, Threshold_for_classifition, myregionsplus );
	extract_the_largest_component( labelmap, GSize, XSize, YSize );

	for( int k = 0; k < REGIONS; ++k )
	{
		float HISTOGRAM_SUPPORT = ( region_index[k]==0 ? HISTOGRAM_SUPPORT0 : HISTOGRAM_SUPPORT1 );
		float regionLen, regionSize;
		myregionsplus[k].initialize2( labelmap, region_index[k], regionLen, regionSize );
		std::cout<<"***** Refine forward map and occlusion for *** "<<k<<"th Region ***"<<std::endl;
		std::cout<<" ObjLen: "<<regionLen<<"  ObjSize: "<<regionSize<<std::endl;
		myregionsplus[k].estimate_warp_and_occlusion( false, regionLen*HISTOGRAM_SUPPORT, regionSize*HISTOGRAM_SUPPORT, 3, 6, MAX_COMBO );
	}
	markimages_using_label( img1, labelmap, "labelmap", iterno );
	
	delete[] newchange; newchange=0;
	delete[] oldchange; oldchange=0;
    
	regularize_labelmap( labelmap, GSize, XSize, YSize );
	markimages_using_label( img1, labelmap, "segmentation", frameno );
	//save_labelmap( labelmap, maskimg, "mask", frameno );
    write_array_to_text(labelmap, XSize, YSize, frameno, "labels");

    
#ifdef WRITE_DATA_FOR_BRIAN
    double *combined_f = new double[2*myregionsplus[0].GridSize];
    double *combined_b = new double[2*myregionsplus[0].GridSize];
    double *combined_fr = new double[2*myregionsplus[0].GridSize];
    double *combined_br = new double[2*myregionsplus[0].GridSize];
    for( int p = 0; p < myregionsplus[0].GridSize; ++p )
    {
        int regionId = 0;
        for( int k = 0; k < REGIONS; ++k )
            if( region_index[k] == labelmap[p] )
                { regionId = k; break; }
        combined_f[2*p+0] = myregionsplus[regionId].forward_map[2*p+0]-p%XSize;
        combined_f[2*p+1] = myregionsplus[regionId].forward_map[2*p+1]-p/XSize;
        combined_b[2*p+0] = myregionsminus[regionId].forward_map[2*p+0]-p%XSize;
        combined_b[2*p+1] = myregionsminus[regionId].forward_map[2*p+1]-p/XSize;
        combined_fr[2*p+0] = myregionsplus[regionId].backward_map[2*p+0]-p%XSize;
        combined_fr[2*p+1] = myregionsplus[regionId].backward_map[2*p+1]-p/XSize;
        combined_br[2*p+0] = myregionsminus[regionId].backward_map[2*p+0]-p%XSize;
        combined_br[2*p+1] = myregionsminus[regionId].backward_map[2*p+1]-p/XSize;
    }
    write_array_to_text_double( combined_f, 2*XSize, YSize, frameno, "uvf" );
    write_array_to_text_double( combined_b, 2*XSize, YSize, frameno, "uvb" );
    write_array_to_text_double( combined_fr, 2*XSize, YSize, frameno, "uvfr" );
    write_array_to_text_double( combined_br, 2*XSize, YSize, frameno, "uvbr" );
    write_array_to_text_double( myregionsminus[0].image1, 3*XSize, YSize, frameno, "img1" );
    write_array_to_text_double( myregionsminus[0].image0, 3*XSize, YSize, frameno, "img2" );
    write_array_to_text_double( myregionsplus[0].image1,  3*XSize, YSize, frameno, "img3" );
    delete[] combined_f; combined_f = new double[myregionsplus[0].GridSize];
    delete[] combined_b; combined_b = new double[myregionsplus[0].GridSize];
    for( int p=0; p<myregionsplus[0].GridSize; ++p ) combined_f[p] = combined_b[p] = 0;
    for( int k = 0; k < REGIONS; ++k )
    {
        for( int p=0; p<myregionsplus[0].GridSize; ++p )
        {
            if( myregionsplus[k].region_indicator[p] > 0.5 && myregionsplus[k].occlusion_map[p] == 1 ) combined_f[p] = 1;
            if( myregionsminus[k].region_indicator[p] > 0.5 && myregionsminus[k].occlusion_map[p] == 1 ) combined_b[p] = 1;
        }
    }
    write_array_to_text_double( combined_f, XSize, YSize, frameno, "occf" );
    write_array_to_text_double( combined_b, XSize, YSize, frameno, "occb" );
    delete[] combined_f; combined_f = 0;
    delete[] combined_b; combined_b = 0;
    delete[] combined_fr; combined_fr = 0;
    delete[] combined_br; combined_br = 0;
#endif
    
    
    
    
    cv::Mat occmat;
	hole_indicator.copyTo(occmat);
	cv::MatIterator_<uchar> ito = occmat.begin<uchar>();
    int *warped_label = new int[GSize];
    for( int p=0; p<GSize; ++p )
    {
        warped_label[p] = -1;
        *(ito+p)=0;
    }
	for( int k=0; k<REGIONS; ++k )
	{
		double *reg1 = new double[GSize];
		double *reg2 = new double[GSize];
 		double *reg3 = new double[GSize];
		double *reg4 = new double[GSize];
		
		for( int p=0; p<GSize; ++p )
		{
            reg1[p] = myregionsplus[k].region_indicator[p] > 0.5 && myregionsplus[k].occlusion_map[p] == 0 ? 1.0 : 0;//covisible
            reg3[p] = myregionsplus[k].region_indicator[p] > 0.5 ? 1.0 : 0;//region
			reg2[p] = 0;
			reg4[p] = 0;
		}
		myregionsplus[k].get_evolved_image( reg1, myregionsplus[k].backward_map, reg2, 1 );//warped covisible
		myregionsplus[k].get_evolved_image( reg3, myregionsplus[k].backward_map, reg4, 1 );//warped region
		for( int p=0; p<GSize; ++p )
        {
			if( reg2[p] > 0.5 ) warped_label[p] = region_index[k];
			if( reg2[p] <= 0.5 && reg4[p] > 0.5 ) *(ito+p) = 255;
		 }
		 
		delete[] reg1; reg1=0;
		delete[] reg2; reg2=0;
		delete[] reg3; reg3=0;
		delete[] reg4; reg4=0;
	}
	frameno += NEXT_FRAME;
    write_array_to_text(warped_label, XSize, YSize, frameno, LABELMAP_NAME );
    delete[] warped_label; warped_label = 0;

    int bbbb = frameno/1000, bbb = ( frameno - 1000*bbbb )/100, bb = ( frameno - 1000*bbbb - 100*bbb )/10, b = frameno%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
    std::string dddname = "transformed_occlusion";
	dddname = dddname+aaaa+aaa+aa+a+".png";
	cv::imwrite( dddname, occmat );
	occmat.release();
    
	delete[] labelmap; labelmap=0;
	img0.release();
	img1.release();
	img2.release();
	for( int k = 0; k < REGIONS; ++k )
	{
		myregionsminus[k].deallocate();
		myregionsplus[k].deallocate();
	}
	delete[] myregionsminus; myregionsminus=0;
	delete[] myregionsplus; myregionsplus=0;
	hole_indicator.release();
    disocc_indicator.release();
}

void update_label_new( region_solver *regionPlus, region_solver *regionMinus, int *newlabel, int *oldlabel, bool onlyintensity )
{
    cv::MatIterator_<uchar> ithole = hole_indicator.begin<uchar>();
    cv::MatIterator_<uchar> itdis = disocc_indicator.begin<uchar>();
	int GridSize = regionPlus[0].GridSize;
	for( int p = 0; p < GridSize; ++p ) newlabel[p] = oldlabel[p];
	//***** label assignment for each p *****//
	float max_shape_difference = 0;
	double *Rar = new double[REGIONS]; //records ratio of k's affect p
	double *Err = new double[REGIONS];
	double *Qee = new double[REGIONS];
	int	  *Kar  = new int[REGIONS];
	for( int p = 0; p < GridSize; ++p )
	{
		for( int k = 0; k < REGIONS; ++k ){ Rar[k] = Err[k] = Qee[k] = -1; Kar[k] = -1; }
		int	  candidateNo = 0;
		bool constreg = false;
		float min_shape = 100;
		float max_shape = -100;
		for( int k = 0; k < REGIONS; ++k )
		{
			if( regionPlus[k].F_ratio[p] > -10 )
			{
				if( regionPlus[k].F_shape[p] < min_shape ) min_shape = regionPlus[k].F_shape[p];
				if( regionPlus[k].F_shape[p] > max_shape ) max_shape = regionPlus[k].F_shape[p];
				Rar[candidateNo] = regionPlus[k].F_ratio[p] + 1.0e-10;
				Err[candidateNo] = MIN( sqrt(regionPlus[k].F_error[p]),\
										sqrt(regionMinus[k].F_error[p]) );
				Kar[candidateNo] = k;
				candidateNo += 1;
				if( regionPlus[k].ConstRegion[p] > 0 ) constreg = true;
			}
		}
		if( candidateNo < 1 ) continue; // p is not on the boundary, label is oldlabel
		if( max_shape - min_shape > MAX_SPDIFF ) continue;
		if( max_shape - min_shape > max_shape_difference ) max_shape_difference = max_shape - min_shape;
		double minERROR = FLT_MAX;
		for( int j = 0; j < candidateNo; ++j ) minERROR = MIN( minERROR, Err[j] );
		for( int j = 0; j < candidateNo; ++j )
		{
			double step_p;
			double t = Err[j] - minERROR;
			if( t > 1.25*ERROR_MARGIN ) step_p = 0;
			else if( t >= 0.75*ERROR_MARGIN && t <= 1.25*ERROR_MARGIN )
				step_p = 1 - (t - 0.75*ERROR_MARGIN)/(0.5*ERROR_MARGIN);
			else step_p = 1.0;
			Qee[j] = Rar[j]*step_p;
			if( *(ithole+p) > 100 ) Qee[j] = Rar[j];
			if( onlyintensity ) Qee[j] = Rar[j];
			if( constreg ) Qee[j] = Rar[j];
			if( minERROR > BOTH_OCCLUSION ) Qee[j] = Rar[j];
		}
		double maxQ = -FLT_MAX;
		int ind_maxQ = -1;
		for( int j = 0; j < candidateNo; ++j )
		{
			if( Qee[j] > maxQ ){ maxQ = Qee[j]; ind_maxQ = j; }
		}
        if( ind_maxQ == -1 )
        {
            for( int j = 0; j < candidateNo; ++j )
            {
                std::cout<<" Qee["<<j<<"] : "<<Qee[j]<<std::endl;
                std::cout<<" Err["<<j<<"] : "<<Err[j]<<std::endl;
                std::cout<<" Rar["<<j<<"] : "<<Rar[j]<<std::endl;
            }
            std::cout<<"somthing wrong in update_label"<<std::endl<<" candidateNo: "<<candidateNo<<std::endl; while(1);
        }
		
		double secondQ = -FLT_MAX;
		for( int j = 0; j < candidateNo; ++j )
		{
			if( j == ind_maxQ ) continue;
			if( Qee[j] > secondQ ){ secondQ = Qee[j]; }
		}
		if( *(itdis+p)==0 )//disocclusion
		{
			if( secondQ / maxQ < DISCRIMINATE_RATIO  )
				newlabel[p] = region_index[ Kar[ind_maxQ] ];
		}
		else
		{
			if( secondQ / maxQ < DISCRIMINATE_RATIO2  )
				newlabel[p] = region_index[ Kar[ind_maxQ] ];
		}
	}
	delete[] Rar; Rar = 0;
	delete[] Err; Err = 0;
	delete[] Qee; Qee = 0;
	delete[] Kar; Kar = 0;
	std::cout<<" max_shape_difference is : "<<max_shape_difference<<std::endl;
}

void smoothing_labelmap( int *labelmap, int windowsize, float threshold, region_solver *regplus )
{
	int GSize = regplus[0].GridSize;
	float *src_labelmap = new float[GSize];
	float *des_labelmap = new float[GSize];
	float *regind = new float[GSize]; for( int p=0; p<GSize; ++p )regind[p] = 1.0;
	int *temp_newlabel = new int[GSize]; for( int p=0; p<GSize; ++p ) temp_newlabel[p]=0;

	for( int k=1; k<REGIONS; ++k )
	{
		for( int p=0; p<GSize; ++p ) src_labelmap[p] = labelmap[p]== region_index[k] ? 1.0 : 0.0;
		regplus[0].residueblur( src_labelmap, des_labelmap, windowsize, regind );
		for( int p=0; p<GSize; ++p )
		{
			if( des_labelmap[p]>threshold )temp_newlabel[p] = region_index[k];
		}
	}
	delete[] src_labelmap; src_labelmap = 0;
	delete[] des_labelmap; des_labelmap = 0;
	delete[] regind; regind = 0;
	for( int p=0; p<GSize; ++p ) labelmap[p] = temp_newlabel[p];
	delete[] temp_newlabel; temp_newlabel = 0;
}