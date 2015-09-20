#include <stdio.h>
#include "array_to_matrix.h"

int write_array_to_text( int *labels, int XSize, int YSize, int frameno, std::string filename )
{
    int bbbb = frameno/1000;
    int bbb = ( frameno - 1000*bbbb )/100;
    int bb = ( frameno - 1000*bbbb - 100*bbb )/10;
    int b = frameno%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
    
    std::string myfilename = filename + aaaa + aaa + aa + a + ".txt";
    std::ofstream myfile( myfilename );
    
    myfile<<XSize<<std::endl;
    myfile<<YSize<<std::endl;
    for( int p=0; p<XSize*YSize; ++p )
    {
        myfile<<labels[p]<<std::endl;
    }
    myfile.close();
    std::cout<<"File: "<<myfilename<<" has been created!"<<std::endl;
    return 1;
}

int write_array_to_text_double( double *labels, int XSize, int YSize, int frameno, std::string filename )
{
    int bbbb = frameno/1000;
    int bbb = ( frameno - 1000*bbbb )/100;
    int bb = ( frameno - 1000*bbbb - 100*bbb )/10;
    int b = frameno%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;
    
    std::string myfilename = filename + aaaa + aaa + aa + a + ".txt";
    std::ofstream myfile( myfilename );
    
    myfile<<XSize<<std::endl;
    myfile<<YSize<<std::endl;
    for( int p=0; p<XSize*YSize; ++p )
    {
        myfile<<std::setprecision(5)<<labels[p]<<std::endl;
    }
    myfile.close();
    std::cout<<"File: "<<myfilename<<" has been created!"<<std::endl;
    return 1;
}

int get_array_from_text( int *labels, int &XSize, int &YSize, int frameno, std::string filename )
{
    int bbbb = frameno/1000;
    int bbb = ( frameno - 1000*bbbb )/100;
    int bb = ( frameno - 1000*bbbb - 100*bbb )/10;
    int b = frameno%10;
    char aaaa = '0'+bbbb, aaa = '0'+bbb, aa = '0'+bb, a = '0'+b;

    std::string myfilename = filename + aaaa + aaa + aa + a + ".txt";
    std::ifstream myfile( myfilename, std::ifstream::in );
    
    if( !myfile ){ std::cout<<"Failed to read file "<<myfilename<<std::endl; return 0; }
    
    myfile>>XSize;
    myfile>>YSize;
    
    int temp;
    int p = 0;
    while( myfile >> temp )
    {
        labels[p] = temp;
        ++p;
    }
    
    if( p == XSize*YSize ) return 1;
    else
    {
        std::cout<<"Error in get_array_from_text!"<<std::endl;
        while(1);
    }
}

void FindBlobs( const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs )
{
    blobs.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    cv::Mat label_image;
    binary.convertTo( label_image, CV_32SC1 );
    
    int label_count = 2; // starts at 2 because 0,1 are used already
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }
            
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            
            std::vector <cv::Point2i> blob;
            
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    
                    blob.push_back(cv::Point2i(j,i));
                }
            }
            
            blobs.push_back(blob);
            
            label_count++;
        }
    }
}
