#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "cvsettings.h"
#include "multi_object_tracking.h"

int main( int argc, char **argv )
{
    int starting_frame, ending_frame;
//	std::cout<<"starting_frame: "; std::cin>>starting_frame;
//	std::cout<<"ending_frame: ";   std::cin>>ending_frame;
    starting_frame = atoi( argv[1] );
    ending_frame = atoi( argv[2] );
    
    read_parameters();
    int NEXT_FRAME = get_next_frame();
    
    Timer tstart;
    
    if( starting_frame <= ending_frame )
    {
        for( int frameno = starting_frame; frameno <= ending_frame; frameno += NEXT_FRAME )
        {
            //Timer tstart;

            
            std::string imname0 = VIDEO_NAME;
            std::string imname1 = VIDEO_NAME;
            std::string imname2 = VIDEO_NAME;
            std::string maskname = MASKNAME;
            std::string labelname = LABELMAP_NAME;
				
            int bbbb, bbb, bb, b; char aaaa, aaa, aa, a;
				
            int tempframe = frameno - NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname0 = imname0 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            tempframe = frameno;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname1 = imname1 + aaaa + aaa + aa + a + VIDEO_SSD;
            maskname = maskname + aaaa + aaa + aa + a + ".png";
            labelname = labelname + aaaa + aaa + aa + a + ".txt";
            
            tempframe = frameno + NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname2 = imname2 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            std::ifstream mylabel( labelname, std::ifstream::in );
            if( !mylabel )
            {
                std::cout<<"The file "<<labelname<<" doesn't exist, need to create one.. "<<std::endl;
                get_initial_labels_from_mask(frameno, maskname);
            }
            mylabel.close();
            
            multi_object_tracking_and_segmentation( imname0, imname1, imname2, LABELMAP_NAME, frameno );
            
            //std::cout<<"time used: "<<tstart.elapsed().count()<<std::endl;while(1);
        }
    }
    else
    {
        for( int frameno = starting_frame; frameno >= ending_frame; frameno += NEXT_FRAME )
        {
            std::string imname0 = VIDEO_NAME;
            std::string imname1 = VIDEO_NAME;
            std::string imname2 = VIDEO_NAME;
            std::string maskname = MASKNAME;
            std::string labelname = LABELMAP_NAME;
				
            int bbbb, bbb, bb, b; char aaaa, aaa, aa, a;
            
            int tempframe = frameno - NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname0 = imname0 + aaaa + aaa + aa + a + VIDEO_SSD;
				
            tempframe = frameno;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname1 = imname1 + aaaa + aaa + aa + a + VIDEO_SSD;
            maskname = maskname + aaaa + aaa + aa + a + ".png";
            labelname = labelname + aaaa + aaa + aa + a + ".txt";
            
            tempframe = frameno + NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname2 = imname2 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            std::ifstream mylabel( labelname, std::ifstream::in );
            if( !mylabel )
            {
                std::cout<<"The file "<<labelname<<" doesn't exist, need to create one.. "<<std::endl;
                get_initial_labels_from_mask(frameno, maskname);
            }
            mylabel.close();

            multi_object_tracking_and_segmentation( imname0, imname1, imname2, LABELMAP_NAME, frameno );
        }
    }
    
    std::cout<<"time used: "<<tstart.elapsed().count()/1000<<" secs"<<std::endl;
    
	return 0;
}


/*
int main()
{
    int starting_frame, ending_frame;
    std::cout<<"starting_frame: "; std::cin>>starting_frame;
    std::cout<<"ending_frame: ";   std::cin>>ending_frame;
    
    read_parameters();
    int NEXT_FRAME = get_next_frame();
    
    Timer tstart;
    
    if( starting_frame <= ending_frame )
    {
        for( int frameno = starting_frame; frameno <= ending_frame; frameno += NEXT_FRAME )
        {
            //Timer tstart;
            
            
            std::string imname0 = VIDEO_NAME;
            std::string imname1 = VIDEO_NAME;
            std::string imname2 = VIDEO_NAME;
            std::string maskname = MASKNAME;
            std::string labelname = LABELMAP_NAME;
            
            int bbbb, bbb, bb, b; char aaaa, aaa, aa, a;
            
            int tempframe = frameno - NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname0 = imname0 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            tempframe = frameno;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname1 = imname1 + aaaa + aaa + aa + a + VIDEO_SSD;
            maskname = maskname + aaaa + aaa + aa + a + ".png";
            labelname = labelname + aaaa + aaa + aa + a + ".txt";
            
            tempframe = frameno + NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname2 = imname2 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            std::ifstream mylabel( labelname, std::ifstream::in );
            if( !mylabel )
            {
                std::cout<<"The file "<<labelname<<" doesn't exist, need to create one.. "<<std::endl;
                get_initial_labels_from_mask(frameno, maskname);
            }
            mylabel.close();
            
            multi_object_tracking_and_segmentation( imname0, imname1, imname2, LABELMAP_NAME, frameno );
            
            //std::cout<<"time used: "<<tstart.elapsed().count()<<std::endl;while(1);
        }
    }
    else
    {
        for( int frameno = starting_frame; frameno >= ending_frame; frameno += NEXT_FRAME )
        {
            std::string imname0 = VIDEO_NAME;
            std::string imname1 = VIDEO_NAME;
            std::string imname2 = VIDEO_NAME;
            std::string maskname = MASKNAME;
            std::string labelname = LABELMAP_NAME;
            
            int bbbb, bbb, bb, b; char aaaa, aaa, aa, a;
            
            int tempframe = frameno - NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname0 = imname0 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            tempframe = frameno;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname1 = imname1 + aaaa + aaa + aa + a + VIDEO_SSD;
            maskname = maskname + aaaa + aaa + aa + a + ".png";
            labelname = labelname + aaaa + aaa + aa + a + ".txt";
            
            tempframe = frameno + NEXT_FRAME;
            bbbb = tempframe/1000; bbb = ( tempframe - 1000*bbbb )/100; bb = ( tempframe - 1000*bbbb - 100*bbb)/10; b = tempframe%10;
            aaaa = '0'+bbbb; aaa = '0'+bbb; aa = '0'+bb; a = '0'+b;
            imname2 = imname2 + aaaa + aaa + aa + a + VIDEO_SSD;
            
            std::ifstream mylabel( labelname, std::ifstream::in );
            if( !mylabel )
            {
                std::cout<<"The file "<<labelname<<" doesn't exist, need to create one.. "<<std::endl;
                get_initial_labels_from_mask(frameno, maskname);
            }
            mylabel.close();
            
            multi_object_tracking_and_segmentation( imname0, imname1, imname2, LABELMAP_NAME, frameno );
        }
    }
    
    std::cout<<"time used: "<<tstart.elapsed().count()/1000<<" secs"<<std::endl;while(1);
    
    return 0;
}
*/