#include "narrowband.h"
#include <cmath>
#include <string>
#include <cstdlib>

#define Diag_Dist 2.0

int narrowband::allocate( int _XSize, int _YSize )
{
  int gridsize=_XSize*_YSize;
  if (GridSize!=gridsize)
  {
    deallocate();
    GridSize=gridsize;
    if (!( Psi       = new          double[GridSize]) ||
		!( initial_Psi = new        double[GridSize]) ||
		!( newPsi    = new          double[GridSize]) ||
        !( distance  = new          double[GridSize]) ||
		!( norm_x    = new          double[GridSize]) ||
		!( norm_y    = new          double[GridSize]) ||
        !( band1     = new             int[GridSize]) ||
        !( band2     = new             int[GridSize]) ||
		!( band3     = new             int[GridSize]) ||
        !( inband    = new   unsigned char[GridSize]) ||
        !( edgebits  = new   unsigned char[GridSize]) ||
		!( dirbits   = new             int[GridSize]) )
	{
      deallocate();
      return -1;
    }
  }

  XSize=_XSize; YSize=_YSize;
  initialize();

  return 1;
}

void narrowband::deallocate() {
  delete[] Psi;       Psi=0;
  delete[] initial_Psi;       initial_Psi=0;
  delete[] newPsi;    newPsi=0;
  delete[] distance;  distance=0;
  delete[] norm_x;    norm_x=0;
  delete[] norm_y;    norm_y=0;
  delete[] band1;     band1=0;
  delete[] band2;     band2=0;
  delete[] band3;     band3=0;
  delete[] inband;    inband=0;
  delete[] edgebits;  edgebits=0;
  delete[] dirbits;   dirbits=0;
  
  band=0;      tail=0;
  edgeband=0;  edgetail=0;
  nhbrband=0;  nhbrtail=0;
  interior=0;  intetail=0;
  contour=0;   contail=0;
  GridSize=XSize=YSize=0;
}

void narrowband::initialize() {
	
  for (int p=0; p<GridSize; ++p) 
  {
	  Psi[p]       = 0;
	  initial_Psi[p] = 0;
	  newPsi[p]    = 0;
	  distance[p]  = 0;
	  norm_x[p]    = 0;
	  norm_y[p]    = 0;
	  band1[p]     = 0;
	  band2[p]     = 0;
	  band3[p]     = 0;
	  inband[p]    = 0;
	  edgebits[p]  = 0;
	  dirbits[p]   = -1;
  }
  band=band1; edgeband=band1+GridSize-1; nhbrband=band2;    interior=band3+GridSize-1;
  tail=band;  edgetail=edgeband;         nhbrtail=nhbrband; intetail=interior;
  contour=band3;       
  contail=contour;
  markEdges();
}

void narrowband::markEdges() {
  int p;
  //Y faces
  for (p=0; p<XSize; p+=X) 
  {
      edgebits[p]|=YNEG;
      edgebits[GridSize-1-p]|=YPOS;
  }
  //X faces
  for (p=0; p<GridSize; p+=Y) 
  {
      edgebits[p]|=XNEG;
      edgebits[XSize-1+p]|=XPOS;
  }
  // std::cout<<"markedge done"<<std::endl;
}

int narrowband::makeLSF(cv::Mat mask) 
{
	if( mask.depth() != CV_8U ){std::cout<<"mask is not uchar type(makeLSF)...";while(1);}
	if( mask.channels() != 1 ) {std::cout<<"use gray image as mask(makeLSF)...";while(1);}
  
    cv::MatIterator_<uchar> it, end;
    it  = mask.begin<uchar>();
    end = mask.end<uchar>();

    float foreground = 0, background = 0;

    for( int p = 0 ; ( it != end ) && ( p < GridSize ); ++p, ++it )
    {
	    if( *it == 255 ) { initial_Psi[p] = -1.0 ; Psi[p]= -1.0; ++foreground; }
	    else { initial_Psi[p] = 1.0 ; Psi[p] = 1.0; ++background; }
    }
	
	if( foreground == 0 ) { std::cout<<"there is no foreground(makeLSF).."; while(1); };
    
	//std::cout<<"foreground ratio: "<<foreground/(foreground+background)<<std::endl;
    
	return 1;
}

int narrowband::createBand()
{ 
	int p;               //Index for current grid point
    double deltaPsi;      //Change in LS function between neighboring grid points
    double dist;          //Distance to be propagated

    //Initialize narrowband points

    tail = band; edgetail = edgeband; intetail = interior;
    for(int i = 0; i < GridSize; ++i ) inband[i] = 0;
  
  
    //Enqueue and compute distances for points near ZLS

    //(search entire grid for ZLS)

    int *extdir = dirbits;  //Directions in which to look for extensions
	int bandno = 0;
    #define ENQUEUE(p,dir) if (edgebits[p]) *edgetail-- = p; else *tail++ = p; \
                         inband[p] = 1 ; distance[p] = dist ; extdir[p] = dir; ++bandno
    #define MODIFY(p,dir)  distance[p]=dist; extdir[p]=dir
    

	int intepoints = 0;
    for ( p = GridSize-1 ; p >= 0 ; --p )
    {
		if( Psi[p] < 0 ){ *intetail-- = p; ++intepoints;}

        if ( !( edgebits[p] & XPOS ) && ( Psi[p] < 0 ) != ( Psi[p+X] < 0 ) )
	    {
		    deltaPsi = Psi[p+X] - Psi[p];
		    dist = -Psi[p] / deltaPsi;

		    if ( !inband[p] ) { ENQUEUE(p,XPOS); } 
		    else if ( dist < distance[p] ) { MODIFY(p,XPOS); }

		    dist = 1.0 - dist;

		    if ( !inband[p+X] ) { ENQUEUE(p+X,XNEG); } 
		    else if ( dist < distance[p+X] ) { MODIFY(p+X,XNEG); }
	    }
 
        if ( !(edgebits[p] & YPOS ) && ( Psi[p] < 0 ) != ( Psi[p+Y] < 0 ) )
	    {
		    deltaPsi = Psi[p+Y] - Psi[p];
		    dist = -Psi[p] / deltaPsi;

		    if ( !inband[p] ) { ENQUEUE(p,YPOS); } 
		    else if ( dist < distance[p] ) { MODIFY(p,YPOS); }

		    dist = 1.0 - dist;

		    if ( !inband[p+Y] ) { ENQUEUE(p+Y,YNEG); } 
		    else if ( dist < distance[p+Y] ) { MODIFY(p+Y,YNEG); }
	    }
    }
    if( bandno == 0 ){ std::cout<<"no band points found (createBand).."<<std::endl; while(1);}
//    std::cout<<"band points :"<<bandno<<std::endl;
//	std::cout<<"interior points is: "<<intepoints<<std::endl;

    #undef ENQUEUE
    #undef MODIFY

    //Initialize neighbor point list

    nhbrtail = nhbrband;

    //Propagate distances to neighbors of band points

    int *extorig = dirbits;  //Grid points from which extensions originate
	int neighbourno = 0;
    #define ENQUEUE(p,q) distance[p] = dist ; inband[p] = 2 ; *nhbrtail++ = p; \
                         extorig[p] = q ; ++neighbourno
    #define MODIFY(p,q)  distance[p] = dist ; extorig[p] = q
   

    for ( BANDPOINTS(p) ) 
	{
		//Adjacent neighbors
		if( distance[p] > 1 || distance[p] < 0 )
		{
			std::cout<<"impossible value for bandpoint(createBand)..";
			while(1);
		}

		dist = distance[p] + 1.0;

		if ( !inband[p+X] ) { ENQUEUE(p+X,p); } 
		else if ( dist < distance[p+X] ) { MODIFY(p+X,p); }

		if ( !inband[p-X] ) { ENQUEUE(p-X,p); }
		else if ( dist < distance[p-X] ) { MODIFY(p-X,p); }

		if ( !inband[p+Y] ) { ENQUEUE(p+Y,p); }
		else if ( dist < distance[p+Y] ) { MODIFY(p+Y,p); }

		if ( !inband[p-Y] ) { ENQUEUE(p-Y,p); }
		else if ( dist < distance[p-Y] ) { MODIFY(p-Y,p); }

		//Diagonal neighbors
    
		dist = distance[p] + Diag_Dist;

		if ( !inband[p+X+Y] ) { ENQUEUE(p+X+Y,p); }
		else if ( dist < distance[p+X+Y] ) { MODIFY(p+X+Y,p); }

		if ( !inband[p+X-Y] ) { ENQUEUE(p+X-Y,p); }
		else if ( dist < distance[p+X-Y] ) { MODIFY(p+X-Y,p); }

		if ( !inband[p-X+Y] ) { ENQUEUE(p-X+Y,p); } 
		else if ( dist < distance[p-X+Y] ) { MODIFY(p-X+Y,p); }

		if ( !inband[p-X-Y] ) { ENQUEUE(p-X-Y,p); }
		else if ( dist < distance[p-X-Y] ) { MODIFY(p-X-Y,p); }
    } 

    for ( EDGEPOINTS(p) )
    {
		//Adjacent neighbors
		if( distance[p] > 1 || distance[p] < 0 )
		{
			std::cout<<"impossible value for edgepoint(createBand)..";
			while(1);
		}

		dist = distance[p] + 1.0;

		if ( !( edgebits[p] & XPOS ) ) 
		{
			if ( !inband[p+X] ) { ENQUEUE(p+X,p); }
			else if ( dist < distance[p+X] ) { MODIFY(p+X,p); }
		}
		if ( !( edgebits[p] & XNEG ) ) 
		{
			if ( !inband[p-X] ) { ENQUEUE(p-X,p); }
			else if ( dist < distance[p-X] ) { MODIFY(p-X,p); }
		}
		if ( !( edgebits[p] & YPOS ) ) 
		{
			if ( !inband[p+Y] ) { ENQUEUE(p+Y,p); }
			else if ( dist < distance[p+Y] ) { MODIFY(p+Y,p); } 
		}
		if ( !( edgebits[p] & YNEG ) ) 
		{
			if ( !inband[p-Y] ) { ENQUEUE(p-Y,p); }
			else if ( dist < distance[p-Y] ) { MODIFY(p-Y,p); }
		}

		//Diagonal neighbors
    
		dist = distance[p] + Diag_Dist;

		if ( !( edgebits[p] & ( XPOS | YPOS ) ) )
		{
			if ( !inband[p+X+Y] ) { ENQUEUE(p+X+Y,p); }
			else if ( dist < distance[p+X+Y] ) { MODIFY(p+X+Y,p); }
		}
		if ( !( edgebits[p] & ( XPOS | YNEG ) ) )
		{
			if ( !inband[p+X-Y] ) { ENQUEUE(p+X-Y,p); }
			else if ( dist < distance[p+X-Y] ) { MODIFY(p+X-Y,p); }
		}
		if ( !( edgebits[p] & ( XNEG | YPOS ) ) ) 
		{
			if ( !inband[p-X+Y] ) { ENQUEUE(p-X+Y,p); }
			else if ( dist < distance[p-X+Y] ) { MODIFY(p-X+Y,p); }
		}
		if ( !( edgebits[p] & ( XNEG | YNEG ) ) )
		{
			if ( !inband[p-X-Y] ) { ENQUEUE(p-X-Y,p); }
			else if ( dist < distance[p-X-Y] ) { MODIFY(p-X-Y,p); }
		}
    }
	if( neighbourno == 0 )
    {
        std::cout<<"no neighbour points found (createBand).."<<std::endl;
	    while(1);
    }
//	std::cout<<"neighbour points: "<<neighbourno<<std::endl;
  
	#undef ENQUEUE
	#undef MODIFY

	for ( BANDPOINTS(p) ) { Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p]; } 
	for ( EDGEPOINTS(p) ) { Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p]; }
	for ( NHBRPOINTS(p) ) { Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p]; }

    return 1;
}

void narrowband::createContour()
{
	contail = contour;
	for ( BANDPOINTS(p) )
	{
		if ( Psi[p] < 0 ) *contail++ = p;
	}
	for ( EDGEPOINTS(p) )
	{
		if ( Psi[p] < 0 ) *contail++ = p;
	}
}

void narrowband::get_norm()
{
	for( CONTPOINTS(p) )
	{
		double psix = 0 , psiy = 0;
		psix = edgebits[p] & XNEG ? (Psi[p+X]-Psi[p]) : ( edgebits[p] & XPOS ? ( Psi[p]-Psi[p-X]) : ( (Psi[p+X]-Psi[p-X]) / 2.0 ) );
		psiy = edgebits[p] & YNEG ? (Psi[p+Y]-Psi[p]) : ( edgebits[p] & YPOS ? ( Psi[p]-Psi[p-Y]) : ( (Psi[p+Y]-Psi[p-Y]) / 2.0 ) );
		double arclen = sqrt( psix*psix + psiy*psiy );
		if ( arclen < 1.0e-10 )
		{
			norm_x[p] = 0;
			norm_y[p] = 0;
		}
		else
		{
			norm_x[p] = psix / arclen;
			norm_y[p] = psiy / arclen;
		}
	}
}

void narrowband::extendBand()
{
	//Add neighbors of band points which have changed sign to narrowband
	int *newtail = tail;
    int *newedgetail = edgetail;

    #define ENQUEUE(p) if ( edgebits[p] ) *newedgetail-- = p ;\
                       else *newtail++ = p

    for ( BANDPOINTS(p) )
    {
      if ( ( Psi[p] < 0 ) != ( newPsi[p] < 0 ) )
	  {
        if ( inband[p+X] == 2 ) { ENQUEUE(p+X); }
        if ( inband[p-X] == 2 ) { ENQUEUE(p-X); }
        if ( inband[p+Y] == 2 ) { ENQUEUE(p+Y); }
        if ( inband[p-Y] == 2 ) { ENQUEUE(p-Y); } 
      }
    }

    for ( EDGEPOINTS(p) )
    {
      if ( ( Psi[p] < 0 ) != ( newPsi[p] < 0 ) )
	  {
        if ( !( edgebits[p] & XPOS ) && inband[p+X] == 2 ) { ENQUEUE(p+X); }
        if ( !( edgebits[p] & XNEG ) && inband[p-X] == 2 ) { ENQUEUE(p-X); }
        if ( !( edgebits[p] & YPOS ) && inband[p+Y] == 2 ) { ENQUEUE(p+Y); }
        if ( !( edgebits[p] & YNEG ) && inband[p-Y] == 2 ) { ENQUEUE(p-Y); }
      }
    }

    #undef ENQUEUE

    //Put neighbors of newly added band points temporarily into band list

    int *tmptail = newtail;
    int *tmpedgetail = newedgetail;

    #define NEWBANDPOINTS(p) int *ptr=tail,    p=*ptr ; ptr != newtail;     p = *++ptr
    #define NEWEDGEPOINTS(p) int *ptr=edgetail,p=*ptr ; ptr != newedgetail; p = *--ptr
    #define ENQUEUE(p) if( edgebits[p] ) *tmpedgetail-- = p;\
	   				   else *tmptail++ = p ; inband[p]=3
  
    for (NEWBANDPOINTS(p))
    {
		if ( !inband[p+X] ) { ENQUEUE(p+X); }
		if ( !inband[p-X] ) { ENQUEUE(p-X); }
		if ( !inband[p+Y] ) { ENQUEUE(p+Y); }
		if ( !inband[p-Y] ) { ENQUEUE(p-Y); }
		if ( !inband[p+X+Y] ) { ENQUEUE(p+X+Y); }
		if ( !inband[p+X-Y] ) { ENQUEUE(p+X-Y); }
		if ( !inband[p-X+Y] ) { ENQUEUE(p-X+Y); }
		if ( !inband[p-X-Y] ) { ENQUEUE(p-X-Y); }
    }
  
    for (NEWEDGEPOINTS(p)) 
    {
		if ( !( edgebits[p] & XPOS ) && !inband[p+X] ) { ENQUEUE(p+X); }
		if ( !( edgebits[p] & XNEG ) && !inband[p-X] ) { ENQUEUE(p-X); }
		if ( !( edgebits[p] & YPOS ) && !inband[p+Y] ) { ENQUEUE(p+Y); }
		if ( !( edgebits[p] & YNEG ) && !inband[p-Y] ) { ENQUEUE(p-Y); }
		if ( !( edgebits[p] & (XPOS|YPOS) ) && !inband[p+X+Y] ) { ENQUEUE(p+X+Y); } 
		if ( !( edgebits[p] & (XPOS|YNEG) ) && !inband[p+X-Y] ) { ENQUEUE(p+X-Y); }
		if ( !( edgebits[p] & (XNEG|YPOS) ) && !inband[p-X+Y] ) { ENQUEUE(p-X+Y); }
		if ( !( edgebits[p] & (XNEG|YNEG) ) && !inband[p-X-Y] ) { ENQUEUE(p-X-Y); }
    }

	#undef NEWBANDPOINTS
	#undef NEWEDGEPOINTS
	#undef ENQUEUE
  
	//Compute signed distances for temporarily added points

	double dist;

	#define TMPBANDPOINTS(p) int *ptr = newtail,    p = *ptr ; ptr != tmptail ;     p = *++ptr
	#define TMPEDGEPOINTS(p) int *ptr = newedgetail,p = *ptr ; ptr != tmpedgetail ; p = *--ptr

	int next = 0 ;
	for ( TMPBANDPOINTS(p) ) 
	{
		dist = 5.0;
		next = 0;
		if ( ( inband[p+X] == 2 ) && ( 1.0 + distance[p+X] < dist ) ) { dist = 1.0 + distance[p+X] ; next = X;  }
		if ( ( inband[p-X] == 2 ) && ( 1.0 + distance[p-X] < dist ) ) { dist = 1.0 + distance[p-X] ; next = -X; }
		if ( ( inband[p+Y] == 2 ) && ( 1.0 + distance[p+Y] < dist ) ) { dist = 1.0 + distance[p+Y] ; next = Y;  }
		if ( ( inband[p-Y] == 2 ) && ( 1.0 + distance[p-Y] < dist ) ) { dist = 1.0 + distance[p-Y] ; next = -Y; }
		if ( ( inband[p+X+Y] == 2 ) && ( Diag_Dist + distance[p+X+Y] < dist ) ) { dist = Diag_Dist + distance[p+X+Y] ; next = X+Y;  }
		if ( ( inband[p+X-Y] == 2 ) && ( Diag_Dist + distance[p+X-Y] < dist ) ) { dist = Diag_Dist + distance[p+X-Y] ; next = X-Y;  }
		if ( ( inband[p-X+Y] == 2 ) && ( Diag_Dist + distance[p-X+Y] < dist ) ) { dist = Diag_Dist + distance[p-X+Y] ; next = -X+Y; }
		if ( ( inband[p-X-Y] == 2 ) && ( Diag_Dist + distance[p-X-Y] < dist ) ) { dist = Diag_Dist + distance[p-X-Y] ; next = -X-Y; }
		if ( next == 0 )
		{
			std::cout<<"some errors happened in (extendBand TMPBAND next=0)..";
			while(1);
		}
		else
		{
			if( dist < 1 )
			{
				std::cout<<"there must be an error in(extendBand TMPBAND dist<1)..";
				while(1);
			}
			Psi[p] = Psi[p+next] < 0 ? -dist : dist;
		}
        inband[p]=0;
     }
  
	 for (TMPEDGEPOINTS(p))
	 {
		 dist = 5.0;
		 next = 0;
		 if ( !(edgebits[p]&XPOS)        && inband[p+X] == 2   && 1.0 + distance[p+X] < dist ) { dist = 1.0 + distance[p+X] ; next = X;  }
		 if ( !(edgebits[p]&XNEG)        && inband[p-X] == 2   && 1.0 + distance[p-X] < dist ) { dist = 1.0 + distance[p-X] ; next = -X; }
		 if ( !(edgebits[p]&YPOS)        && inband[p+Y] == 2   && 1.0 + distance[p+Y] < dist ) { dist = 1.0 + distance[p+Y] ; next = Y;  }
		 if ( !(edgebits[p]&YNEG)        && inband[p-Y] == 2   && 1.0 + distance[p-Y] < dist ) { dist = 1.0 + distance[p-Y] ; next = -Y; }
		 if ( !(edgebits[p]&(XPOS|YPOS)) && inband[p+X+Y] == 2 && Diag_Dist + distance[p+X+Y] < dist ) { dist = Diag_Dist + distance[p+X+Y] ; next = X+Y;  }
		 if ( !(edgebits[p]&(XPOS|YNEG)) && inband[p+X-Y] == 2 && Diag_Dist + distance[p+X-Y] < dist ) { dist = Diag_Dist + distance[p+X-Y] ; next = X-Y;  }
		 if ( !(edgebits[p]&(XNEG|YPOS)) && inband[p-X+Y] == 2 && Diag_Dist + distance[p-X+Y] < dist ) { dist = Diag_Dist + distance[p-X+Y] ; next = -X+Y; }
		 if ( !(edgebits[p]&(XNEG|YNEG)) && inband[p-X-Y] == 2 && Diag_Dist + distance[p-X-Y] < dist ) { dist = Diag_Dist + distance[p-X-Y] ; next = -X-Y; }
         if ( next == 0 )
		 {
		 	 std::cout<<"some errors happened in (extendBand TMPEDGE next=0)..";
			 while(1);
		 }
		 else
		 {
			 if( dist < 1 )
			 {
				 std::cout<<"there must be an error in(extendBand TMPEDGE dist<1)..";
				 while(1);
			 }
			 Psi[p] = Psi[p+next] < 0 ? -dist : dist;
		 }
         inband[p]=0;
     }

     #undef TMPBANDPOINTS
     #undef TMPEDGEPOINTS

    //Omit original band points from narrowband list (keep only the new points)

    band=tail; tail=newtail;
    edgeband=edgetail; edgetail=newedgetail;

}

void narrowband::updateBand( )
{

	double deltaPsi;      //Change in LS function between neighboring grid points
    double dist;          //Distance to be propagated

    //Update level set function and unmark points from old "extended" narrowband
    for ( NHBRPOINTS(p) ) inband[p] = 0;
    for ( BANDPOINTS(p) ) { Psi[p] = newPsi[p] ; inband[p]=0; }
    for ( EDGEPOINTS(p) ) { Psi[p] = newPsi[p] ; inband[p]=0; }

    int *oldband = band, *oldtail = tail;
    int *oldedgeband = edgeband, *oldedgetail = edgetail;

    //Initialize new narrowband list

    band = nhbrband ; tail = band;
    edgeband = band + GridSize - 1 ; edgetail = edgeband;

    //Enqueue and compute distances for points near ZLS
    //(only need to search over old extended narrowband)

    int *extdir = dirbits;  //Directions in which to look for extensions
	int bandno = 0;
    #define OLDBANDPOINTS(p) int *ptr = oldband,    p = *ptr ; ptr != oldtail ;    p = *++ptr
    #define OLDEDGEPOINTS(p) int *ptr = oldedgeband,p = *ptr ; ptr != oldedgetail; p = *--ptr
    #define ENQUEUE(p,dir)   if (edgebits[p]) *edgetail-- = p; \
							 else *tail++ = p; \
                             inband[p] = 1; distance[p] = dist; extdir[p] = dir ; ++bandno
    #define MODIFY(p,dir)    distance[p] = dist ; extdir[p] = dir

    for (OLDBANDPOINTS(p))
	{
		if ( ( Psi[p]<0 ) != ( Psi[p+X]<0 ) ) 
		{
			deltaPsi = Psi[p+X] - Psi[p];
			dist = -Psi[p] / deltaPsi;

			if ( !inband[p] ) { ENQUEUE(p,XPOS); }
			else if ( dist < distance[p] ) { MODIFY(p,XPOS); }

			dist = 1.0 - dist;

			if ( !inband[p+X] ) { ENQUEUE(p+X,XNEG); }
			else if ( dist < distance[p+X] ) { MODIFY(p+X,XNEG); }
		}

        if ( ( Psi[p]<0 ) != ( Psi[p+Y]<0 ) )
	    {
		    deltaPsi = Psi[p+Y] - Psi[p];
		    dist = -Psi[p] / deltaPsi;

			if ( !inband[p] ) { ENQUEUE(p,YPOS); }
			else if ( dist < distance[p] ) { MODIFY(p,YPOS); }

			dist = 1.0 - dist;

			if ( !inband[p+Y] ) { ENQUEUE(p+Y,YNEG); }
			else if ( dist < distance[p+Y] ) { MODIFY(p+Y,YNEG); }
		}
	}

	for ( OLDEDGEPOINTS(p) )
	{
		if ( !( edgebits[p] & XPOS ) && (Psi[p]<0) != (Psi[p+X]<0) )
		{
			deltaPsi = Psi[p+X] - Psi[p];
			dist = -Psi[p] / deltaPsi;

			if ( !inband[p] ) { ENQUEUE(p,XPOS); }
			else if ( dist < distance[p] ) { MODIFY(p,XPOS); }

			dist = 1.0 - dist;

			if ( !inband[p+X] ) { ENQUEUE(p+X,XNEG); }
			else if ( dist < distance[p+X] ) { MODIFY(p+X,XNEG); }
		}

		if ( !( edgebits[p] & YPOS ) && (Psi[p]<0) != (Psi[p+Y]<0) )
		{
			deltaPsi = Psi[p+Y] - Psi[p];
			dist = -Psi[p] / deltaPsi;

			if ( !inband[p] ) { ENQUEUE(p,YPOS); }
			else if ( dist < distance[p] ) { MODIFY(p,YPOS); }

			dist = 1.0 - dist;

			if ( !inband[p+Y] ) { ENQUEUE(p+Y,YNEG); }
			else if ( dist < distance[p+Y] ) { MODIFY(p+Y,YNEG); }
		}
	}
	if( bandno == 0 )
    {
	    std::cout<<"no band points found (updateBand)..";
	    while(1);
    }
//    std::cout<<"band points :"<<bandno<<std::endl;

	#undef OLDBANDPOINTS
	#undef OLDEDGEPOINTS
	#undef ENQUEUE
	#undef MODIFY

	//Initialize neighbor point list (overwrite old band list)

	nhbrtail=nhbrband=oldband;

	//Propagate distances to neighbors of band points

	int *extorig= dirbits;  //Grid points from which extensions originate
	int neighbourno = 0;
	#define ENQUEUE(p,q) distance[p] = dist ; inband[p]=2 ; *nhbrtail++ = p; \
						 extorig[p] = q ; ++neighbourno
	#define MODIFY(p,q)  distance[p] = dist ; extorig[p] = q

	for ( BANDPOINTS(p) )
	{
		//Adjacent neighbors
		if( distance[p] > 1 || distance[p] < 0 )
		{
			std::cout<<"impossible value for bandpoint(updateBand)..";
			while(1);
		}

		dist = distance[p] + 1.0;

		if ( !inband[p+X] ) { ENQUEUE(p+X,p); }
		else if ( dist < distance[p+X] ) { MODIFY(p+X,p); }

		if ( !inband[p-X] ) { ENQUEUE(p-X,p); }
		else if ( dist < distance[p-X] ) { MODIFY(p-X,p); }

		if ( !inband[p+Y] ) { ENQUEUE(p+Y,p); }
		else if ( dist < distance[p+Y] ) { MODIFY(p+Y,p); }

		if ( !inband[p-Y] ) { ENQUEUE(p-Y,p); }
		else if ( dist < distance[p-Y] ) { MODIFY(p-Y,p); }

		//Diagonal neighbors
    
		dist = distance[p] + Diag_Dist;

		if ( !inband[p+X+Y] ) { ENQUEUE(p+X+Y,p); } 
		else if ( dist < distance[p+X+Y] ) { MODIFY(p+X+Y,p); }

		if ( !inband[p+X-Y] ) { ENQUEUE(p+X-Y,p); }
		else if ( dist < distance[p+X-Y] ) { MODIFY(p+X-Y,p); }

		if ( !inband[p-X+Y] ) { ENQUEUE(p-X+Y,p); }
		else if ( dist < distance[p-X+Y] ) { MODIFY(p-X+Y,p); }

		if ( !inband[p-X-Y] ) { ENQUEUE(p-X-Y,p); }
		else if ( dist < distance[p-X-Y] ) { MODIFY(p-X-Y,p); }
    } 

	for (EDGEPOINTS(p))
	{
		//Adjacent neighbors
		if( distance[p] > 1 || distance[p] < 0 )
		{
			std::cout<<"impossible value for edgepoint(updateBand)..";
			while(1);
		}
		dist = distance[p] + 1.0;

		if ( !(edgebits[p]&XPOS) )
		{
			if ( !inband[p+X] ) { ENQUEUE(p+X,p); }
			else if ( dist < distance[p+X] ) { MODIFY(p+X,p); }
		}
		if ( !(edgebits[p]&XNEG) )
		{
			if ( !inband[p-X] ) { ENQUEUE(p-X,p); }
			else if ( dist < distance[p-X] ) { MODIFY(p-X,p); }
		}
		if ( !(edgebits[p]&YPOS) )
		{
			if ( !inband[p+Y] ) { ENQUEUE(p+Y,p); }
			else if ( dist < distance[p+Y] ) { MODIFY(p+Y,p); }
		}
		if ( !(edgebits[p]&YNEG) )
		{
			if ( !inband[p-Y] ) { ENQUEUE(p-Y,p); }
			else if ( dist < distance[p-Y] ) { MODIFY(p-Y,p); }
		}

		//Diagonal neighbors
    
		dist = distance[p] + Diag_Dist;

		if ( !( edgebits[p] & (XPOS|YPOS) ) )
		{
			if ( !inband[p+X+Y] ) { ENQUEUE(p+X+Y,p); }
			else if ( dist < distance[p+X+Y] ) { MODIFY(p+X+Y,p); }
		}
		if ( !( edgebits[p] & (XPOS|YNEG) ) )
		{
			if ( !inband[p+X-Y] ) { ENQUEUE(p+X-Y,p); }
			else if ( dist < distance[p+X-Y] ) { MODIFY(p+X-Y,p); }
		}
		if ( !( edgebits[p] & (XNEG|YPOS) ) )
		{
			if ( !inband[p-X+Y] ) { ENQUEUE(p-X+Y,p); }
			else if ( dist < distance[p-X+Y] ) { MODIFY(p-X+Y,p); }
		}
		if ( !( edgebits[p] & (XNEG|YNEG) ) )
		{
			if ( !inband[p-X-Y] ) { ENQUEUE(p-X-Y,p); }
			else if ( dist < distance[p-X-Y] ) { MODIFY(p-X-Y,p); }
		}
	}
	if( neighbourno == 0 )
    {
	    std::cout<<"no neighbour points found (updateBand)..";
	    while(1);
    }
//	std::cout<<"neighbour points: "<<neighbourno<<std::endl;

#undef ENQUEUE
#undef MODIFY
  
  //Set level set function to signed distance at narrowband points 
  //and neighbors
	for ( BANDPOINTS(p) )
	{
		if ( distance[p] < 0 )
		{
			std::cout<<"distance should not be negative(updateBand BAND)..";
			while(1);
		}
		else
		{
			Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p];
		}
	}
	for ( EDGEPOINTS(p) )
	{
		if ( distance[p] < 0 )
		{
			std::cout<<"distance should not be negative(updateBand EDGE)..";
			while(1);
		}
		else
		{
			Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p];
		}
    }
    for ( NHBRPOINTS(p) )
	{
		if ( distance[p] < 0 )
		{
			std::cout<<"distance should not be negative(updateBand NHBR)..";
			while(1);
		}
		else
		{
			Psi[p] = Psi[p] < 0 ? -distance[p] : distance[p];
		}
	}
}