#include "distancemap.h"


int  distancemap::allocate( int dim, int *size, float *P )
{
	if ( dim < 2 ) return 0;

	this->P=P;
	this->dim=dim;
	for (int i=0; i<dim; i++) this->size[i]=size[i];

	increments[0]=1;
	for (int i=1; i<dim; i++) increments[i]=increments[i-1]*size[i-1];
	GridSize=increments[dim-1]*size[dim-1];

	if( !(u      =new float[GridSize])         ||
		!(label  =new unsigned char[GridSize]) ||
		!(minpath=new int[GridSize])           ||
		!triallist.allocate(GridSize,GridSize) )
	{
		deallocate();
		return 0;
	}

	return 1;
}


void distancemap::deallocate()
{
	delete[] u;			u=0;
	delete[] label;		label=0;
	delete[] minpath;	minpath=0;
	P=0;
	triallist.deallocate();
}


void distancemap::findDistanceMap( double *psi, int *Loc, int &Length, int *back, bool *indis )
{
	int p, pneighbor;
	heap_element toadd;
	float U;
	triallist.deallocate();
	if(!triallist.allocate(GridSize,GridSize)){std::cout<<"can't allocate for triallist";while(1);}
	//add seed points to triallist
	for (p=0; p<GridSize; ++p)
	{
		if (back[p]==-1){ u[p]=0; triallist.push(p,0); label[p]=TRIAL; back[p]=p; }
		else            { u[p]=FLT_MAX; label[p]=FA; }
		indis[p] = false;
	}

	Length = 0;
	int incre[4] = {-1,1,-size[0],size[0]};

	while ((p=triallist.pop())!=-1)
	{
		getcoords(p, coords);
		if( u[p]> MAXDIST ) break;
		if( u[p] > 0 && psi[p]>=0 )
		{
			Loc[Length++] = p;
			indis[p]= true;
			float mindist = FLT_MAX;
			int k=-1;;
			if( p%size[0] > 0 && label[p+incre[0]]==ALIVE){
				if( mindist > u[p+incre[0]] ){ 
					mindist = u[p+incre[0]]; k = 0;
				}
			}
			if( p%size[0] < size[0]-1 && label[p+incre[1]]==ALIVE){
				if( mindist > u[p+incre[1]] ){ 
					mindist = u[p+incre[1]]; k = 1;
				}
			}
			if( (p/size[0]) > 0 && label[p+incre[2]]==ALIVE){
				if( mindist > u[p+incre[2]] ){
					mindist = u[p+incre[2]]; k = 2;
				}
			}
			if( (p/size[0]) < size[1]-1 && label[p+incre[3]]==ALIVE){
				if( mindist > u[p+incre[3]] ){
					mindist = u[p+incre[3]]; k = 3;
				}
			}
			if(k==-1){std::cout<<"Impossible case...(distancemap)";while(1);}
			back[p] = back[p+incre[k]];
		}
		label[p]=ALIVE;
		for (int d=0; d<dim; d++)
		{
			for (int sgn=-1; sgn<=1; sgn+=2)
			{
				if (inGrid(d,coords[d]+sgn) && label[pneighbor=p+sgn*increments[d]]!=ALIVE)
				{
					U=computeDistance(pneighbor);
					if (label[pneighbor]==FA)
					{
						u[pneighbor]=U;
						triallist.push(pneighbor, U);
						label[pneighbor]=TRIAL;
					}
					else
					{
						if (U<u[pneighbor]) triallist.order(pneighbor,u[pneighbor]=U);
					}
				}
			}
		}
	}

	return;
}


float distancemap::computeDistance(int p)
{
	static int coords[MAXDIM];
	static float a[MAXDIM];
	float up, um, umin;
	int len=0, n;

#define add(toAdd) {						\
	int i=0, j;                             \
											\
	if (len==0) {a[len++]=toAdd;}           \
	else {                                  \
	while (i<len && toAdd<a[i]) i++;        \
	for (j=len-1; j>=i; j--) a[j+1]=a[j];   \
	a[i]=toAdd;                             \
	len++;                                  \
	}                                       \
	}

	getcoords(p, coords);
	for (int d=0; d<dim; d++) {
		up=um=FLT_MAX;
		if (inGrid(d,coords[d]+1)) up=u[p+increments[d]];
		if (inGrid(d,coords[d]-1)) um=u[p-increments[d]];
		umin=up<um ? up : um;
		if (umin<FLT_MAX) add(umin);
	}

	float sum, sumsq, avg, avgsumsq, sqavg, tmp, discrim;

	sum=sumsq=0;
	for (int i=0; i<len; i++) {sum+=a[i]; sumsq+=a[i]*a[i];}
	for (int d=0; d<len; d++) {
		n=len-d;
		avg=sum/n; sqavg=avg*avg; avgsumsq=sumsq/n;
		tmp=a[d]-avg; tmp*=tmp;
		discrim=sqavg-avgsumsq+P[p]*P[p]/n;
		if (discrim>=tmp)  return avg+sqrt(discrim);
		else if (d==len-1) return avg+P[p];
		sum-=a[d]; sumsq-=a[d]*a[d];
	}

#undef add

	return -1;  // should never happen
}


int distancemap::findMinPath(int p)
{
	int q, qn, minpixel, *tailpath;
	float minvalue;

	if (p<0 || p>=GridSize) return 0;
	q=p; size_minpath=0;
	tailpath=minpath;
	while (1) {
		*tailpath++=q; size_minpath++;
		minvalue=FLT_MAX;
		getcoords(q,coords);
		for (int d=0; d<dim; d++) {
			for (int sgn=-1; sgn<=1; sgn+=2)  {
				if (inGrid(d,coords[d]+sgn) && minvalue>u[qn=q+sgn*increments[d]]) {
					minvalue=u[qn]; minpixel=qn;
				}
			}
		}
		if (u[minpixel]<u[q]) q=minpixel;
		else break;
	}

	return 1;
}


int distancemap::findMinPath2(int p)
{
	int q, qn, minpixel, *tailpath, nbrnumb=(int)pow(3.0,dim), nbrcoords[MAXDIM];
	float minvalue;

	if (p<0 || p>=GridSize) return 0;
	q=p; size_minpath=0;
	tailpath=minpath;
	while (1) {
		*tailpath++=q; size_minpath++;
		minvalue=FLT_MAX;
		getcoords(q,coords);
		for (int n=0; n<nbrnumb; n++) {
			int m=n;

			for (int d=dim-1; d>=0; d--) {
				int incr=(int)pow(3.0,d);
				nbrcoords[d]=m/incr-1; m%=incr;
			}
			bool ingrid=1;
			for (int d=0; d<dim; d++) 
				ingrid=ingrid && inGrid(d,coords[d]+nbrcoords[d]);
			if (ingrid) {
				qn=q;
				for (int d=0; d<dim; d++)
					qn+=increments[d]*nbrcoords[d];
				if (qn!=q && minvalue>u[qn]) {minvalue=u[qn]; minpixel=qn;}
			}
		}
		if (u[minpixel]<u[q]) q=minpixel;
		else break;
	}

	return 1;
}