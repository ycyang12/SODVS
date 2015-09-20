#ifndef MINHEAP_H
#define MINHEAP_H


class heap_element {
 public:
  int   pixel;
  float value;
  
  heap_element() {
    value=0;
	pixel=0;
  }

  heap_element(int p, float val) {
    value=val; pixel=p;
  }
  
  int operator < (heap_element rhs) {return value < rhs.value;}
  int operator > (heap_element rhs) {return value > rhs.value;} 
  void set(int pixel, float value) { 
    this->pixel=pixel;
    this->value=value;
  }
};


class minheap {
 public:
  int MAX_SIZE;
  int BACKPTR_SIZE;
  int Size;
  heap_element *heap;
  int *backptr;

 public:
  minheap() {
    MAX_SIZE=BACKPTR_SIZE=Size=0;
    heap=0;
    backptr=0;
  }

  int allocate(int MAX_SIZE, int BACKPTR_SIZE) {
    this->MAX_SIZE=MAX_SIZE;
    this->BACKPTR_SIZE=BACKPTR_SIZE;
    Size=0;
    if (!(heap=new heap_element[MAX_SIZE]) ||
	!(backptr=new int[BACKPTR_SIZE]) ) {
      deallocate();
      return 0;
    }
    return 1;
  }
  
  void deallocate() {
    delete[] heap; heap=0;
    delete[] backptr; backptr=0;
    BACKPTR_SIZE=MAX_SIZE=Size=0;
    return;
  }
  
  void swap(int i, int j) {
    heap_element tmph=heap[i];
    int tmpbp=backptr[heap[i].pixel];

    backptr[heap[i].pixel]=backptr[heap[j].pixel]; 
    backptr[heap[j].pixel]=tmpbp;
    heap[i]=heap[j]; heap[j]=tmph;
    return;
  }

  int push(int p, float value) {
    heap_element toAdd(p, value);
    return push(toAdd);
  }

  int push(heap_element toAdd) {
    Size++;
    if (Size>MAX_SIZE) {
      Size--;
      return 0;
    }
    heap[Size-1]=toAdd; backptr[toAdd.pixel]=Size-1;
    upheap(Size-1);
    
    return 1;
  }
  
  void upheap(int i) {
    int parenti;

    parenti=i==0 ? -1 : (i-1)/2;
    while (parenti>=0) {
      if (heap[i]<heap[parenti]) swap(i,parenti);
      else break;
      i=parenti; parenti=i==0 ? -1 : (i-1)/2;
    }
    return;
  }

  void downheap(int i) {
    int child1, child2, min_child;

    child1=2*i+1; child2=child1+1;
    while (child1<=Size-1) {
      min_child=child2>Size-1 ? child1 : 
	(heap[child1]<heap[child2] ? child1 : child2);
      if (heap[i]>heap[min_child]) swap(i,min_child);
      else break;
      i=min_child; child1=2*i+1; child2=child1+1;
    }
    return;
  }

  int pop() {
    int toreturn;
    
    if (Size==0) return -1;
    toreturn=heap[0].pixel;
    heap[0]=heap[--Size]; backptr[heap[0].pixel]=0;
    downheap(0);

    return toreturn;
  }
  
  void order(int p, float value) {
    heap[backptr[p]].value=value;
    upheap(backptr[p]);
    return;
  }
  
  //friend ostream& operator <<(ostream &os, const minheap& rhs);
};

/*
ostream& operator << (ostream& os, const minheap& rhs) {
  os << "heap:" << endl;
  for (int i=0; i<rhs.Size; i++)
    os << rhs.heap[i].value << "(" << rhs.heap[i].pixel << ") ";
  os << endl << "backptr:" << endl;
  for (int i=0; i<rhs.BACKPTR_SIZE; i++)
    os << rhs.backptr[i] << " ";  
  return os<<endl;
}
*/

#endif
