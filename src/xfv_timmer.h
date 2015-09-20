#ifndef Seg_by_Complementarity_xfv_timmer_h
#define Seg_by_Complementarity_xfv_timmer_h
#include <chrono>

class Timer{
    
public:
    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }
    void reset(){
        start = std::chrono::high_resolution_clock::now();
    }
    std::chrono::milliseconds elapsed() const{
        return std::chrono::duration_cast< std::chrono::milliseconds > (
                                                                        std::chrono::high_resolution_clock::now() - start );
    }
    
private:
    std::chrono::high_resolution_clock::time_point start;
};


#endif
