/*
 <DUALSPHYSICS>  Copyright (c) 2016, Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/


/// \file JTimerCuda.h \brief Declares the class \ref JTimerCuda.

#ifndef _JTimerCuda_
#define _JTimerCuda_

#include <cuda_runtime_api.h>

//##############################################################################
//# JTimerCuda
//##############################################################################
/// \brief Defines a class to measure short time intervals on the GPU using cudaEvent.

class JTimerCuda
{
private:
  bool Stopped;
  cudaEvent_t EventIni,EventEnd;

public:
  JTimerCuda(){ EventIni=NULL; EventEnd=NULL; Stopped=false; }
  ~JTimerCuda(){ Reset(); }
  void Reset(){
    if(EventIni)cudaEventDestroy(EventIni); EventIni=NULL;
    if(EventEnd)cudaEventDestroy(EventEnd); EventEnd=NULL;
    Stopped=false;
  }
  void Start(){
    if(!EventIni)cudaEventCreate(&EventIni);
    cudaEventRecord(EventIni,0);
  }
  void Stop(){
    if(!EventEnd)cudaEventCreate(&EventEnd);
    cudaEventRecord(EventEnd,0); 
    cudaEventSynchronize(EventEnd);
    Stopped=true;
  }
  //-Returns time in miliseconds.
  float GetElapsedTimeF(){ 
    float elapsed=0; if(Stopped&&EventIni&&EventEnd)cudaEventElapsedTime(&elapsed,EventIni,EventEnd);
    return(elapsed); 
  }
  double GetElapsedTimeD(){ return(GetElapsedTimeF()); }
};

#endif







