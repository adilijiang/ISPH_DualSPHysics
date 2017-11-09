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


/// \file JTimer.h \brief Declares the class \ref JTimer.

#ifndef _JTimer_
#define _JTimer_

#ifdef WIN32
//==============================================================================
// Windows version 
//==============================================================================
#include <windows.h>


//##############################################################################
//# JTimer
//##############################################################################
/// \brief Defines a class to measure short time intervals.

class JTimer
{
private:
  bool Started,Stopped;
  LARGE_INTEGER Freq;
  LARGE_INTEGER CounterIni,CounterEnd;

  LARGE_INTEGER GetElapsed(){ 
    LARGE_INTEGER dif; dif.QuadPart=(Stopped? CounterEnd.QuadPart-CounterIni.QuadPart: 0);
    return(dif);
  }

public:
  JTimer(){ QueryPerformanceFrequency(&Freq); Reset(); }
  void Reset(){ Started=Stopped=false; CounterIni.QuadPart=0; CounterEnd.QuadPart=0; }
  void Start(){ Stopped=false; QueryPerformanceCounter(&CounterIni); Started=true; }
  void Stop(){ if(Started){ QueryPerformanceCounter(&CounterEnd); Stopped=true; } }
  //-Returns time in miliseconds.
  float GetElapsedTimeF(){ return((float(GetElapsed().QuadPart)*float(1000))/float(Freq.QuadPart)); }
  double GetElapsedTimeD(){ return((double(GetElapsed().QuadPart)*double(1000))/double(Freq.QuadPart)); }
};

#else
//==============================================================================
// Linux version 
//==============================================================================
//#include "JTimerClock.h"
//#define JTimer JTimerClock

#include <cstdio>
#include <sys/time.h>

//==============================================================================
//##############################################################################
//==============================================================================
/// \brief Defines a class to measure short time intervals.

class JTimer
{
private:
  bool Started,Stopped;
  timeval CounterIni,CounterEnd;

public:
  JTimer(){ Reset(); }
  void Reset(){ Started=Stopped=false; CounterIni.tv_sec=0; CounterIni.tv_usec=0; CounterEnd.tv_sec=0; CounterEnd.tv_usec=0; }
  void Start(){ Stopped=false; gettimeofday(&CounterIni,NULL); Started=true; }
  void Stop(){if(Started){ gettimeofday(&CounterEnd,NULL); Stopped=true; } }
  //-Returns time in miliseconds.
  float GetElapsedTimeF(){ 
    return((CounterEnd.tv_sec-CounterIni.tv_sec)*1000+(float(CounterEnd.tv_usec)/1000.f)-(float(CounterIni.tv_usec)/1000.f));
  }
  double GetElapsedTimeD(){
    return((CounterEnd.tv_sec-CounterIni.tv_sec)*1000+(double(CounterEnd.tv_usec)/1000.0)-(double(CounterIni.tv_usec)/1000.0));
  }
};

#endif

#endif








