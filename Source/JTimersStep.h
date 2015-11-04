/*
 <DUALSPHYSICS>  Copyright (c) 2015, Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

#ifndef _JTimersStep_
#define _JTimersStep_

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Implementacion de clase para almacenar los timers cada cierto tiempo y 
//#   generar csv con estos valores. (10/11/2012)
//# - Permite indicar un instante inicial distinto de cero para cuando se 
//#   continuan simulaciones. (06/09/2013)
//# - EN:
//# Changes:
//# =========
//# - Class implementation to store timers every so often and 
//# generates csv with these values. (10/11/2012)
//# - Let's you specify a different initial time than zero when you
//# continue the simulation. (06.09.2013)
//#############################################################################

#include "JObject.h"
#include "Types.h"
#include "JTimer.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>


//==============================================================================
//##############################################################################
//==============================================================================
class JTimersStepValue
{
protected:
  std::string Name;
  const double *PtrTime;
  double TimeM1;
public:
  JTimersStepValue(){ Reset(); }
  void Reset(){ Name=""; PtrTime=NULL; TimeM1=0; }
  void Config(std::string name,const double *ptrtime){ Name=name; PtrTime=ptrtime; TimeM1=0; }
  float GetInterval(){
    double tnew=(*PtrTime);
    float ti=float(tnew-TimeM1);
    TimeM1=tnew;
    return(ti);
  }
  std::string GetName()const{ return(Name); }
};

//==============================================================================
//##############################################################################
//==============================================================================
class JTimersStep : protected JObject
{
protected:
  typedef struct{
    float timestep;
    double tsimula;
    unsigned steps;
    unsigned np;
    unsigned npbin;
    unsigned npbout;
    unsigned npf;
    unsigned nct;
  }StStepInfo;

  const int MpiRank,MpiCount;
  const float TimeInterval;
  std::string DirOut;
  std::ofstream *Pf;

  unsigned CountTot;
  unsigned StepsM1;
  float TimeStepInit;
  float TimeStepM1;
  float TimeStepNext;

  static const unsigned TIMERSMAX=30;
  unsigned TimersCount;
  JTimersStepValue Timers[TIMERSMAX];

  //-Vars para almacenar informacion.
  //-Variables for stored information.
  static const unsigned BUFFERSIZE=1000;
  float *Values;
  StStepInfo StepInfo[BUFFERSIZE];
  unsigned Count;

public:
  JTimersStep(const std::string &dirout,float tinterval,int mpirank,int mpicount);
  ~JTimersStep();

  void Reset();
  unsigned GetAllocMemory()const;
  void ClearData();
  void AddTimer(std::string name,const double* ptrtime);
  void SetInitialTime(float timestep);

  bool Check(float timestep)const{ return(timestep>=TimeStepNext); }
  void AddStep(float timestep,double tsimula,unsigned steps,unsigned np,unsigned npb,unsigned npbok,unsigned nct);
  void SaveData();
  float GetTimeInterval()const{ return(TimeInterval); };
};

#endif


