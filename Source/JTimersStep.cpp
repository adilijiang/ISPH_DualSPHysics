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

#include "JTimersStep.h"
#include "Functions.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JTimersStep::JTimersStep(const std::string &dirout,float tinterval,int mpirank,int mpicount):TimeInterval(tinterval),MpiRank(mpirank),MpiCount(mpicount){
  ClassName="JTimersStep";
  DirOut=fun::GetDirWithSlash(dirout);
  Pf=NULL;
  Values=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JTimersStep::~JTimersStep(){
  if(Pf){
    SaveData();
    if(Pf->is_open())Pf->close();
    delete Pf; Pf=NULL;
  }
}

//==============================================================================
/// Initialization of variables.
//==============================================================================
void JTimersStep::Reset(){
  ClearData();
  TimeStepInit=0;
  TimersCount=0;
}

//==============================================================================
/// Devuelve la memoria reservada.
/// Returns the allocated memory
//==============================================================================
unsigned JTimersStep::GetAllocMemory()const{
  unsigned s=0;
  s+=sizeof(JTimersStepValue)*TIMERSMAX;
  s+=sizeof(StStepInfo)*BUFFERSIZE;
  if(Values)s+=sizeof(float)*BUFFERSIZE*TimersCount;
  return(s);
}

//==============================================================================
/// Elimina informacion de tiempos.
/// Clears time information.
//==============================================================================
void JTimersStep::ClearData(){
  delete[] Values; Values=NULL;
  CountTot=1;
  Count=0;
  StepsM1=0;
  TimeStepM1=0;
  TimeStepNext=TimeInterval;
}

//==============================================================================
/// Añade nuevo timer.
/// Adds new timer.
//==============================================================================
void JTimersStep::AddTimer(std::string name,const double* ptrtime){
  ClearData();
  if(TimersCount>=TIMERSMAX)RunException("AddTimer","You can not create more timers.");
  Timers[TimersCount].Config(name,ptrtime);
  TimersCount++;
}

//==============================================================================
/// Cambia.
/// Changes.
//==============================================================================
void JTimersStep::SetInitialTime(float timestep){
  TimeStepInit=timestep;
  TimeStepM1=TimeStepInit;
  TimeStepNext=TimeStepInit+TimeInterval;
}

//==============================================================================
/// Anota informacion de paso.
/// Records step information.
//==============================================================================
void JTimersStep::AddStep(float timestep,double tsimula,unsigned steps,unsigned np,unsigned npb,unsigned npbok,unsigned nct){
  if(!Values)Values=new float[TimersCount*BUFFERSIZE];
  if(Count>=BUFFERSIZE)SaveData();
  StStepInfo *sinfo=StepInfo+Count;
  sinfo->timestep=timestep;
  sinfo->tsimula=tsimula;
  sinfo->steps=steps-StepsM1;
  StepsM1=steps;
  sinfo->np=np;
  sinfo->npbin=npbok;
  sinfo->npbout=npb-npbok;
  sinfo->npf=np-npb;
  sinfo->nct=nct;
  float* val=Values+(TimersCount*Count);
  for(unsigned c=0;c<TimersCount;c++)val[c]=Timers[c].GetInterval();
  Count++;
  CountTot++;
  TimeStepM1=timestep;
  TimeStepNext=TimeStepInit+TimeInterval*CountTot;
}

//==============================================================================
/// Graba datos almacenados.
/// Records stored data.
//==============================================================================
void JTimersStep::SaveData(){
  const char* met="SaveData";
  if(!Pf){
    Pf=new ofstream;
    string fname;
    if(MpiCount>1)fname=DirOut+"TimersStep_"+fun::IntStrFill(MpiRank,MpiCount-1)+".csv";    
    else fname=DirOut+"TimersStep.csv"; 
    Pf->open(fname.c_str());
    if(!(*Pf))RunException(met,"Cannot open the file.",fname);
    (*Pf) << "TimeStep;Steps;Np;NpbIn;NpbOut;Npf;Nct;TimeSim";
    for(unsigned ct=0;ct<TimersCount;ct++)(*Pf) << ";" << Timers[ct].GetName();
    (*Pf) << endl;
  }
  if(Pf){
    unsigned cv=0;
    for(unsigned c=0;c<Count;c++){
      (*Pf) << fun::PrintStr("%g;%u;%u;%u;%u;%u;%u;%g",StepInfo[c].timestep,StepInfo[c].steps,StepInfo[c].np,StepInfo[c].npbin,StepInfo[c].npbout,StepInfo[c].npf,StepInfo[c].nct,StepInfo[c].tsimula);
      for(unsigned ct=0;ct<TimersCount;ct++,cv++){
        (*Pf) << fun::PrintStr(";%g",Values[cv]);
      }
      (*Pf) << endl;
    }
  }
  Count=0;
}




