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

/// \file JSphDtFixed.cpp \brief Implements the class \ref JSphDtFixed.

#include "JSphDtFixed.h"
#include "Functions.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cfloat>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSphDtFixed::JSphDtFixed(){
  ClassName="JSphDtFixed";
  Times=NULL;
  Values=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphDtFixed::~JSphDtFixed(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphDtFixed::Reset(){
  delete[] Times;  Times=NULL;
  delete[] Values; Values=NULL;
  File="";
  Size=Count=Position=0;
  GetDtError(true);
}

//==============================================================================
/// Resizes memory space for values.
//==============================================================================
void JSphDtFixed::Resize(unsigned size){
  if(size>SIZEMAX)size=SIZEMAX;
  if(size==Size)RunException("Resize","It has reached the maximum size allowed.");
  Times=fun::ResizeAlloc(Times,Count,size);
  Values=fun::ResizeAlloc(Values,Count,size);
  Size=size;
}

//==============================================================================
/// Returns the allocated memory.
//==============================================================================
unsigned JSphDtFixed::GetAllocMemory()const{
  unsigned s=0;
  if(Times)s+=sizeof(double)*Size;
  if(Values)s+=sizeof(double)*Size;
  return(s);
}

//==============================================================================
/// Loads values of dt (MILISECONDS) for different instants (in SECONDS).
//==============================================================================
void JSphDtFixed::LoadFile(std::string file){
  const char met[]="LoadFile";
  Reset();
  ifstream pf;
  pf.open(file.c_str());
  if(pf){
    pf.seekg(0,ios::end);
    unsigned len=(unsigned)pf.tellg();
    pf.seekg(0,ios::beg);
    Resize(SIZEINITIAL);
    Count=0;
    while(!pf.eof()){
      double time,value;
      pf >> time;
      pf >> value;
      if(!pf.fail()){
        if(Count>=Size){
          unsigned newsize=unsigned(double(len)/double(pf.tellg())*1.05*(Count+1))+100;
          Resize(newsize);
        } 
        Times[Count]=time; Values[Count]=value;
        Count++;
      }
    }
    //if(pf.fail())RunException(met,"Error leyendo datos de fichero.",fname);
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
  if(Count<2)RunException(met,"Cannot be less than two values.",file);
  File=file;
}

//==============================================================================
/// Returns the value of dt (in SECONDS) for a given instant.
//==============================================================================
double JSphDtFixed::GetDt(double timestep,double dtvar){
  double ret=0;
  //-Busca intervalo del instante indicado.
  //-Searches indicated interval of time.
  double tini=Times[Position];
  double tnext=(Position+1<Count? Times[Position+1]: tini);
  for(;tnext<timestep&&Position+2<Count;Position++){
    tini=tnext;
    tnext=Times[Position+2];
  }
  //-Calcula dt en el instante indicado.
  //-Computes dt for the indicated instant.
  if(timestep<=tini)ret=Values[Position]/1000;
  else if(timestep>=tnext)ret=Values[Position+1]/1000;
  else{
    const double tfactor=(timestep-tini)/(tnext-tini);
    double vini=Values[Position];
    double vnext=Values[Position+1];
    ret=(tfactor*(vnext-vini)+vini)/1000;
  }
  double dterror=ret-dtvar;
  if(DtError<dterror)DtError=dterror;
  return(ret);
}

//==============================================================================
/// Returns the maximum error regarding the dtvariable. max(DtFixed-DtVariable).
//==============================================================================
double JSphDtFixed::GetDtError(bool reset){
  double ret=DtError;
  if(reset)DtError=-DBL_MAX;
  return(ret);
}



