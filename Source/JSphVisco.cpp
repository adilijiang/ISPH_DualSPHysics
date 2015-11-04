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

#include "JSphVisco.h"
#include "Functions.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <float.h>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSphVisco::JSphVisco(){
  ClassName="JSphVisco";
  Times=NULL;
  Values=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphVisco::~JSphVisco(){
  Reset();
}

//==============================================================================
/// Initialization of variables.
//==============================================================================
void JSphVisco::Reset(){
  delete[] Times;  Times=NULL;
  delete[] Values; Values=NULL;
  File="";
  Size=Count=Position=0;
}

//==============================================================================
/// Redimensiona espacio para valores.
/// Resizes allocated space for values.
//==============================================================================
void JSphVisco::Resize(unsigned size){
  if(size>SIZEMAX)size=SIZEMAX;
  if(size==Size)RunException("Resize","It has reached the maximum size allowed.");
  Times=fun::ResizeAlloc(Times,Count,size);
  Values=fun::ResizeAlloc(Values,Count,size);
  Size=size;
}

//==============================================================================
/// Devuelve la memoria reservada.
/// Returns the allocated memory.
//==============================================================================
unsigned JSphVisco::GetAllocMemory()const{
  unsigned s=0;
  if(Times)s+=sizeof(float)*Size;
  if(Values)s+=sizeof(float)*Size;
  return(s);
}

//==============================================================================
/// Carga valores de dt (EN MILISEGUNDOS) para diferentes instantes (en segundos).
/// Loads dt values (in milliseconds) for different instances (in secods).
//==============================================================================
void JSphVisco::LoadFile(std::string file){
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
      float time,value;
      pf >> time;
      pf >> value;
      if(!pf.fail()){
        if(Count>=Size){
          unsigned newsize=unsigned(float(len)/float(pf.tellg())*1.05f*(Count+1))+100;
          Resize(newsize);
        } 
        Times[Count]=time; Values[Count]=value;
        printf("[%u]>  t:%f  v:%f\n",Count,time,value);
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
/// Devuelve el valor de viscosidad para el instante indicado.
/// Returns the viscosity value for the indicated instant.
//==============================================================================
float JSphVisco::GetVisco(float timestep){
  float ret=0;
  //-Busca intervalo del instante indicado.
  //-Searches indicated interval of time.
  float tini=Times[Position];
  float tnext=(Position+1<Count? Times[Position+1]: tini);
  for(;tnext<timestep&&Position+2<Count;Position++){
    tini=tnext;
    tnext=Times[Position+2];
  }
  //-Calcula dt en el instante indicado.
  //-Computes dt for the indicated instant.
  if(timestep<=tini)ret=Values[Position];
  else if(timestep>=tnext)ret=Values[Position+1];
  else{
    const double tfactor=double(timestep-tini)/double(tnext-tini);
    float vini=Values[Position];
    float vnext=Values[Position+1];
    ret=float(tfactor*(vnext-vini)+vini);
  }
  return(ret);
}




