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

#include "JLinearValue.h"
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
JLinearValue::JLinearValue(){
  ClassName="JLinearValue";
  Times=NULL;
  Values=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JLinearValue::~JLinearValue(){
  Reset();
}

//==============================================================================
/// Initialization of variables.
//==============================================================================
void JLinearValue::Reset(){
  SetSize(0);
  File="";
}

//==============================================================================
/// Devuelve la memoria reservada.
/// Returns the allocated memory.
//==============================================================================
unsigned JLinearValue::GetAllocMemory()const{
  unsigned s=0;
  if(Times)s+=sizeof(double)*Size;
  if(Values)s+=sizeof(double)*Size;
  return(s);
}

//==============================================================================
/// Ajusta al tamano indicado manteniendo el contenido.
/// Sets the indicated size to maintain the content.
//==============================================================================
void JLinearValue::SetSize(unsigned size){
  if(size>=SIZEMAX)RunException("SetSize","It has reached the maximum size allowed.");
  Size=size;
  if(Count>Size)Count=Size;
  if(Size){
    Times=fun::ResizeAlloc(Times,Count,size);
    Values=fun::ResizeAlloc(Values,Count,size);
  }
  else{
    delete[] Times;  Times=NULL;
    delete[] Values; Values=NULL;
  }
  Position=0;
}

//==============================================================================
/// Añade valores al final de la lista.
/// Adds values at the end of the list.
//==============================================================================
void JLinearValue::AddTimeValue(double time,double value){
  if(Count==Size)SetSize(Size+SIZEINITIAL);
  Times[Count]=time;
  Values[Count]=value;
  Count++;
}

//==============================================================================
/// ES:
/// Devuelve valor el valor interpolado para el instante indicado.
/// Si no hay valores siempre devuelve 0.
/// Si solo hay un valor siempre devuelve ese valor.
/// Si el t indicado es menor que el minimo devuelve el primer valor.
/// Si el t indicado es mayor que el maximo devuelve el ultimo valor.
/// - EN:
/// Returns the interpolated value value for the time indicated.
/// If no values always returns 0.
/// If only one value always returns that value.
/// If the indicated t is less than the minimum returns the first value.
/// If the indicated t is greater than the maximum returns the last value.
//==============================================================================
double JLinearValue::GetValue(double timestep){
  double ret=0;
  if(Count==1)ret=Values[0];
  else if(Count>1){
    double tini=Times[Position];
    //-Si t de position es mayor que el timestep solicitado reinicia position. 
	//-If the t of the position is greater than the requested time step, restart position.
    if(tini>timestep && Position>0){
      Position=0;
      tini=Times[Position];
    }
    //-Busca intervalo del instante indicado.
	//=Finds indicated interval of time.
    double tnext=(Position+1<Count? Times[Position+1]: tini);
    for(;tnext<timestep&&Position+2<Count;Position++){
      tini=tnext;
      tnext=Times[Position+2];
    }
    //-Calcula valor en el instante indicado.
	//-computes value in the indicated instant.
    if(timestep<=tini)ret=Values[Position];
    else if(timestep>=tnext)ret=Values[Position+1];
    else{
      const double tfactor=(timestep-tini)/(tnext-tini);
      double vini=Values[Position];
      double vnext=Values[Position+1];
      ret=(tfactor*(vnext-vini)+vini);
    }
  }
  return(ret);
}

//==============================================================================
/// Carga valores para diferentes instantes.
/// Loads values for different times.
//==============================================================================
void JLinearValue::LoadFile(std::string file){
  const char met[]="LoadFile";
  Reset();
  ifstream pf;
  pf.open(file.c_str());
  if(pf){
    pf.seekg(0,ios::end);
    unsigned len=(unsigned)pf.tellg();
    pf.seekg(0,ios::beg);
    SetSize(SIZEINITIAL);
    Count=0;
    while(!pf.eof()){
      double time,value;
      pf >> time;
      pf >> value;
      if(!pf.fail()){
        if(Count>=Size){
          unsigned newsize=unsigned(double(len)/double(pf.tellg())*1.05*(Count+1))+100;
          SetSize(newsize);
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




