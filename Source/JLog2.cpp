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

/// \file JLog2.cpp \brief Implements the class \ref JLog2.

#include "JLog2.h"
#include "Functions.h"
#include <stdarg.h>

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

using std::string;
using std::ofstream;
using std::endl;

//==============================================================================
/// Constructor.
//==============================================================================
JLog2::JLog2(TpMode_Out modeoutdef):ModeOutDef(modeoutdef){
  ClassName="JLog2";
  Pf=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JLog2::~JLog2(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JLog2::Reset(){
  FileName="";
  Ok=false;
  MpiRun=false;
  MpiRank=0; MpiLaunch=0;
  if(Pf){
    if(Pf->is_open())Pf->close();
    delete Pf; Pf=NULL;
  }
}

//==============================================================================
/// Initialisation of log file.
//==============================================================================
void JLog2::Init(std::string fname,bool mpirun,int mpirank,int mpilaunch){
  Reset();
  MpiRun=mpirun; MpiRank=mpirank; MpiLaunch=mpilaunch;
  if(MpiRun){
    string ext=fun::GetExtension(fname);
    fname=fun::GetWithoutExtension(fname);
    fname=fname+"_"+fun::IntStrFill(mpirank,MpiLaunch-1);
    if(!ext.empty())fname=fname+"."+ext;
  }
  FileName=fname;
  if(ModeOutDef&Out_File){
    Pf=new ofstream; 
    Pf->open(fname.c_str());
    if(Pf)Ok=true;
    else RunException("Init","Cannot open the file.",fname);
  }
}

//==============================================================================
/// Returns output directory.
//==============================================================================
std::string JLog2::GetDirOut()const{
  return(fun::GetDirWithSlash(fun::GetDirParent(FileName)));
}
  
//==============================================================================
/// Visualises and/or stores information of the execution.
//==============================================================================
void JLog2::Print(const std::string &tx,TpMode_Out mode,bool flush){
  if(mode==Out_Default)mode=ModeOutDef;
  if(mode&Out_Screen){
    if(MpiRun){
      int pos=0;
      for(;pos<int(tx.length())&&tx[pos]=='\n';pos++)printf("%d>\n",MpiRank);
      printf("%d>%s\n",MpiRank,tx.substr(pos).c_str());
    }
    else printf("%s\n",tx.c_str());
  }
  if((mode&Out_File) && Pf)(*Pf) << tx << endl;
  if(flush)fflush(stdout);
}
  
//==============================================================================
/// Visualises and/or stores information of the execution.
//==============================================================================
void JLog2::Printf(const char *format,...){
  const unsigned SIZE=1024;
  char buffer[SIZE+1];
  va_list args;
  va_start(args, format);
  unsigned size=vsnprintf(buffer,SIZE,format,args);
  if(size<SIZE)Print(buffer);
  else{
    char *buff=new char[size+1];
    vsnprintf(buff,size,format,args);
    Print(buff);
    delete[] buff;
  }
  va_end(args);
}
  
//==============================================================================
/// Visualises and/or stores information of the execution.
//==============================================================================
void JLog2::PrintfDbg(const char *format,...){
  const unsigned SIZE=1024;
  char buffer[SIZE+1];
  va_list args;
  va_start(args, format);
  unsigned size=vsnprintf(buffer,SIZE,format,args);
  if(size<SIZE)Print(buffer,Out_ScrFile,true);
  else{
    char *buff=new char[size+1];
    vsnprintf(buff,size,format,args);
    Print(buff,Out_ScrFile,true);
    delete[] buff;
  }
  va_end(args);
}


