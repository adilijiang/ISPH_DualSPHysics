/*
 <DUALSPHYSICS>  Copyright (c) 2016, 
 Dr Jose M. Dominguez, Dr Alejandro Crespo,Prof. Moncho Gomez Gesteira, Dr Anxo Barreiro,
 Dr Benedict Rogers, Dr Georgios Fourtakas, Dr Athanasios Mokos,
 Dr Renato Vacondio, Dr Ricardo Canelas, Dr Stephen Longshaw, Dr Corrado Altomare.

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/** \mainpage DualSPHysics Documentation
\section main-des Description
DualSPHysics is based on the Smoothed Particle Hydrodynamics <a href="http://www.sphysics.org">SPHysics code.</a> \n
The package is a set of C++ and CUDA codes. \n
DualSPHysics is developed to deal with real-life engineering problems <a href="http://www.youtube.com/user/DualSPHysics">DualSPHysics animations.</a> \n

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain. \n
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.
\section compile_sec Project files
Please download source files and documentation from <a href="http://dual.sphysics.org">DualSPHysics website.</a> \n
\author <a href="http://dual.sphysics.org/index.php/developers">DualSPHysics Developers.</a> 
\version 4.00
\date 18-05-2016
\copyright GNU Public License <a href="http://www.gnu.org/licenses/">GNU licenses.</a>
*/

/// \file main.cpp \brief Main file of the project that executes the code on CPU or GPU.

#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include "JLog2.h"
#include "JCfgRun.h"
#include "JException.h"
#include "JSphCpuSingle.h"
#ifdef _WITHGPU
  #include "JSphGpuSingle.h"
#endif

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

using namespace std;

//==============================================================================
/// GPL License.
//==============================================================================
std::string getlicense_gpl(const std::string &name){
  std::string tx="";
  tx=tx+"\n\n <"+fun::StrUpper(name)+"> Copyright (C) 2016 by"; 
  tx=tx+"\n Dr Jose M. Dominguez, Dr Alejandro Crespo,";
  tx=tx+"\n Prof. Moncho Gomez Gesteira, Dr Anxo Barreiro,";
  tx=tx+"\n Dr Benedict Rogers, Dr Georgios Fourtakas, Dr Athanasios Mokos,";
  tx=tx+"\n Dr Renato Vacondio, Dr Ricardo Canelas,";
  tx=tx+"\n Dr Stephen Longshaw, Dr Corrado Altomare.\n";
  tx=tx+"\n EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo";
  tx=tx+"\n School of Mechanical, Aerospace and Civil Engineering, University of Manchester\n";
  tx=tx+"\n DualSPHysics is free software: you can redistribute it and/or"; 
  tx=tx+"\n modify it under the terms of the GNU General Public License as";
  tx=tx+"\n published by the Free Software Foundation, either version 3 of"; 
  tx=tx+"\n the License, or (at your option) any later version.\n"; 
  tx=tx+"\n DualSPHysics is distributed in the hope that it will be useful,"; 
  tx=tx+"\n but WITHOUT ANY WARRANTY; without even the implied warranty of"; 
  tx=tx+"\n MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the"; 
  tx=tx+"\n GNU General Public License for more details.\n";
  tx=tx+"\n You should have received a copy of the GNU General Public License,"; 
  tx=tx+"\n along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.\n\n";
  return(tx);
}

//==============================================================================
//==============================================================================
int main(int argc, char** argv){
  int errcode=1;
  std::string progname="DualSPHysics4";
  std::string proginfo;
  std::string license=getlicense_gpl(progname);
  printf("%s",license.c_str());
  char appname[256],appnamesub[256];
  sprintf(appname,"%s v4.0.056 (18-05-2016)%s",progname.c_str(),proginfo.c_str());
  for(unsigned c=0;c<=strlen(appname);c++)appnamesub[c]='='; appnamesub[strlen(appname)+1]='\0';
  printf("\n%s\n%s\n",appname,appnamesub);

  JCfgRun cfg;
  JLog2 log;
  try{
    cfg.LoadArgv(argc,argv);
    //cfg.VisuConfig();
    if(!cfg.PrintInfo){
      log.Init(cfg.DirOut+"/Run.out");
      log.Print(license,JLog2::Out_File);
      log.Print(appname,JLog2::Out_File);
      log.Print(appnamesub,JLog2::Out_File);
      //- SPH Execution
      #ifndef _WITHGPU
        cfg.Cpu=true;
      #endif
      if(cfg.Cpu){
        JSphCpuSingle sph;
        sph.Run(appname,&cfg,&log);
      }
      #ifdef _WITHGPU
      else{
        JSphGpuSingle sph;
        sph.Run(appname,&cfg,&log);
      }
      #endif
    }
    errcode=0;
  }
  catch(const char *cad){
    string tx=string("\n*** Exception: ")+cad+"\n";
    if(log.IsOk())log.Print(tx); else printf("%s",tx.c_str());
  }
  catch(const string &e){
    string tx=string("\n*** Exception: ")+e+"\n";
    if(log.IsOk())log.Print(tx); else printf("%s",tx.c_str());
  }
  catch (const exception &e){
    string tx=string("\n*** ")+e.what()+"\n";
    if(log.IsOk())log.Print(tx); else printf("%s",tx.c_str());
  }
  catch(...){
    printf("\n*** Attention: Unknown exception...\n");
  }
  return(errcode);
}



