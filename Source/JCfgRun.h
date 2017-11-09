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

/// \file JCfgRun.h \brief Declares the class \ref JCfgRun.

#ifndef _JCfgRun_
#define _JCfgRun_

#include "Types.h"
#include "Functions.h"
#include "JObject.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

//##############################################################################
//# JCfgRun
//##############################################################################
/// \brief Defines the class responsible for collecting the execution parameters by command line.

class JCfgRun : protected JObject
{
public:
protected:
  bool SvDef;
  int DirsDef;
  static void LoadDouble3(std::string txopt,double def,tdouble3 &v1);
  static void LoadFloat3(std::string txopt,float def,tfloat3 &v1);
  static void LoadDouble6(std::string txopt,double def,tdouble3 &v1,tdouble3 &v2);
  static void LoadFloat6(std::string txopt,float def,tfloat3 &v1,tfloat3 &v2);

public:
  bool PrintInfo;
  bool Cpu;
  bool Gpu;
  int GpuId;
  bool GpuFree;
  bool Stable;
  int PosDouble;  ///<Precision in particle interaction. 0:Simple, 1:Double, 2:Uses and save double (default=0).

  int OmpThreads;
  TpBlockSizeMode BlockSizeMode;

  TpCellOrder CellOrder;
  TpCellMode  CellMode;
  TpStep TStep;
  int VerletSteps;
  TpKernel TKernel;
  TpVisco TVisco;
  float Visco;
  float ViscoBoundFactor;
  double TimeMax,TimePart;
  float DeltaSph;
  int Shifting; //-Shifting mode -1:sin definir, 0:none, 1:nobound, 2:nofixed, 3:full
  bool SvRes,SvTimers,SvDomainVtk;
  bool Sv_Binx,Sv_Info,Sv_Csv,Sv_Vtk;
  std::string CaseName,RunName,DirOut;
  std::string PartBeginDir;
  unsigned PartBegin,PartBeginFirst;
  float FtPause;
  bool RhopOutModif;              ///<Indicates whether \ref RhopOutMin or RhopOutMax is changed.
  float RhopOutMin,RhopOutMax;    ///<Limits for \ref RhopOut density correction.

  byte DomainMode; //0:No configured, 1:Particles, 2:Fixed
  tdouble3 DomainParticlesMin,DomainParticlesMax;
  tdouble3 DomainParticlesPrcMin,DomainParticlesPrcMax;
  tdouble3 DomainFixedMin,DomainFixedMax;



  JCfgRun();
  void Reset();
  void VisuInfo()const;
  void VisuConfig()const;
  void LoadArgv(int argc,char** argv);
  void LoadFile(std::string fname,int lv);
  void LoadOpts(std::string *optlis,int optn,int lv,std::string file);
  void ErrorParm(const std::string &opt,int optc,int lv,const std::string &file)const;
};

#endif


