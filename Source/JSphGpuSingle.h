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

#ifndef _JSphGpuSingle_
#define _JSphGpuSingle_

#include "Types.h"
#include "JSphGpu.h"
#include <string>

class JCellDivGpuSingle;
class JPartsLoad4;

class JSphGpuSingle : public JSphGpu
{
protected:
  JCellDivGpuSingle* CellDivSingle;
  JPartsLoad4* PartsLoaded;

  llong GetAllocMemoryCpu()const;  
  llong GetAllocMemoryGpu()const;  
  llong GetMemoryGpuNp()const;
  llong GetMemoryGpuNct()const;
  void UpdateMaxValues();
  void LoadConfig(JCfgRun *cfg);
  void LoadCaseParticles();
  void ConfigDomain();

  void ResizeParticlesSize(unsigned newsize,float oversize,bool updatedivide);
  void RunPeriodic();
  void RunCellDivide(bool updateperiodic);

  void RunRenCorrection();
  void Interaction_Forces(TpInter tinter);
  double ComputeAceMax(float *auxmem);

  double ComputeStep(){ return(TStep==STEP_Verlet? ComputeStep_Ver(): ComputeStep_Sym()); }

  double ComputeStep_Ver();
  double ComputeStep_Sym();

  void RunFloating(double dt,bool predictor);

  void SaveData();
  void FinishRun(bool stop);

public:
  JSphGpuSingle();
  ~JSphGpuSingle();
  void Run(std::string appname,JCfgRun *cfg,JLog2 *log);
};

#endif


