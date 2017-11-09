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

#ifndef _JCellDivGpuSingle_
#define _JCellDivGpuSingle_

//#############################################################################
//# Cambios:
//# =========
//# Changes:
//# =========
//#############################################################################

#include "JCellDivGpu.h"

class JCellDivGpuSingle : public JCellDivGpu
{
protected:
  void CalcCellDomain(const unsigned *dcellg,const word* codeg);
  void MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const;
  void PrepareNct();

  void PreSort(const unsigned *dcellg,const word *codeg);

public:
  JCellDivGpuSingle(bool stable,bool floating,byte periactive,TpCellOrder cellorder,TpCellMode cellmode,float scell,tdouble3 mapposmin,tdouble3 mapposmax,tuint3 mapcells,unsigned casenbound,unsigned casenfixed,unsigned casenpb,JLog2 *log,std::string dirout);

  void Divide(unsigned npb1,unsigned npf1,unsigned npb2,unsigned npf2,bool boundchanged,const unsigned *dcellg,const word* codeg,TimersGpu timers,const double2 *posxy,const double *posz,const unsigned *idp);
	void MirrorDCellSingle(unsigned bsbound,unsigned npb,const word *code,const unsigned *idpg,const double3 *mirrorPos,unsigned *mirrorCell,tdouble3 domrealposmin,tdouble3 domrealposmax,tdouble3 domposmin,float scell,int domcellcode);
  void MatrixMirrorDCellSingle(const unsigned bsbound,const unsigned bsfluid,const unsigned npf,const unsigned npb,const unsigned npbok,const double2 *posxy,const double *posz
	,const word *code,const unsigned *idpg,unsigned int *row,unsigned int *col,tdouble3 domrealposmin,tdouble3 domrealposmax,tdouble3 domposmin,float scell
	,int domcellcode,const bool PeriActive,const tdouble3 MapRealPosMin,const tdouble3 MapRealSize,const tdouble3 PeriXinc,const tdouble3 PeriYinc,const tdouble3 PeriZinc);
	ullong GetAllocMemoryCpu()const{ return(JCellDivGpu::GetAllocMemoryCpu()); }
  ullong GetAllocMemoryGpu()const{ return(JCellDivGpu::GetAllocMemoryGpu()); }
  ullong GetAllocMemoryGpuNp()const{ return(JCellDivGpu::GetAllocMemoryGpuNp()); };
  ullong GetAllocMemoryGpuNct()const{ return(JCellDivGpu::GetAllocMemoryGpuNct()); };
};

#endif




