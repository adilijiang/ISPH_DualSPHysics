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

#ifndef _JCellDivGpuSingle_ker_
#define _JCellDivGpuSingle_ker_

#include "JCellDivGpu_ker.h"

namespace cudiv{

void PreSortFull(unsigned np,unsigned cellcode,const unsigned *dcell,const word *code,tuint3 cellmin,tuint3 ncells,unsigned *cellpart,unsigned *sortpart,JLog2 *log);
void PreSortFluid(unsigned npf,unsigned pini,unsigned cellcode,const unsigned *dcell,const word *code,tuint3 cellmin,tuint3 ncells,unsigned *cellpart,unsigned *sortpart,JLog2 *log);
//#Sorts mirrored positions into cells
void MirrorDCell(unsigned bsbound,unsigned npb,const word *code,const unsigned *idpg,const double3 *mirrorPos,unsigned *mirrorCell,tdouble3 domrealposmin,tdouble3 domrealposmax,tdouble3 domposmin,float scell,int domcellcode);
void MatrixMirrorDCell(const unsigned bsbound,const unsigned bsfluid,const unsigned npf,const unsigned npb,const unsigned npbok,const double2 *posxy,const double *posz
	,const word *code,const unsigned *idpg,unsigned int *row,unsigned int *col,tdouble3 domrealposmin,tdouble3 domrealposmax,tdouble3 domposmin,float scell
	,int domcellcode,const bool PeriActive,const tdouble3 MapRealPosMin,const tdouble3 MapRealSize,const tdouble3 PeriXinc,const tdouble3 PeriYinc,const tdouble3 PeriZinc);

}
#endif



