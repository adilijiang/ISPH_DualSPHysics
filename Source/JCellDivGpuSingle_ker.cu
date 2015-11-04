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

#include "JCellDivGpuSingle_ker.h"
#include "Types.h"
#include <float.h>
#include "JLog2.h"

namespace cudiv{

//------------------------------------------------------------------------------
/// Carga cellpart[] y sortpart[] para ordenar particulas con radixsort
/// Loads cellpart[] and sortpart [] to sort particles with radixsort
//------------------------------------------------------------------------------
__global__ void KerPreSortFull(unsigned np,unsigned cellcode,const unsigned *dcell,const word *code,uint3 cellzero,uint3 ncells,unsigned *cellpart,unsigned *sortpart)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula //-NI of the particle
  if(p<np){
    sortpart[p]=p;
    const unsigned nsheet=ncells.x*ncells.y;
    const unsigned cellignore=nsheet*ncells.z; //- cellignore==nct
    const unsigned cellfluid=cellignore+1;
    const unsigned cellboundout=cellfluid+cellignore;        //-Para bound //-For boundaries 
    const unsigned cellfluidout=cellboundout+1;              //-Para fluid and floatings //-For fluid and floatings
    const unsigned cellboundoutignore=cellfluidout+1;        //-Para bound //-For boundaries 
    const unsigned cellfluidoutignore=cellboundoutignore+1;  //-Para fluid and floatings //-For fluid and floatings
    unsigned rcell=dcell[p];
    unsigned cx=PC__Cellx(cellcode,rcell)-cellzero.x;
    unsigned cy=PC__Celly(cellcode,rcell)-cellzero.y;
    unsigned cz=PC__Cellz(cellcode,rcell)-cellzero.z;
    const unsigned cellsort=cx+cy*ncells.x+cz*nsheet;
    const word rcode=code[p];
    const bool xbound=(CODE_GetType(rcode)<CODE_TYPE_FLOATING);
    const word codeout=CODE_GetSpecialValue(rcode);
    if(xbound){//-Particulas bound no floating. //-Boundary particles but not floating
      cellpart[p]=(codeout<CODE_OUTIGNORE? ((cx<ncells.x && cy<ncells.y && cz<ncells.z)? cellsort: cellignore): (codeout==CODE_OUTIGNORE? cellboundoutignore: cellboundout));
    }
    else{//-Particulas fluid and floating. //Fluid and floating particles
      cellpart[p]=(codeout<CODE_OUTIGNORE? cellfluid+cellsort: (codeout==CODE_OUTIGNORE? cellfluidoutignore: cellfluidout));
    }
  }
}

//==============================================================================
/// ES:
/// Procesa particulas bound y fluid que pueden estar mezcladas.
/// Calcula celda de cada particula (CellPart[]) a partir de su celda en mapa,
/// todas las particulas excluidas ya fueron marcadas en code[].
/// Asigna valores consecutivos a SortPart[].
/// - EN:
/// Processes bound and fluid particles that may be mixed.
/// Computes cell of each particle (CellPart[]) from his cell in the map,
/// all excluded particles were already marked in code[].
/// Assigns consecutive values to SortPart[].
//==============================================================================
void PreSortFull(unsigned np,unsigned cellcode,const unsigned *dcell,const word *code,tuint3 cellmin,tuint3 ncells,unsigned *cellpart,unsigned *sortpart,JLog2 *log){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerPreSortFull <<<sgrid,DIVBSIZE>>> (np,cellcode,dcell,code,make_uint3(cellmin.x,cellmin.y,cellmin.z),make_uint3(ncells.x,ncells.y,ncells.z),cellpart,sortpart);
  }
}

//------------------------------------------------------------------------------
/// Carga cellpart[] y sortpart[] para ordenar particulas de fluido con radixsort
/// Loads cellpart[] and sortpart[] to reorder fluid particles with radixsort
//------------------------------------------------------------------------------
__global__ void KerPreSortFluid(unsigned n,unsigned pini,unsigned cellcode,const unsigned *dcell,const word *code,uint3 cellzero,uint3 ncells,unsigned *cellpart,unsigned *sortpart)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula //-NI of the particle
  if(p<n){
    p+=pini;
    sortpart[p]=p;
    const unsigned nsheet=ncells.x*ncells.y;
    const unsigned cellfluid=nsheet*ncells.z+1;
    const unsigned cellfluidout=cellfluid+cellfluid;   //-Para fluid and floatings //-For fluid and floatings
    const unsigned cellfluidoutignore=cellfluidout+2;  //-Para fluid and floatings //-For fluid and floatings
    unsigned rcell=dcell[p];
    unsigned cx=PC__Cellx(cellcode,rcell)-cellzero.x;
    unsigned cy=PC__Celly(cellcode,rcell)-cellzero.y;
    unsigned cz=PC__Cellz(cellcode,rcell)-cellzero.z;
    const unsigned cellsort=cellfluid+cx+cy*ncells.x+cz*nsheet;
    const word codeout=CODE_GetSpecialValue(code[p]);
    //-Particulas fluid and floatings.
	//-Fluid and floating particles.
    cellpart[p]=(codeout<CODE_OUTIGNORE? cellsort: (codeout==CODE_OUTIGNORE? cellfluidoutignore: cellfluidout));
  }
}

//==============================================================================
/// ES:
/// Procesa solo particulas fluid.
/// Calcula celda de cada particula (CellPart[]) a partir de su celda en mapa,
/// todas las particulas excluidas ya fueron marcadas en code[].
/// Asigna valores consecutivos a SortPart[].
/// - EN:
/// Processes only fluid particles.
/// Computes cell of each particle (CellPart[]) from his cell in the map,
/// all excluded particles were already marked in code[].
/// Assigns consecutive values to SortPart[].
//==============================================================================
void PreSortFluid(unsigned npf,unsigned pini,unsigned cellcode,const unsigned *dcell,const word *code,tuint3 cellmin,tuint3 ncells,unsigned *cellpart,unsigned *sortpart,JLog2 *log){
  if(npf){
    dim3 sgrid=GetGridSize(npf,DIVBSIZE);
    KerPreSortFluid <<<sgrid,DIVBSIZE>>> (npf,pini,cellcode,dcell,code,make_uint3(cellmin.x,cellmin.y,cellmin.z),make_uint3(ncells.x,ncells.y,ncells.z),cellpart,sortpart);
  }
}


}



