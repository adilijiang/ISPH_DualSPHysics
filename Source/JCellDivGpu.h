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

#ifndef _JCellDivGpu_
#define _JCellDivGpu_

//#############################################################################
//# Cambios:
//# =========
//# Changes:
//# =========
//#############################################################################

#include "Types.h"
#include "JObjectGpu.h"
#include "JSphTimersGpu.h"
#include "JLog2.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>

class JCellDivGpu : protected JObjectGpu
{
protected:
  const bool Stable;
  const bool Floating;
  const byte PeriActive;
  const TpCellOrder CellOrder;
  const TpCellMode CellMode;    ///<ES: Modo de division en celdas. EN: Division mode in cells.
  const unsigned Hdiv;          ///<ES: Valor por el que se divide a DosH EN: Value with ehich to divide 2h
  const float Scell,OvScell;
  const tdouble3 Map_PosMin,Map_PosMax,Map_PosDif;
  const tuint3 Map_Cells;
  const unsigned CaseNbound,CaseNfixed,CaseNpb;
  JLog2 *Log;
  std::string DirOut;

  bool AllocFullNct;	///<ES: Reserva memoria para el numero maximo de celdas del dominio (DomCells). EN: Allocates memory for the maximum number of cells in the doamin (DomCells).
  float OverMemoryNp;	///<ES: Porcentaje que se añade a la reserva de memoria de Np. (def=0). EN: Percentage to be added to the allocated memory for Np (def=0).
  word OverMemoryCells; ///<ES: Numero celdas que se incrementa en cada dimension reservar memoria. (def=0). EN: Number of cells that increase the allocated memoery in each dimension

  //-Vars del dominio definido.
  //-Variables to define the domain.
  unsigned DomCellCode;  ///<ES: Clave para la codificacion de la celda de posicion. EN: Key for the codification of the cell position
  tuint3 DomCelIni,DomCelFin;
  tdouble3 DomPosMin,DomPosMax;
  tuint3 DomCells;

  //-Memoria reservada en funcion de particulas en GPU.
  //-Variables with allocated memory as afunction of the number of particles in GPU
  unsigned SizeNp;
  unsigned *CellPart;
  unsigned *SortPart;
  unsigned SizeAuxMem;
  float *AuxMem;

  //-Memoria reservada en funcion de celdas en GPU.
  //-Variables with allocated memory as a function of the number of cells in GPU 
  unsigned SizeNct;
  int2 *BeginEndCell;  ///<ES: Contiene el principio y final de cada celda.  EN: Contains the first and final particle of each cell
  // BeginEndCell=[BoundOk(nct),BoundIgnore(1),Fluid(nct),BoundOut(1),FluidOut(1),BoundOutIgnore(1),FluidOutIgnore(1)]

  ullong MemAllocGpuNp;  ///<ES: Mermoria reservada en Gpu para particulas. EN: Allocated GPU memory for particles
  ullong MemAllocGpuNct; ///<ES: Mermoria reservada en Gpu para celdas. EN: Allocated GPU memory for cells

  unsigned Ndiv,NdivFull;

  //-Numero de particulas por tipo al iniciar el divide.
  //-Particle number by type at the beginning of the division
  unsigned Npb1;
  unsigned Npf1;
  unsigned Npb2;
  unsigned Npf2;

  unsigned Nptot;  ///<ES: Numero total de particulas incluidas las que se excluyeron al terminar el divide. EN: Number of particles including those excluded at the end of the division
  unsigned NpbOut,NpfOut,NpbOutIgnore,NpfOutIgnore;
  
  unsigned NpFinal,NpbFinal;
  unsigned NpfOutRhop,NpfOutMove,NpbIgnore;

  tuint3 CellDomainMin,CellDomainMax; ///<ES: Limites del dominio en celdas dentro de DomCells. EN: Domain limits in cells within DomCells.
  unsigned Ncx,Ncy,Ncz,Nsheet,Nct;
  ullong Nctt; ///<ES: Numero total de celdas incluyendo las especiales Nctt=SizeBeginEndCell() EN: Number of cells including the special Nctt=SizeBeginEndCell() 
  unsigned BoxIgnore,BoxFluid,BoxBoundOut,BoxFluidOut,BoxBoundOutIgnore,BoxFluidOutIgnore;

  bool BoundLimitOk;  ///<ES: Indica que los limites del contorno ya estan calculados en BoundLimitCellMin y BoundLimitCellMax. EN: Indicates boundary limits are already computed in BoundLimitCellMin and BoundLimitCellMax
  tuint3 BoundLimitCellMin,BoundLimitCellMax;

  bool BoundDivideOk;   ///<ES: Indica que los limites del contorno utilizados en el divide previo fueron BoundDivideCellMin y BoundDivideCellMax. EN: Indicates the boundary limits used in the previous division were computed in BoundDivideCellMin y BoundDivideCellMax
  tuint3 BoundDivideCellMin,BoundDivideCellMax;

  bool DivideFull;  ///<ES: Indica que el divide se aplico a fluido y contorno (no solo al fluido). EN: Indicates that the division applies to fluid and boundary (not only fluid)

  void Reset();

  //-Gestion de reserva dinamica de memoria.
  //-Management of allocated dynamic memory.
  void FreeMemoryNct();
  void FreeMemoryAll();
  void AllocMemoryNp(ullong np);
  void AllocMemoryNct(ullong nct);
  void CheckMemoryNp(unsigned npmin);
  void CheckMemoryNct(unsigned nctmin);

  ullong SizeBeginEndCell(ullong nct)const{ return((nct*2)+5); } //-[BoundOk(nct),BoundIgnore(1),Fluid(nct),BoundOut(1),FluidOut(1),BoundOutIgnore(1),FluidOutIgnore(1)]

  ullong GetAllocMemoryCpu()const{ return(0); }
  ullong GetAllocMemoryGpuNp()const{ return(MemAllocGpuNp); };
  ullong GetAllocMemoryGpuNct()const{ return(MemAllocGpuNct); };
  ullong GetAllocMemoryGpu()const{ return(GetAllocMemoryGpuNp()+GetAllocMemoryGpuNct()); };

  void VisuBoundaryOut(unsigned p,unsigned id,tdouble3 pos,word check)const;
  void CalcCellDomainBound(unsigned n,unsigned pini,unsigned n2,unsigned pini2,const unsigned* dcellg,const word* codeg,tuint3 &cellmin,tuint3 &cellmax);
  void CalcCellDomainFluid(unsigned n,unsigned pini,unsigned n2,unsigned pini2,const unsigned* dcellg,const word* codeg,tuint3 &cellmin,tuint3 &cellmax);

  void CellBeginEnd(unsigned cell,unsigned ndata,unsigned* data)const;
  int2 CellBeginEnd(unsigned cell)const;
  unsigned CellSize(unsigned cell)const{ int2 v=CellBeginEnd(cell); return(unsigned(v.y-v.x)); }

public:
  JCellDivGpu(bool stable,bool floating,byte periactive,TpCellOrder cellorder,TpCellMode cellmode,float scell,tdouble3 mapposmin,tdouble3 mapposmax,tuint3 mapcells,unsigned casenbound,unsigned casenfixed,unsigned casenpb,JLog2 *log,std::string dirout,bool allocfullnct=true,float overmemorynp=CELLDIV_OVERMEMORYNP,word overmemorycells=CELLDIV_OVERMEMORYCELLS);
  ~JCellDivGpu();
  void FreeMemoryGpu(){ FreeMemoryAll(); }

  void DefineDomain(unsigned cellcode,tuint3 domcelini,tuint3 domcelfin,tdouble3 domposmin,tdouble3 domposmax);

  void SortBasicArrays(const unsigned *idp,const word *code,const unsigned *dcell,const double2 *posxy,const double *posz,const float4 *velrhop,unsigned *idp2,word *code2,unsigned *dcell2,double2 *posxy2,double *posz2,float4 *velrhop2);
  void SortDataArrays(const float4 *a,float4 *a2);
  void SortDataArrays(const float *a,const float *b,float *a2,float *b2);
  void SortDataArrays(const double2 *a,const double *b,const float4 *c,double2 *a2,double *b2,float4 *c2);
  void SortDataArrays(const tsymatrix3f *a,tsymatrix3f *a2);

  void CheckParticlesOut(unsigned npfout,const unsigned *idp,const tdouble3 *pos,const float *rhop,const word *code);
  float* GetAuxMem(unsigned size);

  TpCellMode GetCellMode()const{ return(CellMode); }
  unsigned GetHdiv()const{ return(Hdiv); }
  float GetScell()const{ return(Scell); }

  unsigned GetNct()const{ return(Nct); }
  unsigned GetNcx()const{ return(Ncx); }
  unsigned GetNcy()const{ return(Ncy); }
  unsigned GetNcz()const{ return(Ncz); }
  tuint3 GetNcells()const{ return(TUint3(Ncx,Ncy,Ncz)); }
  unsigned GetBoxFluid()const{ return(BoxFluid); }

  tuint3 GetCellDomainMin()const{ return(CellDomainMin); }
  tuint3 GetCellDomainMax()const{ return(CellDomainMax); }
  tdouble3 GetDomainLimits(bool limitmin,unsigned slicecellmin=0)const;

  unsigned GetNpFinal()const{ return(NpFinal); }
  unsigned GetNpbFinal()const{ return(NpbFinal); }
  unsigned GetNpbIgnore()const{ return(NpbIgnore); }
  unsigned GetNpOut()const{ return(NpbOut+NpfOut); }
  unsigned GetNpbOutIgnore()const{ return(NpbOutIgnore); }
  unsigned GetNpfOutIgnore()const{ return(NpfOutIgnore); }

  unsigned GetNpfOutPos()const{ return(NpfOut-(NpfOutMove+NpfOutRhop)); }
  unsigned GetNpfOutMove()const{ return(NpfOutMove); }
  unsigned GetNpfOutRhop()const{ return(NpfOutRhop); }

  const int2* GetBeginCell(){ return(BeginEndCell); }
};

#endif


