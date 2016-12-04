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

#include "JCellDivGpuSingle.h"
#include "JCellDivGpuSingle_ker.h"
#include "Functions.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JCellDivGpuSingle::JCellDivGpuSingle(bool stable,bool floating,byte periactive,TpCellOrder cellorder,TpCellMode cellmode,float scell,tdouble3 mapposmin,tdouble3 mapposmax,tuint3 mapcells,unsigned casenbound,unsigned casenfixed,unsigned casenpb,JLog2 *log,std::string dirout):JCellDivGpu(stable,floating,periactive,cellorder,cellmode,scell,mapposmin,mapposmax,mapcells,casenbound,casenfixed,casenpb,log,dirout){
  ClassName="JCellDivGpuSingle";
}

//==============================================================================
/// Calcula limites del dominio en celdas ajustando al fluido (CellDomainMin/Max). 
/// Computes cell domains adjusting to the fluid CellDomainMin/Max
//==============================================================================
void JCellDivGpuSingle::CalcCellDomain(const unsigned *dcellg,const word* codeg){
  //-Calcula dominio del contorno.
  //-Computes the boundary domain
  tuint3 celbmin,celbmax;
  if(!BoundLimitOk){
    CalcCellDomainBound(Npb1,0,Npb2,Npb1+Npf1,dcellg,codeg,celbmin,celbmax);
    BoundLimitOk=true; BoundLimitCellMin=celbmin; BoundLimitCellMax=celbmax;
  } 
  else{ celbmin=BoundLimitCellMin; celbmax=BoundLimitCellMax; }
  //-Calcula dominio del fluido.
  //-Computes the fluid domain
  tuint3 celfmin,celfmax;
  CalcCellDomainFluid(Npf1,Npb1,Npf2,Npb1+Npf1+Npb2,dcellg,codeg,celfmin,celfmax);
  //-Calcula dominio ajustando al contorno y al fluido (con halo de 2h). 
  //-Computes the domain adjusting to the boundary and the fluid ( with 2h halo)
  MergeMapCellBoundFluid(celbmin,celbmax,celfmin,celfmax,CellDomainMin,CellDomainMax);
  celfmin=CellDomainMin;
  celfmax=CellDomainMax;
  MergeMapCellBoundFluid(celbmin,celbmax,celfmin,celfmax,CellDomainMin,CellDomainMax);
}

//==============================================================================
/// ES:
/// Combina limite de celdas de contorno y fluido con limites de mapa.
/// Con UseFluidDomain=TRUE se queda con el dominio del fluido mas 2h si hay 
/// contorno, en caso contrario se queda con el dominio que incluya fluido y
/// contorno.
/// En caso de que el dominio sea nulo CellDomainMin=CellDomainMax=(0,0,0).
///
/// EN:
/// Combines cell limits of boundary and fluid with map limits
/// If UseFluidDomain=TRUE, uses fluid domain plus 2h if there is a boundary;
/// if not, uses the fluid and boundary domain
/// If the domain is null CellDomainMin=CellDomainMax=(0,0,0).
//==============================================================================
void JCellDivGpuSingle::MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const{
  celmin=TUint3(max(min(celbmin.x,celfmin.x),(celfmin.x>=Hdiv? celfmin.x-Hdiv: 0)),max(min(celbmin.y,celfmin.y),(celfmin.y>=Hdiv? celfmin.y-Hdiv: 0)),max(min(celbmin.z,celfmin.z),(celfmin.z>=Hdiv? celfmin.z-Hdiv: 0)));
  celmax=TUint3(min(max(celbmax.x,celfmax.x),celfmax.x+Hdiv),min(max(celbmax.y,celfmax.y),celfmax.y+Hdiv),min(max(celbmax.z,celfmax.z),celfmax.z+Hdiv));
  if(celmax.x>=DomCells.x)celmax.x=DomCells.x-1;
  if(celmax.y>=DomCells.y)celmax.y=DomCells.y-1;
  if(celmax.z>=DomCells.z)celmax.z=DomCells.z-1;
  if(celmin.x>celmax.x||celmin.y>celmax.y||celmin.z>celmax.z){ celmin=celmax=TUint3(0,0,0); }
}

//==============================================================================
/// ES:
/// Calcula numero de celdas a partir de (CellDomainMin/Max). 
/// Obtiene localizacion de celdas especiales.
///
/// EN:
/// Computes number of cells starting from (CellDomainMIN/Max)
/// Obtains location of special cells
//==============================================================================
void JCellDivGpuSingle::PrepareNct(){
  //-Calcula numero de celdas.
  //-Computes number of cells.
  Ncx=CellDomainMax.x-CellDomainMin.x+1;
  Ncy=CellDomainMax.y-CellDomainMin.y+1;
  Ncz=CellDomainMax.z-CellDomainMin.z+1;
  Nsheet=Ncx*Ncy; Nct=Nsheet*Ncz; Nctt=SizeBeginEndCell(Nct);
  if(Nctt!=unsigned(Nctt))RunException("PrepareNct","The number of cells is too big.");
  BoxIgnore=Nct; 
  BoxFluid=BoxIgnore+1; 
  BoxBoundOut=BoxFluid+Nct; 
  BoxFluidOut=BoxBoundOut+1; 
  BoxBoundOutIgnore=BoxFluidOut+1;
  BoxFluidOutIgnore=BoxBoundOutIgnore+1;
}

//==============================================================================
/// ES:
/// Calcula celda de cada particula (CellPart[]) a partir de dcell[], todas las
/// particulas excluidas ya fueron marcadas en code[].
/// Asigna valores consecutivos a SortPart[].
///
/// EN:
/// Computes cell of each particle (CellPart[]) from dcell[],
/// all the excluded particles have been marked in the code.
/// Assigns consecutive values to SortPart[]
//==============================================================================
void JCellDivGpuSingle::PreSort(const unsigned *dcellg,const word *codeg){
  if(DivideFull)cudiv::PreSortFull(Nptot,DomCellCode,dcellg,codeg,CellDomainMin,TUint3(Ncx,Ncy,Ncz),CellPart,SortPart,Log);
  else cudiv::PreSortFluid(Npf1,Npb1,DomCellCode,dcellg,codeg,CellDomainMin,TUint3(Ncx,Ncy,Ncz),CellPart,SortPart,Log);
}

//==============================================================================
/// ES:
/// Inicia proceso de Divide: Calcula limites de dominio y calcula nueva posicion
/// para cada particula (SortPart).
/// El valor np incluye las periodicas bound y fluid (npbper y npfper).
/// Las floating se tratan como si fuesen fluido (tanto al ser excluidas como ignoradas).
///
/// EN:
/// Initial processing of Divide: Calculte the limits of the domain and
/// compute the new position of each particle (SortPart).
/// The value for np includes periodic boundary and fluid particles (npbper and npfper)
/// The floating bodies are treated as fluids (both to be ignored as excluded)
//==============================================================================
void JCellDivGpuSingle::Divide(unsigned npb1,unsigned npf1,unsigned npb2,unsigned npf2,bool boundchanged,const unsigned *dcellg,const word* codeg,TimersGpu timers,const double2 *posxy,const double *posz,const unsigned *idp){
  const char met[]="Divide";
  DivideFull=false;
  TmgStart(timers,TMG_NlLimits);

  //-Establece numero de particulas.
  //-Establishes particle number
  Npb1=npb1; Npf1=npf1; Npb2=npb2; Npf2=npf2;
  Nptot=Npb1+Npf1+Npb2+Npf2;
  NpbOut=NpfOut=NpbOutIgnore=NpfOutIgnore=0;
  NpFinal=NpbFinal=0;
  NpfOutRhop=NpfOutMove=NpbIgnore=0;

  //-Comprueba si hay memoria reservada y si es suficiente para Nptot.
  //-Checks if the allocated memory is esufficient for Nptot
  CheckMemoryNp(Nptot);

  //-Si la posicion del contorno cambia o hay condiciones periodicas es necesario recalcular limites y reordenar todas las particulas. 
  //-If the boundary postion changes or there are periodic conditions it is necessary to recalculate the limits and reorder every particle.
  if(boundchanged || PeriActive){
    BoundLimitOk=BoundDivideOk=false;
    BoundLimitCellMin=BoundLimitCellMax=TUint3(0);
    BoundDivideCellMin=BoundDivideCellMax=TUint3(0);
  }

  //-Calcula limites del dominio.
  //-Computes domain limits
  CalcCellDomain(dcellg,codeg);
  //-Calcula numero de celdas para el divide y comprueba reserva de memoria para celdas.
  //-Computes the number of cells to be divided and check the allocated memory for the cells. 
  PrepareNct();
  //-Comprueba si hay memoria reservada y si es suficiente para Nptot.
  //-Checks if the allocated memory is sufficient for Nptot
  CheckMemoryNct(Nct);
  TmgStop(timers,TMG_NlLimits);

  //-ES:
  //-Determina si el divide afecta a todas las particulas.
  //-BoundDivideOk se vuelve false al reservar o liberar memoria para particulas o celdas.
  //-EN:
  //-Determines if the divide affects all the particles
  //-If BoundDivideOk becomes false, reserve free memory for particles or cells
  if(!BoundDivideOk || BoundDivideCellMin!=CellDomainMin || BoundDivideCellMax!=CellDomainMax){
    DivideFull=true;
    BoundDivideOk=true; BoundDivideCellMin=CellDomainMin; BoundDivideCellMax=CellDomainMax;
  }
  else DivideFull=false;

  //-Calcula CellPart[] y asigna valores consecutivos a SortPart[].
  //-Computes CellPart[] and assigns consecutive values to SortPart[].
  TmgStart(timers,TMG_NlPreSort);
  PreSort(dcellg,codeg);
  TmgStop(timers,TMG_NlPreSort);

  //-Ordena CellPart y SortPart en funcion de la celda.
  //-Sorts CellPart and SortPart as a function of the cell.
  TmgStart(timers,TMG_NlRadixSort);
  if(DivideFull)cudiv::Sort(CellPart,SortPart,Nptot,Stable);
  else cudiv::Sort(CellPart+Npb1,SortPart+Npb1,Nptot-Npb1,Stable);
  TmgStop(timers,TMG_NlRadixSort);

  //-Calcula particula inicial y final de cada celda (BeginEndCell).
  //-Computes the initial and the last paeticle of each cell (BeginEndCell).
  TmgStart(timers,TMG_NlCellBegin);
  cudiv::CalcBeginEndCell(DivideFull,Nptot,Npb1,unsigned(SizeBeginEndCell(Nct)),BoxFluid,CellPart,BeginEndCell);

  //-Calcula numeros de particulas.
  //-Computes number of particles.
  NpbIgnore=CellSize(BoxIgnore);
  unsigned beginendcell[8];
  CellBeginEnd(BoxBoundOut,8,beginendcell);
  NpbOut=beginendcell[1]-beginendcell[0];
  NpfOut=beginendcell[3]-beginendcell[2];
  NpbOutIgnore=beginendcell[5]-beginendcell[4];
  NpfOutIgnore=beginendcell[7]-beginendcell[6];
  NpFinal=Nptot-NpbOut-NpfOut-NpbOutIgnore-NpfOutIgnore;
  NpbFinal=Npb1+Npb2-NpbOut-NpbOutIgnore;

  Ndiv++;
  if(DivideFull)NdivFull++;
  TmgStop(timers,TMG_NlCellBegin);
  CheckCudaError(met,"Error in NL construction.");
}

void JCellDivGpuSingle::MirrorDCellSingle(unsigned bsbound,unsigned npb,const word *codeg,int *irelation,unsigned *idpg,const double3 *mirror,tdouble3 domrealposmin,tdouble3 domrealposmax,tdouble3 domposmin,float scell,int domcellcode){
	cudiv::MirrorDCell(bsbound,npb,codeg,irelation,idpg,mirror,domrealposmin,domrealposmax,domposmin,scell,domcellcode);
}


