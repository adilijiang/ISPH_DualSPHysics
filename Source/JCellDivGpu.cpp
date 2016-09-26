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

#include "JCellDivGpu.h"
#include "JCellDivGpu_ker.h"
#include "Functions.h"
#include "JFormatFiles2.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JCellDivGpu::JCellDivGpu(bool stable,bool floating,byte periactive,TpCellOrder cellorder,TpCellMode cellmode,float scell,tdouble3 mapposmin,tdouble3 mapposmax,tuint3 mapcells,unsigned casenbound,unsigned casenfixed,unsigned casenpb,JLog2 *log,std::string dirout,bool allocfullnct,float overmemorynp,word overmemorycells):Stable(stable),Floating(floating),PeriActive(periactive),CellOrder(cellorder),CellMode(cellmode),Hdiv(cellmode==CELLMODE_2H? 1: (cellmode==CELLMODE_H? 2: 0)),Scell(scell),OvScell(1.f/scell),Map_PosMin(mapposmin),Map_PosMax(mapposmax),Map_PosDif(mapposmax-mapposmin),Map_Cells(mapcells),CaseNbound(casenbound),CaseNfixed(casenfixed),CaseNpb(casenpb),Log(log),DirOut(dirout),AllocFullNct(allocfullnct),OverMemoryNp(overmemorynp),OverMemoryCells(overmemorycells)
{
  ClassName="JCellDivGpu";
  CellPart=NULL;  SortPart=NULL;  AuxMem=NULL;
  BeginEndCell=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JCellDivGpu::~JCellDivGpu(){
  Reset();
}
 
//==============================================================================
/// Initialization of variables.
//==============================================================================
void JCellDivGpu::Reset(){
  SizeNp=SizeAuxMem=SizeNct=0;
  FreeMemoryAll();
  Ndiv=NdivFull=0;
  Nptot=Npb1=Npf1=Npb2=Npf2=0;
  MemAllocGpuNp=MemAllocGpuNct=0;
  NpbOut=NpfOut=NpbOutIgnore=NpfOutIgnore=0;
  NpFinal=NpbFinal=0;
  NpfOutRhop=NpfOutMove=NpbIgnore=0;
  CellDomainMin=TUint3(1);
  CellDomainMax=TUint3(0);
  Ncx=Ncy=Ncz=Nsheet=Nct=0;
  Nctt=0;
  BoundLimitOk=BoundDivideOk=false;
  BoundLimitCellMin=BoundLimitCellMax=TUint3(0);
  BoundDivideCellMin=BoundDivideCellMax=TUint3(0);
  DivideFull=false;
}

//==============================================================================
/// Libera memoria reservada para celdas.
/// Releases the memory allocated for cells
//==============================================================================
void JCellDivGpu::FreeMemoryNct(){
  cudaFree(BeginEndCell); BeginEndCell=NULL;
  MemAllocGpuNct=0;
  BoundDivideOk=false;
}

//==============================================================================
/// Libera memoria basica reservada para particulas y celdas.
/// Releases the basic memory allocated for particles and cells.
//==============================================================================
void JCellDivGpu::FreeMemoryAll(){
  FreeMemoryNct();
  cudaFree(CellPart);  CellPart=NULL;
  cudaFree(SortPart);  SortPart=NULL;
  cudaFree(AuxMem);    AuxMem=NULL; 
  MemAllocGpuNp=0;
  BoundDivideOk=false;
}

//==============================================================================
/// Asigna memoria basica segun numero de particulas.
/// Assigns basic memory according to the particle number.
//==============================================================================
void JCellDivGpu::AllocMemoryNp(ullong np){
  const char met[]="AllocMemoryNp";
  FreeMemoryAll();
  SizeNp=unsigned(np);
  //-Comprueba numero de particulas.
  //-Checks particle number
  if(np!=SizeNp)RunException(met,string("Failed GPU memory allocation for ")+fun::UlongStr(np)+" particles.");
  //-Reserva memoria para particulas.
  //-Allocates memory for the particles
  MemAllocGpuNp=0;
  size_t m=sizeof(unsigned)*SizeNp;
  cudaMalloc((void**)&CellPart,m); MemAllocGpuNp+=m;
  cudaMalloc((void**)&SortPart,m); MemAllocGpuNp+=m;
  SizeAuxMem=cudiv::LimitsPosSize(SizeNp);
  m=sizeof(float)*SizeAuxMem;
  cudaMalloc((void**)&AuxMem,m);   MemAllocGpuNp+=m;
  //-Comprueba reserva de memoria.
  //-Checks allocated memory.
  cudaError_t cuerr=cudaGetLastError();
  if(cuerr!=cudaSuccess){
    RunExceptionCuda(met,fun::PrintStr("Failed CPU memory allocation of %.1f MB for %u particles.",double(MemAllocGpuNp)/(1024*1024),SizeNp),cuerr);
  }
  //-Muestra la memoria solicitada.
  //-Displays the requested memory.
  Log->Printf("**CellDiv: Requested gpu memory for %u particles: %.1f MB.",SizeNp,double(MemAllocGpuNp)/(1024*1024));
}

//==============================================================================
/// Asigna memoria segun numero de celdas.
/// Assigns memory according to the cell number.
//==============================================================================
void JCellDivGpu::AllocMemoryNct(ullong nct){
  const char met[]="AllocMemoryNct";
  FreeMemoryNct();
  SizeNct=unsigned(nct);
  //-Comprueba numero de celdas.
  //-Checks cell number.
  if(nct!=SizeNct)RunException(met,string("Failed GPU memory allocation for ")+fun::UlongStr(nct)+" cells.");
  //-Reserva memoria para celdas.
  //-Allocates memory for cells.
  MemAllocGpuNct=0;
  size_t m=sizeof(int2)*SizeBeginEndCell(SizeNct);
  cudaMalloc((void**)&BeginEndCell,m); MemAllocGpuNct+=m;
  //-Comprueba reserva de memoria.
  //-Checks allocated memory
  cudaError_t cuerr=cudaGetLastError();
  if(cuerr!=cudaSuccess){
    RunExceptionCuda(met,fun::PrintStr("Failed GPU memory allocation of %.1f MB for %u cells.",double(MemAllocGpuNct)/(1024*1024),SizeNct),cuerr);
  }
  //-Muestra la memoria solicitada.
  //-Displays requested memory
  Log->Printf("**CellDiv: Requested gpu memory for %u cells (CellMode=%s): %.1f MB.",SizeNct,GetNameCellMode(CellMode),double(MemAllocGpuNct)/(1024*1024));
}

//==============================================================================
/// ES:
/// Comprueba la reserva de memoria para el numero indicado de particulas. 
/// Si no es suficiente o no hay reserva, entonces reserva la memoria requerida.
///
/// EN:
/// Checks allocated memory for the indicated number of particles.
/// If the allocated memeory is not sufficient, reserve the required memory.
//==============================================================================
void JCellDivGpu::CheckMemoryNp(unsigned npmin){
  if(SizeNp<npmin)AllocMemoryNp(ullong(npmin)+ullong(OverMemoryNp*npmin));
  else if(!CellPart)AllocMemoryNp(SizeNp);  
}

//==============================================================================
/// ES:
/// Comprueba la reserva de memoria para el numero indicado de celdas. 
/// Si no es suficiente o no hay reserva, entonces reserva la memoria requerida.
///
/// EN:
/// Checks allocated memory for the indicated number of cells.
/// If the allocated memeory is not sufficient, reserve the required memory.
//==============================================================================
void JCellDivGpu::CheckMemoryNct(unsigned nctmin){
  if(SizeNct<nctmin){
    unsigned overnct=0;
    if(OverMemoryCells>0){
      ullong nct=ullong(Ncx+OverMemoryCells)*ullong(Ncy+OverMemoryCells)*ullong(Ncz+OverMemoryCells);
      ullong nctt=SizeBeginEndCell(nct);
      if(nctt!=unsigned(nctt))RunException("CheckMemoryNct","The number of cells is too big.");
      overnct=unsigned(nct);
    }
    AllocMemoryNct(nctmin>overnct? nctmin: overnct);
  }
  else if(!BeginEndCell)AllocMemoryNct(SizeNct);  
}

//==============================================================================
/// Define el dominio de simulacion a usar.
/// Defines the domain to be used by the simulation.
//==============================================================================
void JCellDivGpu::DefineDomain(unsigned cellcode,tuint3 domcelini,tuint3 domcelfin,tdouble3 domposmin,tdouble3 domposmax){
  DomCellCode=cellcode;
  DomCelIni=domcelini;
  DomCelFin=domcelfin;
  DomPosMin=domposmin;
  DomPosMax=domposmax;
  DomCells=DomCelFin-DomCelIni;
}

//==============================================================================
/// Visualiza la informacion de una particula de contorno excluida.
/// Displays information for an excluded boundary particle.
//==============================================================================
void JCellDivGpu::VisuBoundaryOut(unsigned p,unsigned id,tdouble3 pos,word code)const{
  string info="particle boundary out> type:";
  word tp=CODE_GetType(code);
  if(tp==CODE_TYPE_FIXED)info=info+"Fixed";
  else if(tp==CODE_TYPE_MOVING)info=info+"Moving";
  else if(tp==CODE_TYPE_FLOATING)info=info+"Floating";
  info=info+" cause:";
  word out=CODE_GetSpecialValue(code);
  if(out==CODE_OUTMOVE)info=info+"Speed";
  else if(out==CODE_OUTPOS)info=info+"Position";
  else info=info+"???";
  Log->PrintDbg(info+fun::PrintStr(" p:%u id:%u pos:(%f,%f,%f)",p,id,pos.x,pos.y,pos.z));
}

//==============================================================================
/// ES:
/// Calcula posiciones minimas y maximas del rango de particulas Bound indicado.
/// En code[] ya estan marcadas las particulas excluidas.
///
/// EN:
/// Computes the maximum and minimum postion for the indicated range of boundary particles
/// The excluded particles are already marked in code[]
//==============================================================================
void JCellDivGpu::CalcCellDomainBound(unsigned n,unsigned pini,unsigned n2,unsigned pini2,const unsigned* dcellg,const word* codeg,tuint3 &cellmin,tuint3 &cellmax){
  tuint3 cmin,cmax;
  cudiv::LimitsCell(n,pini,DomCellCode,dcellg,codeg,(unsigned*)AuxMem,cmin,cmax,Log);
  cellmin=(cmin.x>cmax.x? DomCells: cmin);
  cellmax=(cmin.x>cmax.x? TUint3(0): cmax);
  if(n2){
    cudiv::LimitsCell(n2,pini2,DomCellCode,dcellg,codeg,(unsigned*)AuxMem,cmin,cmax,Log);
    cmin=(cmin.x>cmax.x? DomCells: cmin);
    cmax=(cmin.x>cmax.x? TUint3(0): cmax);
    cellmin=MinValues(cellmin,cmin);
    cellmax=MaxValues(cellmax,cmax);
  }
}

//==============================================================================
/// ES:
/// Calcula posiciones minimas y maximas del rango de particulas Fluid indicado.
/// Ignora particulas excluidas que ya estan marcadas en code[].
///
/// EN:
/// Computes the maximum and minimum postion for the indicated range of fluid particles
/// Ignores the excluded particles already marked in code[]
//==============================================================================
void JCellDivGpu::CalcCellDomainFluid(unsigned n,unsigned pini,unsigned n2,unsigned pini2,const unsigned* dcellg,const word* codeg,tuint3 &cellmin,tuint3 &cellmax){
  tuint3 cmin,cmax;
  cudiv::LimitsCell(n,pini,DomCellCode,dcellg,codeg,(unsigned*)AuxMem,cmin,cmax,Log);
  cellmin=(cmin.x>cmax.x? DomCells: cmin);
  cellmax=(cmin.x>cmax.x? TUint3(0): cmax);
  if(n2){
    cudiv::LimitsCell(n2,pini2,DomCellCode,dcellg,codeg,(unsigned*)AuxMem,cmin,cmax,Log);
    cmin=(cmin.x>cmax.x? DomCells: cmin);
    cmax=(cmin.x>cmax.x? TUint3(0): cmax);
    cellmin=MinValues(cellmin,cmin);
    cellmax=MaxValues(cellmax,cmax);
  }
}

//==============================================================================
/// Devuelve principo y final de la celda indicada.
/// Returns first and last particle of the indicated cell
//==============================================================================
void JCellDivGpu::CellBeginEnd(unsigned cell,unsigned ndata,unsigned* data)const{
  cudaMemcpy(data,BeginEndCell+cell,sizeof(int)*ndata,cudaMemcpyDeviceToHost);
}

//==============================================================================
// Devuelve principo y final de la celda indicada.
/// Returns first and last particle of the indicated cell
//==============================================================================
int2 JCellDivGpu::CellBeginEnd(unsigned cell)const{
  int2 v;
  cudaMemcpy(&v,BeginEndCell+cell,sizeof(int2),cudaMemcpyDeviceToHost);
  return(v);
}

//==============================================================================
/// Ordena arrays basicos segun SortPart. 
/// Reorders basic arrays according to SortPart
//==============================================================================
void JCellDivGpu::SortBasicArrays(const unsigned *idp,const word *code,const unsigned *dcell,const double2 *posxy,const double *posz,const double4 *velrhop,unsigned *idp2,word *code2,unsigned *dcell2,double2 *posxy2,double *posz2,double4 *velrhop2){
  const unsigned pini=(DivideFull? 0: NpbFinal);
  cudiv::SortDataParticles(Nptot,pini,SortPart,idp,code,dcell,posxy,posz,velrhop,idp2,code2,dcell2,posxy2,posz2,velrhop2);
}

//==============================================================================
/// Ordena arrays de datos segun SortPart. 
/// Reorders data arrays according to SortPart.
//==============================================================================
void JCellDivGpu::SortDataArrays(const float4 *a,float4 *a2){
  const unsigned pini=(DivideFull? 0: NpbFinal);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,a2);
}
//==============================================================================
void JCellDivGpu::SortDataArrays(const float *a,const float *b,float *a2,float *b2){
  const unsigned pini=(DivideFull? 0: NpbFinal);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,b,a2,b2);
}
//==============================================================================
void JCellDivGpu::SortDataArrays(const double2 *a,const double *b,const double4 *c,double2 *a2,double *b2,double4 *c2){
  const unsigned pini=(DivideFull? 0: NpbFinal);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,b,c,a2,b2,c2);
}
//==============================================================================
void JCellDivGpu::SortDataArrays(const tsymatrix3f *a,tsymatrix3f *a2){
  const unsigned pini=(DivideFull? 0: NpbFinal);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,a2);
}

//==============================================================================
/// ES:
/// Finaliza proceso de Divide: Revisando que todas las excluidas sean fluidas
/// y calculando el numero de excluidas por pos, rhop o mov.
/// Las componentes de los datos ya estan en el orden original.
///
/// EN:
/// Finalises Divide process: Checks that all excluded particles are fluid
/// and computes the number of excluded by pos, rhop or mov.
/// The data components are already in the original order.
//==============================================================================
void JCellDivGpu::CheckParticlesOut(unsigned npout,const unsigned *idp,const tdouble3 *pos,const float *rhop,const word *code){
  unsigned nerr=0;
  for(unsigned p=0;p<npout;p++){
    word type=CODE_GetType(code[p]);
    if(nerr<10&&type==CODE_TYPE_FIXED||type==CODE_TYPE_MOVING||type==CODE_TYPE_FLOATING){ //-Hay alguna particula de contorno excluida.
      nerr++;
      if(nerr<100)VisuBoundaryOut(p,idp[p],pos[p],code[p]);
    }
    word out=CODE_GetSpecialValue(code[p]);
    if(out==CODE_OUTRHOP)NpfOutRhop++;
    else if(out==CODE_OUTMOVE)NpfOutMove++;
  }
  if(nerr)RunException("CheckParticlesOut","A boundary particle was excluded.");
}

//==============================================================================
/// ES:
/// Devuelve un puntero con la memoria auxiliar reservada en GPU, que solo se usa
/// como almacenamiento intermedio durante ciertos procesos. Asi es posible
/// aprovechar esta memoria para otros usos.
/// Esta memoria se redimensiona segun el numero de particulas por lo que su
/// tamaño y direccion pueden variar.
///
/// EN:
/// Returns a pointer with the auxiliary memory allocated in the GPU, only
/// used as intermediate in some tasks, in order to use it in other tasks.
/// This memoery is resized according to the particle number thus its
/// size and direction can vary.
//==============================================================================
float* JCellDivGpu::GetAuxMem(unsigned size){
  if(size>SizeAuxMem)RunException("GetAuxMem","The requested memory is not available.");
  return(AuxMem);
}

//==============================================================================
/// Devuelve limites actuales del dominio.
/// Returns the actual limits of the domain.
//==============================================================================
tdouble3 JCellDivGpu::GetDomainLimits(bool limitmin,unsigned slicecellmin)const{
  tuint3 celmin=GetCellDomainMin(),celmax=GetCellDomainMax();
  if(celmin.x>celmax.x)celmin.x=celmax.x=0; else celmax.x++;
  if(celmin.y>celmax.y)celmin.y=celmax.y=0; else celmax.y++;
  if(celmin.z>celmax.z)celmin.z=celmax.z=slicecellmin; else celmax.z++;
  double scell=double(Scell);
  tdouble3 pmin=DomPosMin+TDouble3(scell*celmin.x,scell*celmin.y,scell*celmin.z);
  tdouble3 pmax=DomPosMin+TDouble3(scell*celmax.x,scell*celmax.y,scell*celmax.z);
  return(limitmin? pmin: pmax);
}


