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

/// \file JPartsLoad4.cpp \brief Implements the class \ref JPartsLoad4.

#include "JPartsLoad4.h"
#include "Functions.h"
#include "JPartDataBi4.h"
#include "JRadixSort.h"
#include <climits>
#include <cfloat>

using namespace std;

//==============================================================================
// Constructor.
//==============================================================================
JPartsLoad4::JPartsLoad4(){
  ClassName="JPartsLoad4";
  Idp=NULL; Pos=NULL; VelRhop=NULL;
  Reset();
}

//==============================================================================
// Destructor.
//==============================================================================
JPartsLoad4::~JPartsLoad4(){
  AllocMemory(0);
}

//==============================================================================
// Initialisation of variables.
//==============================================================================
void JPartsLoad4::Reset(){
  Simulate2D=false;
  CaseNp=CaseNfixed=CaseNmoving=CaseNfloat=CaseNfluid=0;
  PeriMode=PERI_None;
  PeriXinc=PeriYinc=PeriZinc=TDouble3(0);
  MapSize=false;
  MapPosMin=MapPosMax=TDouble3(0);
  PartBegin=0;
  PartBeginTimeStep=0;
  AllocMemory(0);
}

//==============================================================================
// Redimensiona espacio para datos de las particulas.
//==============================================================================
void JPartsLoad4::AllocMemory(unsigned count){
  Count=count;
  delete[] Idp;      Idp=NULL; 
  delete[] Pos;      Pos=NULL; 
  delete[] VelRhop;  VelRhop=NULL; 
  if(Count){
    try{
      Idp=new unsigned[Count];
      Pos=new tdouble3[Count];
      VelRhop=new tfloat4[Count];
    }
    catch(const std::bad_alloc){
      RunException("AllocMemory","Could not allocate the requested memory.");
    }
  } 
}

//==============================================================================
// Devuelve la memoria reservada en cpu.
//==============================================================================
llong JPartsLoad4::GetAllocMemory()const{  
  llong s=0;
  //Reservada en AllocMemory()
  if(Idp)s+=sizeof(unsigned)*Count;
  if(Pos)s+=sizeof(tdouble3)*Count;
  if(VelRhop)s+=sizeof(tfloat4)*Count;
  return(s);
}

//==============================================================================
// Ordena valores segun vsort[].
//==============================================================================
template<typename T> T* JPartsLoad4::SortParticles(const unsigned *vsort,unsigned count,T *v)const{
  T* v2=new T[count];
  for(unsigned p=0;p<count;p++)v2[p]=v[vsort[p]];
  delete[] v;
  return(v2);
}

//==============================================================================
// Ordena particulas por Idp[].
//==============================================================================
void JPartsLoad4::SortParticles(){
  //-Comprueba orden.
  bool sorted=true;
  for(unsigned p=1;p<Count&&sorted;p++)sorted=(Idp[p-1]<Idp[p]);
  if(!sorted){
    //-Ordena puntos segun id.
    JRadixSort rs(false);
    rs.Sort(true,Count,Idp);
    rs.SortData(Count,Pos,Pos);
    rs.SortData(Count,VelRhop,VelRhop);
  }
}

//==============================================================================
// Carga particulas de fichero bi4 y las ordena por Id.
//==============================================================================
void JPartsLoad4::LoadParticles(const std::string &casedir,const std::string &casename,unsigned partbegin,const std::string &casedirbegin){
  const char met[]="LoadParticles";
  Reset();
  PartBegin=partbegin;
  JPartDataBi4 pd;
  //-Carga fichero piece_0 y obtiene configuracion.
  const string dir=fun::GetDirWithSlash(!PartBegin? casedir: casedirbegin);
  if(!PartBegin){
    const string file1=dir+JPartDataBi4::GetFileNameCase(casename,0,1);
    if(fun::FileExists(file1))pd.LoadFileCase(dir,casename,0,1);
    else if(fun::FileExists(dir+JPartDataBi4::GetFileNameCase(casename,0,2)))pd.LoadFileCase(dir,casename,0,2);
    else RunException(met,"File of the particles was not found.",file1);
  }
  else{
    const string file1=dir+JPartDataBi4::GetFileNamePart(PartBegin,0,1);
    if(fun::FileExists(file1))pd.LoadFilePart(dir,PartBegin,0,1);
    else if(fun::FileExists(dir+JPartDataBi4::GetFileNamePart(PartBegin,0,2)))pd.LoadFilePart(dir,PartBegin,0,2);
    else RunException(met,"File of the particles was not found.",file1);
  }
  //-Obtiene configuracion.
  PartBeginTimeStep=(!PartBegin? 0: pd.Get_TimeStep());
  Npiece=pd.GetNpiece();
  Simulate2D=pd.Get_Data2d();
  NpDynamic=pd.Get_NpDynamic();
  PartBeginTotalNp=(NpDynamic? pd.Get_NpTotal(): 0);
  CaseNp=pd.Get_CaseNp();
  CaseNfixed=pd.Get_CaseNfixed();
  CaseNmoving=pd.Get_CaseNmoving();
  CaseNfloat=pd.Get_CaseNfloat();
  CaseNfluid=pd.Get_CaseNfluid();
  JPartDataBi4::TpPeri peri=pd.Get_PeriActive();
  if(peri==JPartDataBi4::PERI_None)PeriMode=PERI_None;
  else if(peri==JPartDataBi4::PERI_X)PeriMode=PERI_X;
  else if(peri==JPartDataBi4::PERI_Y)PeriMode=PERI_Y;
  else if(peri==JPartDataBi4::PERI_Z)PeriMode=PERI_Z;
  else if(peri==JPartDataBi4::PERI_XY)PeriMode=PERI_XY;
  else if(peri==JPartDataBi4::PERI_XZ)PeriMode=PERI_XZ;
  else if(peri==JPartDataBi4::PERI_YZ)PeriMode=PERI_YZ;
  else if(peri==JPartDataBi4::PERI_Unknown)PeriMode=PERI_Unknown;
  else RunException(met,"Periodic configuration is invalid.");
  PeriXinc=pd.Get_PeriXinc();
  PeriYinc=pd.Get_PeriYinc();
  PeriZinc=pd.Get_PeriZinc();
  MapPosMin=pd.Get_MapPosMin();
  MapPosMax=pd.Get_MapPosMax();
  MapSize=(MapPosMin!=MapPosMax);
  CasePosMin=pd.Get_CasePosMin();
  CasePosMax=pd.Get_CasePosMax();
  const bool possimple=pd.Get_PosSimple();
  if(!pd.Get_IdpSimple())RunException(met,"Only Idp (32 bits) is valid at the moment.");
  //-Calcula numero de particulas.
  unsigned sizetot=pd.Get_Npok();
  for(unsigned piece=1;piece<Npiece;piece++){
    JPartDataBi4 pd2;
    if(!PartBegin)pd2.LoadFileCase(dir,casename,piece,Npiece);
    else pd2.LoadFilePart(dir,PartBegin,piece,Npiece);
    sizetot+=pd.Get_Npok();
  }
  //-Reserva memoria.
  AllocMemory(sizetot);
  //-Carga particulas.
  {
    unsigned ntot=0;
    unsigned auxsize=0;
    tfloat3 *auxf3=NULL;
    float *auxf=NULL;
    for(unsigned piece=0;piece<Npiece;piece++){
      if(piece){
        if(!PartBegin)pd.LoadFileCase(dir,casename,piece,Npiece);
        else pd.LoadFilePart(dir,PartBegin,piece,Npiece);
      }
      const unsigned npok=pd.Get_Npok();
      if(npok){
        if(auxsize<npok){
          auxsize=npok;
          delete[] auxf3; auxf3=NULL;
          delete[] auxf;  auxf=NULL;
          auxf3=new tfloat3[auxsize];
          auxf=new float[auxsize];
        }
        if(possimple){
          pd.Get_Pos(npok,auxf3);
          for(unsigned p=0;p<npok;p++)Pos[ntot+p]=ToTDouble3(auxf3[p]);
        }
        else pd.Get_Posd(npok,Pos+ntot);
        pd.Get_Idp(npok,Idp+ntot);  
        pd.Get_Vel(npok,auxf3);  
        pd.Get_Rhop(npok,auxf);  
        for(unsigned p=0;p<npok;p++)VelRhop[ntot+p]=TFloat4(auxf3[p].x,auxf3[p].y,auxf3[p].z,auxf[p]);
      }
      ntot+=npok;
    }
    delete[] auxf3; auxf3=NULL;
    delete[] auxf;  auxf=NULL;
  }
  //-Ordena particulas por Id.
  SortParticles();
}

//==============================================================================
// Comprueba validez de la configuracion cargada o lanza excepcion.
//==============================================================================
void JPartsLoad4::CheckConfig(ullong casenp,ullong casenfixed,ullong casenmoving,ullong casenfloat,ullong casenfluid,bool perix,bool periy,bool periz)const
{
  const char met[]="CheckConfig";
  if(casenp!=CaseNp||casenfixed!=CaseNfixed||casenmoving!=CaseNmoving||casenfloat!=CaseNfloat||casenfluid!=CaseNfluid)RunException(met,"Data file does not match the configuration of the case.");
  //-Obtiene modo periodico y compara con el cargado del fichero.
  TpPeri tperi=PERI_None;
  if(perix&&periy)tperi=PERI_XY;
  else if(perix&&periz)tperi=PERI_XZ;
  else if(periy&&periz)tperi=PERI_YZ;
  else if(perix)tperi=PERI_X;
  else if(periy)tperi=PERI_Y;
  else if(periz)tperi=PERI_Z;
  if(tperi!=PeriMode&&PeriMode!=PERI_Unknown)RunException(met,"Data file does not match the periodic configuration of the case.");
}

//==============================================================================
// Elimina particulas de contorno.
//==============================================================================
void JPartsLoad4::RemoveBoundary(){
  const unsigned casenbound=unsigned(CaseNp-CaseNfluid);
  //-Cuenta numero de particulas bound.
  unsigned nbound=0;
  for(;nbound<Count && Idp[nbound]<casenbound;nbound++);
  //-Saves old pointers and allocates new memory.
  unsigned count0=Count;
  unsigned *idp0=Idp;        Idp=NULL;
  tdouble3 *pos0=Pos;        Pos=NULL;
  tfloat4 *velrhop0=VelRhop; VelRhop=NULL;
  AllocMemory(count0-nbound);
  //-Copies data in new pointers.
  memcpy(Idp,idp0+nbound,sizeof(unsigned)*Count);
  memcpy(Pos,pos0+nbound,sizeof(tdouble3)*Count);
  memcpy(VelRhop,velrhop0+nbound,sizeof(tfloat4)*Count);
  //-Frees old pointers.
  delete[] idp0;      idp0=NULL; 
  delete[] pos0;      pos0=NULL; 
  delete[] velrhop0;  velrhop0=NULL; 
}

//==============================================================================
// Devuelve los limites del mapa y si no son validos genera excepcion.
//==============================================================================
void JPartsLoad4::GetMapSize(tdouble3 &mapmin,tdouble3 &mapmax)const{
  if(!MapSizeLoaded())RunException("GetMapSize","The MapSize information is invalid.");
  mapmin=MapPosMin; mapmax=MapPosMax;
}

//==============================================================================
// Calcula limites de las particulas cargadas.
//==============================================================================
void JPartsLoad4::CalculateCasePos(){
  if(!PartBegin)RunException("CalculateCasePos","The limits of the initial case cannot be calculated from a file PART.");
  tdouble3 pmin=TDouble3(DBL_MAX),pmax=TDouble3(-DBL_MAX);
  //-Calcula posicion minima y maxima. 
  for(unsigned p=0;p<Count;p++){
    const tdouble3 ps=Pos[p];
    if(pmin.x>ps.x)pmin.x=ps.x;
    if(pmin.y>ps.y)pmin.y=ps.y;
    if(pmin.z>ps.z)pmin.z=ps.z;
    if(pmax.x<ps.x)pmax.x=ps.x;
    if(pmax.y<ps.y)pmax.y=ps.y;
    if(pmax.z<ps.z)pmax.z=ps.z;
  }
  CasePosMin=pmin; CasePosMax=pmax;
}

//==============================================================================
// Calcula y devuelve limites de particulas con el borde indicado.
//==============================================================================
void JPartsLoad4::CalculeLimits(double border,double borderperi,bool perix,bool periy,bool periz,tdouble3 &mapmin,tdouble3 &mapmax){
  if(CasePosMin==CasePosMax)CalculateCasePos();
  tdouble3 bor=TDouble3(border);
  if(perix)bor.x=borderperi;
  if(periy)bor.y=borderperi;
  if(periz)bor.z=borderperi;
  mapmin=CasePosMin-bor;
  mapmax=CasePosMax+bor;
}