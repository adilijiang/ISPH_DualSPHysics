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

#include "JSphCpuSingle.h"
#include "JCellDivCpuSingle.h"
#include "JArraysCpu.h"
#include "Functions.h"
#include "JXml.h"
#include "JSphMotion.h"
#include "JPartsLoad4.h"
#include "JSphVisco.h"
#include "JWaveGen.h"

#include <climits>
#include <time.h>

#include <vector>

using namespace std;

//==============================================================================
// Constructor.
//==============================================================================
JSphCpuSingle::JSphCpuSingle():JSphCpu(false){
  ClassName="JSphCpuSingle";
  CellDivSingle=NULL;
  PartsLoaded=NULL;
}

//==============================================================================
// Destructor.
//==============================================================================
JSphCpuSingle::~JSphCpuSingle(){
  delete CellDivSingle; CellDivSingle=NULL;
  delete PartsLoaded;   PartsLoaded=NULL;
}

//==============================================================================
/// Devuelve la memoria reservada en cpu.
/// Return memory reserved in CPU.
//==============================================================================
llong JSphCpuSingle::GetAllocMemoryCpu()const{  
  llong s=JSphCpu::GetAllocMemoryCpu();
  //Reservada en otros objetos
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemory();
  if(PartsLoaded)s+=PartsLoaded->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Actualiza los valores maximos de memory, particles y cells.
/// Update maximum values of memory, particles & cells.
//==============================================================================
void JSphCpuSingle::UpdateMaxValues(){
  MaxParticles=max(MaxParticles,Np);
  if(CellDivSingle)MaxCells=max(MaxCells,CellDivSingle->GetNct());
  llong m=GetAllocMemoryCpu();
  MaxMemoryCpu=max(MaxMemoryCpu,m);
}

//==============================================================================
/// Carga la configuracion de ejecucion.
/// Load the execution configuration.
//==============================================================================
void JSphCpuSingle::LoadConfig(JCfgRun *cfg){
  const char met[]="LoadConfig";
  //-Load OpenMP configuraction / Carga configuracion de OpenMP
  ConfigOmp(cfg);
  //-Load basic general configuraction / Carga configuracion basica general
  JSph::LoadConfig(cfg);
  //-Checks compatibility of selected options.
  Log->Print("**Special case configuration is loaded");
}

//==============================================================================
/// Carga particulas del caso a procesar.
/// Load particles of case and process.
//==============================================================================
void JSphCpuSingle::LoadCaseParticles(){
  Log->Print("Loading initial state of particles...");
  PartsLoaded=new JPartsLoad4;
  PartsLoaded->LoadParticles(DirCase,CaseName,PartBegin,PartBeginDir);
  PartsLoaded->CheckConfig(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,PeriX,PeriY,PeriZ);
  Log->Printf("Loaded particles: %u",PartsLoaded->GetCount());
  //-Collect information of loaded particles / Recupera informacion de las particulas cargadas.
  Simulate2D=PartsLoaded->GetSimulate2D();
  if(Simulate2D&&PeriY)RunException("LoadCaseParticles","Can not use periodic conditions in Y with 2D simulations");
  CasePosMin=PartsLoaded->GetCasePosMin();
  CasePosMax=PartsLoaded->GetCasePosMax();

  //-Calculate actual limits of simulation / Calcula limites reales de la simulacion.
  if(PartsLoaded->MapSizeLoaded())PartsLoaded->GetMapSize(MapRealPosMin,MapRealPosMax);
  else{
    PartsLoaded->CalculeLimits(double(H)*BORDER_MAP,Dp/2.,PeriX,PeriY,PeriZ,MapRealPosMin,MapRealPosMax);
    ResizeMapLimits();
  }
  if(PartBegin){
    PartBeginTimeStep=PartsLoaded->GetPartBeginTimeStep();
    PartBeginTotalNp=PartsLoaded->GetPartBeginTotalNp();
  }
  Log->Print(string("MapRealPos(final)=")+fun::Double3gRangeStr(MapRealPosMin,MapRealPosMax));
  MapRealSize=MapRealPosMax-MapRealPosMin;
  Log->Print("**Initial state of particles is loaded");

  //-Configure limits of periodic axes / Configura limites de ejes periodicos
  if(PeriX)PeriXinc.x=-MapRealSize.x;
  if(PeriY)PeriYinc.y=-MapRealSize.y;
  if(PeriZ)PeriZinc.z=-MapRealSize.z;
  //-Calculate simulation limits with periodic boundaries / Calcula limites de simulacion con bordes periodicos.
  Map_PosMin=MapRealPosMin; Map_PosMax=MapRealPosMax;
  float dosh=float(H*2);
  if(PeriX){ Map_PosMin.x=Map_PosMin.x-dosh;  Map_PosMax.x=Map_PosMax.x+dosh; }
  if(PeriY){ Map_PosMin.y=Map_PosMin.y-dosh;  Map_PosMax.y=Map_PosMax.y+dosh; }
  if(PeriZ){ Map_PosMin.z=Map_PosMin.z-dosh;  Map_PosMax.z=Map_PosMax.z+dosh; }
  Map_Size=Map_PosMax-Map_PosMin;
}

//==============================================================================
/// Configuracion del dominio actual.
/// Configuration of present domain.
//==============================================================================
void JSphCpuSingle::ConfigDomain(){
  const char* met="ConfigDomain";
  //-Calculate number of particles / Calcula numero de particulas.
  Np=PartsLoaded->GetCount(); Npb=CaseNpb; NpbOk=Npb;
   
  //-Reserve fixed memory for moving & floating particles / Reserva memoria fija para moving y floating.
  AllocCpuMemoryFixed();
  //-Reserve memory in CPU for particles / Reserva memoria en Cpu para particulas.
  AllocCpuMemoryParticles(Np,0);

  //-Copy particle values / Copia datos de particulas.
  ReserveBasicArraysCpu();
  memcpy(Posc,PartsLoaded->GetPos(),sizeof(tdouble3)*Np);
  memcpy(Idpc,PartsLoaded->GetIdp(),sizeof(unsigned)*Np);
  memcpy(Velrhopc,PartsLoaded->GetVelRhop(),sizeof(tfloat4)*Np);

  //-Calculate floating radius / Calcula radio de floatings.
  if(CaseNfloat && PeriActive!=0)CalcFloatingRadius(Np,Posc,Idpc);

  //-Load particle data / Carga code de particulas.
  LoadCodeParticles(Np,Idpc,Codec);

  //-Free memory of PartsLoaded / Libera memoria de PartsLoaded.
  delete PartsLoaded; PartsLoaded=NULL;
  //-Apply configuration of CellOrder / Aplica configuracion de CellOrder.
  ConfigCellOrder(CellOrder,Np,Posc,Velrhopc);

  //-Configure cells division / Configura division celdas.
  ConfigCellDivision();
  //-Establish local simulation domain inside of Map_Cells & calculate DomCellCode / Establece dominio de simulacion local dentro de Map_Cells y calcula DomCellCode.
  SelecDomain(TUint3(0,0,0),Map_Cells);
  //-Calculate initial cell of particles and check if there are unexpected excluded particles / Calcula celda inicial de particulas y comprueba si hay excluidas inesperadas.
  LoadDcellParticles(Np,Codec,Posc,Dcellc);

  //-Create object for divide in CPU & select a valid cellmode / Crea objeto para divide en Gpu y selecciona un cellmode valido.
  CellDivSingle=new JCellDivCpuSingle(Stable,FtCount!=0,PeriActive,CellOrder,CellMode,Scell,Map_PosMin,Map_PosMax,Map_Cells,CaseNbound,CaseNfixed,CaseNpb,Log,DirOut);
  CellDivSingle->DefineDomain(DomCellCode,DomCelIni,DomCelFin,DomPosMin,DomPosMax);
  ConfigCellDiv((JCellDivCpu*)CellDivSingle);

  ConfigSaveData(0,1,"");

  //-Reoder particles for cell / Reordena particulas por celda.
  BoundChanged=true;
  RunCellDivide(true);
}

//==============================================================================
/// (ES):
/// Redimensiona el espacio reservado para particulas en CPU midiendo el
/// tiempo consumido con TMC_SuResizeNp. Al terminar actualiza el divide.
/// (EN):
/// Redimension space reserved for particles in CPU, measure 
/// time consumed using TMC_SuResizeNp. On finishing, update divide.
//==============================================================================
void JSphCpuSingle::ResizeParticlesSize(unsigned newsize,float oversize,bool updatedivide){
  TmcStart(Timers,TMC_SuResizeNp);
  newsize+=(oversize>0? unsigned(oversize*newsize): 0);
  ResizeCpuMemoryParticles(newsize);
  TmcStop(Timers,TMC_SuResizeNp);
  if(updatedivide)RunCellDivide(true);
}

//==============================================================================
/// (ES):
/// Crea lista de nuevas particulas periodicas a duplicar.
/// Con stable activado reordena lista de periodicas.
/// (EN):
/// Create list of new periodic particles to duplicate.
/// With stable activated reordered list of periodic particles.
//==============================================================================
unsigned JSphCpuSingle::PeriodicMakeList(unsigned n,unsigned pini,bool stable,unsigned nmax,tdouble3 perinc,const tdouble3 *pos,const word *code,unsigned *listp)const{
  unsigned count=0;
  if(n){
    //-Initialize size of list lsph to zero / Inicializa tamaño de lista lspg a cero.
    listp[nmax]=0;
    for(unsigned p=0;p<n;p++){
      const unsigned p2=p+pini;
      //-Keep normal or periodic particles / Se queda con particulas normales o periodicas.
      if(CODE_GetSpecialValue(code[p2])<=CODE_PERIODIC){
        //-Get particle position / Obtiene posicion de particula.
        const tdouble3 ps=pos[p2];
        tdouble3 ps2=ps+perinc;
        if(Map_PosMin<=ps2 && ps2<Map_PosMax){
          unsigned cp=listp[nmax]; listp[nmax]++; if(cp<nmax)listp[cp]=p2;
        }
        ps2=ps-perinc;
        if(Map_PosMin<=ps2 && ps2<Map_PosMax){
          unsigned cp=listp[nmax]; listp[nmax]++; if(cp<nmax)listp[cp]=(p2|0x80000000);
        }
      }
    }
    count=listp[nmax];
    //-Reorder list if it is valid and stability is activated / Reordena lista si es valida y stable esta activado.
    if(stable && count && count<=nmax){
      //-Don't make mistaje because at the moment the list is not created using OpenMP / No hace falta porque de momento no se crea la lista usando OpenMP.
    }
  }
  return(count);
}

//==============================================================================
/// (ES):
/// Duplica la posicion de la particula indicada aplicandole un desplazamiento.
/// Las particulas duplicadas se considera que siempre son validas y estan dentro
/// del dominio.
/// Este kernel vale para single-cpu y multi-cpu porque los calculos se hacen 
/// a partir de domposmin.
/// Se controla que las coordendas de celda no sobrepasen el maximo.
/// (EN):
/// Duplicate the indicated particle position applying displacement.
/// Duplicate particles are considered to be always valid and are inside
/// of the domain.
/// This kernel works for single-cpu & multi-cpu because the computations are done  
/// starting from domposmin.
/// It is controlled that the coordinates of the cell do not exceed the maximum.
//==============================================================================
void JSphCpuSingle::PeriodicDuplicatePos(unsigned pnew,unsigned pcopy,bool inverse,double dx,double dy,double dz,tuint3 cellmax,tdouble3 *pos,unsigned *dcell)const{
  //-Get pos of particle to be duplicated / Obtiene pos de particula a duplicar.
  tdouble3 ps=pos[pcopy];
  //-Apply displacement / Aplica desplazamiento.
  ps.x+=(inverse? -dx: dx);
  ps.y+=(inverse? -dy: dy);
  ps.z+=(inverse? -dz: dz);
  //-Calculate coordinates of cell inside of domain / Calcula coordendas de celda dentro de dominio.
  unsigned cx=unsigned((ps.x-DomPosMin.x)/Scell);
  unsigned cy=unsigned((ps.y-DomPosMin.y)/Scell);
  unsigned cz=unsigned((ps.z-DomPosMin.z)/Scell);
  //-Adjust coordinates of cell is they exceed maximum / Ajusta las coordendas de celda si sobrepasan el maximo.
  cx=(cx<=cellmax.x? cx: cellmax.x);
  cy=(cy<=cellmax.y? cy: cellmax.y);
  cz=(cz<=cellmax.z? cz: cellmax.z);
  //-Record position and cell of new particles /  Graba posicion y celda de nuevas particulas.
  pos[pnew]=ps;
  dcell[pnew]=PC__Cell(DomCellCode,cx,cy,cz);
}

//==============================================================================
/// (ES):
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-cpu y multi-cpu porque usa domposmin. 
/// (ES):
/// Create periodic particles starting from a list of the particles to duplicate.
/// Assume that all the particles are valid.
/// This kernel works for single-cpu & multi-cpu because it uses domposmin.
//==============================================================================
void JSphCpuSingle::PeriodicDuplicateSymplectic(unsigned n,unsigned pini,tuint3 cellmax,tdouble3 perinc,const unsigned *listp
  ,unsigned *idp,word *code,unsigned *dcell,tdouble3 *pos,tfloat4 *velrhop,tsymatrix3f *spstau,tdouble3 *pospre,tfloat4 *velrhoppre)const
{
  for(unsigned p=0;p<n;p++){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Adjust position and cell of new particle / Ajusta posicion y celda de nueva particula.
    PeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,pos,dcell);
    //-Copy the rest of the values / Copia el resto de datos.
    idp[pnew]=idp[pcopy];
    code[pnew]=CODE_SetPeriodic(code[pcopy]);
    velrhop[pnew]=velrhop[pcopy];
    if(pospre)pospre[pnew]=pospre[pcopy];
    if(velrhoppre)velrhoppre[pnew]=velrhoppre[pcopy];
  }
}

//==============================================================================
/// (ES):
/// Crea particulas duplicadas de condiciones periodicas.
/// Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
/// Las nuevas periodicas se situan a partir del Np de entrada, primero las NpbPer
/// de contorno y despues las NpfPer fluidas. El Np de salida contiene tambien las
/// nuevas periodicas.
/// (EN):
/// Create duplicate particles for periodic conditions.
/// Create new periodic particles and mark the old ones to be ignored.
/// New periodic particles are created from Np of the beginning, first the NpbPer
/// of the boundry and then the NpfPer fluid ones. The Np of the those leaving contains also the
/// new periodic ones.
//==============================================================================
/*void JSphCpuSingle::RunPeriodic(){
  const char met[]="RunPeriodic";
  TmcStart(Timers,TMC_SuPeriodic);
  //-Keep number of present periodic / Guarda numero de periodicas actuales.
  NpfPerM1=NpfPer;
  NpbPerM1=NpbPer;
  //-Mark present periodic particles to ignore / Marca periodicas actuales para ignorar.
  for(unsigned p=0;p<Np;p++){
    const word rcode=Codec[p];
    if(CODE_GetSpecialValue(rcode)==CODE_PERIODIC)Codec[p]=CODE_SetOutIgnore(rcode);
  }
  //-Create new periodic particles / Crea las nuevas periodicas.
  const unsigned npb0=Npb;
  const unsigned npf0=Np-Npb;
  const unsigned np0=Np;
  NpbPer=NpfPer=0;
  BoundChanged=true;
  for(unsigned ctype=0;ctype<2;ctype++){//-0:bound, 1:fluid+floating.
    //-Calculat range of particles to be examined (bound or fluid) / Calcula rango de particulas a examinar (bound o fluid).
    const unsigned pini=(ctype? npb0: 0);
    const unsigned num= (ctype? npf0: npb0);
    //-Search for periodic in each direction (X, Y, or Z) / Busca periodicas en cada eje (X, Y e Z).
    for(unsigned cper=0;cper<3;cper++)if((cper==0 && PeriActive&1) || (cper==1 && PeriActive&2) || (cper==2 && PeriActive&4)){
      tdouble3 perinc=(cper==0? PeriXinc: (cper==1? PeriYinc: PeriZinc));
      //-Primero busca en la lista de periodicas nuevas y despues en la lista inicial de particulas (necesario para periodicas en mas de un eje).
      //-First  search in the list of new periodic particles and then in the initial list of particles (this is needed for periodic particles in more than one direction).
      for(unsigned cblock=0;cblock<2;cblock++){//-0:periodicas nuevas, 1:particulas originales
        const unsigned nper=(ctype? NpfPer: NpbPer); //-Number of new periodic particles of type to be processed / Numero de periodicas nuevas del tipo a procesar.
        const unsigned pini2=(cblock? pini: Np-nper);
        const unsigned num2= (cblock? num:  nper);
        //-Repite la busqueda si la memoria disponible resulto insuficiente y hubo que aumentarla.
        //-Repeat the search if the resulting memory available is insufficient and it had to be increased.
        bool run=true;
        while(run && num2){
          //-Reserve memory to create list of periodic particles / Reserva memoria para crear lista de particulas periodicas.
          unsigned* listp=ArraysCpu->ReserveUint();
          unsigned nmax=CpuParticlesSize-1; //-Maximmum number of particles that fit in the list / Numero maximo de particulas que caben en la lista.
          //-Generate list of new periodic particles / Genera lista de nuevas periodicas.
          if(Np>=0x80000000)RunException(met,"The number of particles is too big.");//-Because the last bit is used to mark the direction in which a new periodic particle is created / Pq el ultimo bit se usa para marcar el sentido en que se crea la nueva periodica.
          unsigned count=PeriodicMakeList(num2,pini2,Stable,nmax,perinc,Posc,Codec,listp);
          //-Redimensiona memoria para particulas si no hay espacio suficiente y repite el proceso de busqueda.
          //-Redimension memory for particles if there is insufficient space and repeat the search process.
          if(count>nmax || count+Np>CpuParticlesSize){
            ArraysCpu->Free(listp); listp=NULL;
            TmcStop(Timers,TMC_SuPeriodic);
            ResizeParticlesSize(Np+count,PERIODIC_OVERMEMORYNP,false);
            TmcStart(Timers,TMC_SuPeriodic);
          }
          else{
            run=false;
            //-Crea nuevas particulas periodicas duplicando las particulas de la lista.
            //-Create new duplicate periodic particles in the list
            if(TStep==STEP_Symplectic){
              if((PosPrec || VelrhopPrec) && (!PosPrec || !VelrhopPrec))RunException(met,"Symplectic data is invalid.") ;
              PeriodicDuplicateSymplectic(count,Np,DomCells,perinc,listp,Idpc,Codec,Dcellc,Posc,Velrhopc,SpsTauc,PosPrec,VelrhopPrec);
            }

            //-Free the list and update the number of particles / Libera lista y actualiza numero de particulas.
            ArraysCpu->Free(listp); listp=NULL;
            Np+=count;
            //-Update number of new periodic particles / Actualiza numero de periodicas nuevas.
            if(!ctype)NpbPer+=count;
            else NpfPer+=count;
          }
        }
      }
    }
  }
  TmcStop(Timers,TMC_SuPeriodic);
}*/

//==============================================================================
/// Ejecuta divide de particulas en celdas.
/// Execute divide of particles in cells.
//==============================================================================
void JSphCpuSingle::RunCellDivide(bool updateperiodic){
  const char met[]="RunCellDivide";
  //-Create new periodic particles & mark the old ones to be ignored / Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
  //if(updateperiodic && PeriActive)RunPeriodic();
  
  //-Initial Divide / Inicia Divide.
  CellDivSingle->Divide(Npb,Np-Npb-NpbPer-NpfPer,NpbPer,NpfPer,BoundChanged,Dcellc,Codec,Idpc,Posc,Timers);
  
  //-Order particle data / Ordena datos de particulas
  TmcStart(Timers,TMC_NlSortData);
  CellDivSingle->SortArray(Idpc);
  CellDivSingle->SortArray(Codec);
  CellDivSingle->SortArray(Dcellc);
  CellDivSingle->SortArray(Posc);
  CellDivSingle->SortArray(Velrhopc);
  if(TStep==STEP_Symplectic && (PosPrec || VelrhopPrec)){//In reality, this is only necessary in divide for corrector, not in predictor??? / En realidad solo es necesario en el divide del corrector, no en el predictor???
    if(!PosPrec || !VelrhopPrec)RunException(met,"Symplectic data is invalid.") ;
    CellDivSingle->SortArray(PosPrec);
    CellDivSingle->SortArray(VelrhopPrec);
  }
  
  //-Collect divide data / Recupera datos del divide.
  Np=CellDivSingle->GetNpFinal();
  Npb=CellDivSingle->GetNpbFinal();
  NpbOk=Npb-CellDivSingle->GetNpbIgnore();
  //-Collect position of floating particles / Recupera posiciones de floatings.
  if(CaseNfloat)CalcRidp(PeriActive!=0,Np-Npb,Npb,CaseNpb,CaseNpb+CaseNfloat,Codec,Idpc,FtRidp);
  TmcStop(Timers,TMC_NlSortData);

  //-Gestion de particulas excluidas (solo fluid pq si alguna bound es excluida se genera excepcion en Divide()).
  //-Control of excluded particles (only fluid because if some bound is excluded ti generates an exception in Divide()).
  TmcStart(Timers,TMC_NlOutCheck);
  unsigned npfout=CellDivSingle->GetNpOut();
  if(npfout){
    unsigned* idp=ArraysCpu->ReserveUint();
    tdouble3* pos=ArraysCpu->ReserveDouble3();
    tfloat3* vel=ArraysCpu->ReserveFloat3();
    float* rhop=ArraysCpu->ReserveFloat();
    unsigned num=GetParticlesData(npfout,Np,true,false,idp,pos,vel,rhop,NULL);
    AddParticlesOut(npfout,idp,pos,vel,rhop,CellDivSingle->GetNpfOutRhop(),CellDivSingle->GetNpfOutMove());
    ArraysCpu->Free(idp);
    ArraysCpu->Free(pos);
    ArraysCpu->Free(vel);
    ArraysCpu->Free(rhop);
  }
  TmcStop(Timers,TMC_NlOutCheck);
  BoundChanged=false;
}

//------------------------------------------------------------------------------
/// Devuelve limites de celdas para interaccion.
/// Return cell limits for interaction.
//------------------------------------------------------------------------------
void JSphCpuSingle::GetInteractionCells(unsigned rcell
  ,int hdiv,const tint4 &nc,const tint3 &cellzero
  ,int &cxini,int &cxfin,int &yini,int &yfin,int &zini,int &zfin)const
{
  //-Get interaction limits / Obtiene limites de interaccion
  const int cx=PC__Cellx(DomCellCode,rcell)-cellzero.x;
  const int cy=PC__Celly(DomCellCode,rcell)-cellzero.y;
  const int cz=PC__Cellz(DomCellCode,rcell)-cellzero.z;
  //-code for hdiv 1 or 2 but not zero / Codigo para hdiv 1 o 2 pero no cero.
  cxini=cx-min(cx,hdiv);
  cxfin=cx+min(nc.x-cx-1,hdiv)+1;
  yini=cy-min(cy,hdiv);
  yfin=cy+min(nc.y-cy-1,hdiv)+1;
  zini=cz-min(cz,hdiv);
  zfin=cz+min(nc.z-cz-1,hdiv)+1;
}

//==============================================================================
/// Interaccion para el calculo de fuerzas.
/// Interaction to calculate forces.
//==============================================================================
void JSphCpuSingle::Interaction_Forces(TpInter tinter){
  const char met[]="Interaction_Forces";	
  	
  TmcStart(Timers,TMC_CfForces);

  //-Interaction of Fluid-Fluid/Bound & Bound-Fluid (forces and DEM) / Interaccion Fluid-Fluid/Bound & Bound-Fluid (forces and DEM).
  float viscdt=0;
  if(Psimple)JSphCpu::InteractionSimple_Forces(tinter,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellc,PsPosc,Velrhopc,Idpc,dWxCorr,dWzCorr,Codec,Acec);
  else JSphCpu::Interaction_Forces(tinter,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellc,Posc,Velrhopc,Idpc,dWxCorr,dWzCorr,Codec,Acec);

  //-For 2-D simulations zero the 2nd component / Para simulaciones 2D anula siempre la 2º componente
  if(Simulate2D)for(unsigned p=Npb;p<Np;p++)Acec[p].y=0;

  //-Add Delta-SPH correction to Arg[] / Añade correccion de Delta-SPH a Arg[].
  /*if(Deltac){
    const int ini=int(Npb),fin=int(Np),npf=int(Np-Npb);
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (static) if(npf>LIMIT_COMPUTELIGHT_OMP)
    #endif
    for(int p=ini;p<fin;p++)if(Deltac[p]!=FLT_MAX)Arc[p]+=Deltac[p];
  }*/

  //-Calculates maximum value of ViscDt.
  ViscDtMax=viscdt;
  //-Calculates maximum value of Ace.
  AceMax=ComputeAceMax();	
  TmcStop(Timers,TMC_CfForces);
}

//==============================================================================
/// (ES):
/// Devuelve valor maximo de (ace.x^2 + ace.y^2 + ace.z^2) a partir de Acec[].
/// (ES):
/// Return max value of (ace.x^2 + ace.y^2 + ace.z^2) starting from Acec[].
/// The use of OpenMP here is not efficient.
//==============================================================================
double JSphCpuSingle::ComputeAceMax(){
  float acemax=0;
  const int ini=int(Npb),fin=int(Np),npf=int(Np-Npb);
  if(!PeriActive){//-Without periodic conditions / Sin condiciones periodicas.
    for(int p=ini;p<fin;p++){
      const float ace=Acec[p].x*Acec[p].x+Acec[p].y*Acec[p].y+Acec[p].z*Acec[p].z;
      acemax=max(acemax,ace);
    }
  }
  else{//-With periodic conditions ignore periodic particles / Con condiciones periodicas ignora las particulas periodicas.
    for(int p=ini;p<fin;p++)if(CODE_GetSpecialValue(Codec[p])==CODE_NORMAL){
      const float ace=Acec[p].x*Acec[p].x+Acec[p].y*Acec[p].y+Acec[p].z*Acec[p].z;
      acemax=max(acemax,ace);
    }
  }
  return(sqrt(double(acemax)));
}

void JSphCpuSingle::PreparePosSimple(){
  //-Prepare values for interaction  Pos-Simpe / Prepara datos para interaccion Pos-Simple.
  PsPosc=ArraysCpu->ReserveFloat3();
  const int np=int(Np);
  /*#ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_PREINTERACTION_OMP)
  #endif*/
  for(int p=0;p<np;p++){ PsPosc[p]=ToTFloat3(Posc[p]); }
}
//==============================================================================
/// (ES):
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Symplectic.
/// (ES):
/// Perform interactions and updates of particles according to forces 
/// calculated in the interaction using Symplectic.
//==============================================================================
double JSphCpuSingle::ComputeStep_Sym(){
  const double dt=DtPre;
  //-Predictor
  //-----------
  PreInteraction_Forces(INTER_Forces);
  RunCellDivide(true);
  if(Psimple)PreparePosSimple();
  if(TSlipCond)BoundaryVelocity(TSlipCond);
  Interaction_Forces(INTER_Forces);      //-Interaction / Interaccion
  if(TSlipCond)memset(Velrhopc,0,sizeof(tfloat4)*Npb);
  //const double ddt_p=DtVariable(false);   //-Calculate dt of predictor step / Calcula dt del predictor
  //if(TShifting)RunShifting(dt*.5);        //-Shifting
  ComputeSymplecticPre(dt);               //-Apply Symplectic-Predictor to particles / Aplica Symplectic-Predictor a las particulas
  //if(CaseNfloat)RunFloating(dt*.5,true);  //-Control of floating bodies / Gestion de floating bodies
  PosInteraction_Forces(INTER_Forces);          //-Free memory used for interaction / Libera memoria de interaccion
  //-Pressure Poisson equation
  //-----------
  KernelCorrection();
  SolvePPE(dt); //-Solve pressure Poisson equation
  //-Corrector
  //-----------
  //DemDtForce=dt;                          //(DEM)
  PreInteraction_Forces(INTER_ForcesCorr);
  Interaction_Forces(INTER_ForcesCorr);   //Interaction / Interaccion
  //const double ddt_c=DtVariable(true);    //-Calculate dt of corrector step / Calcula dt del corrector
  ComputeSymplecticCorr(dt);              //-Apply Symplectic-Corrector to particles / Aplica Symplectic-Corrector a las particulas
  //if(CaseNfloat)RunFloating(dt,false);    //-Control of floating bodies / Gestion de floating bodies
  PosInteraction_Forces(INTER_ForcesCorr);             //-Free memory used for interaction / Libera memoria de interaccion
  if(TShifting)RunShifting(dt);           //-Shifting
  // DtPre=min(ddt_p,ddt_c);                 //-Calcula el dt para el siguiente ComputeStep
  return(dt);
}

//==============================================================================
/// Calcula distancia entre pariculas floatin y centro segun condiciones periodicas.
/// Calculate distance between floating particles & centre according to periodic conditions.
//==============================================================================
tfloat3 JSphCpuSingle::FtPeriodicDist(const tdouble3 &pos,const tdouble3 &center,float radius)const{
  tdouble3 distd=(pos-center);
  if(PeriX && fabs(distd.x)>radius){
    if(distd.x>0)distd=distd+PeriXinc;
    else distd=distd-PeriXinc;
  }
  if(PeriY && fabs(distd.y)>radius){
    if(distd.y>0)distd=distd+PeriYinc;
    else distd=distd-PeriYinc;
  }
  if(PeriZ && fabs(distd.z)>radius){
    if(distd.z>0)distd=distd+PeriZinc;
    else distd=distd-PeriZinc;
  }
  return(ToTFloat3(distd));
}

//==============================================================================
/// Calculate forces around floating object particles / Calcula fuerzas sobre floatings.
//==============================================================================
void JSphCpuSingle::FtCalcForces(StFtoForces *ftoforces)const{
  const int ftcount=int(FtCount);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int cf=0;cf<ftcount;cf++){
    const StFloatingData fobj=FtObjs[cf];
    const unsigned fpini=fobj.begin-CaseNpb;
    const unsigned fpfin=fpini+fobj.count;
    const float fradius=fobj.radius;
    const tdouble3 fcenter=fobj.center;
    const float fmassp=fobj.massp;
    //-Computes traslational and rotational velocities.
    tfloat3 face=TFloat3(0);
    tfloat3 fomegavel=TFloat3(0);
    tmatrix3f inert=TMatrix3f(0,0,0,0,0,0,0,0,0);
    //-Calculate summation: face, fomegavel & inert / Calcula sumatorios: face, fomegavel y inert.
    for(unsigned fp=fpini;fp<fpfin;fp++){
      int p=FtRidp[fp];
      //-Ace is initialised with the value of the gravity for all particles.
      float acex=Acec[p].x-Gravity.x,acey=Acec[p].y-Gravity.y,acez=Acec[p].z-Gravity.z;
      face.x+=acex; face.y+=acey; face.z+=acez;
      tfloat3 dist=(PeriActive? FtPeriodicDist(Posc[p],fcenter,fradius): ToTFloat3(Posc[p]-fcenter)); 
      fomegavel.x+= acez*dist.y - acey*dist.z;
      fomegavel.y+= acex*dist.z - acez*dist.x;
      fomegavel.z+= acey*dist.x - acex*dist.y;
      //inertia tensor
      inert.a11+=(float)  (dist.y*dist.y+dist.z*dist.z)*fmassp;
      inert.a12+=(float) -(dist.x*dist.y)*fmassp;
      inert.a13+=(float) -(dist.x*dist.z)*fmassp;
      inert.a21+=(float) -(dist.x*dist.y)*fmassp;
      inert.a22+=(float)  (dist.x*dist.x+dist.z*dist.z)*fmassp;
      inert.a23+=(float) -(dist.y*dist.z)*fmassp;
      inert.a31+=(float) -(dist.x*dist.z)*fmassp;
      inert.a32+=(float) -(dist.y*dist.z)*fmassp;
      inert.a33+=(float)  (dist.x*dist.x+dist.y*dist.y)*fmassp;
    }
    //-Calculates the inverse of the intertia matrix to compute the I^-1 * L= W
    tmatrix3f invinert=TMatrix3f(0,0,0,0,0,0,0,0,0);
    const float detiner=(inert.a11*inert.a22*inert.a33+inert.a12*inert.a23*inert.a31+inert.a21*inert.a32*inert.a13-(inert.a31*inert.a22*inert.a13+inert.a21*inert.a12*inert.a33+inert.a23*inert.a32*inert.a11));
    if(detiner){
      invinert.a11= (inert.a22*inert.a33-inert.a23*inert.a32)/detiner;
      invinert.a12=-(inert.a12*inert.a33-inert.a13*inert.a32)/detiner;
      invinert.a13= (inert.a12*inert.a23-inert.a13*inert.a22)/detiner;
      invinert.a21=-(inert.a21*inert.a33-inert.a23*inert.a31)/detiner;
      invinert.a22= (inert.a11*inert.a33-inert.a13*inert.a31)/detiner;
      invinert.a23=-(inert.a11*inert.a23-inert.a13*inert.a21)/detiner;
      invinert.a31= (inert.a21*inert.a32-inert.a22*inert.a31)/detiner;
      invinert.a32=-(inert.a11*inert.a32-inert.a12*inert.a31)/detiner;
      invinert.a33= (inert.a11*inert.a22-inert.a12*inert.a21)/detiner;
    }
    //-Calculate omega starting from fomegavel & invinert / Calcula omega a partir de fomegavel y invinert.
    {
      tfloat3 omega;
      omega.x=(fomegavel.x*invinert.a11+fomegavel.y*invinert.a12+fomegavel.z*invinert.a13);
      omega.y=(fomegavel.x*invinert.a21+fomegavel.y*invinert.a22+fomegavel.z*invinert.a23);
      omega.z=(fomegavel.x*invinert.a31+fomegavel.y*invinert.a32+fomegavel.z*invinert.a33);
      fomegavel=omega;
    }
    //-Keep result in ftoforces[] / Guarda resultados en ftoforces[].
    ftoforces[cf].face=face;
    ftoforces[cf].fomegavel=fomegavel;
  }
}

//==============================================================================
/// Process floating objects / Procesa floating objects.
//==============================================================================
void JSphCpuSingle::RunFloating(double dt,bool predictor){
  const char met[]="RunFloating";
  if(TimeStep>=FtPause){//-This is used because if FtPause=0 in symplectic-predictor, code would not enter clause / -Se usa >= pq si FtPause es cero en symplectic-predictor no entraria.
    TmcStart(Timers,TMC_SuFloating);
    //-Calculate forces around floating objects / Calcula fuerzas sobre floatings.
    FtCalcForces(FtoForces);

    //-Apply movement around floating objects / Aplica movimiento sobre floatings.
    const int ftcount=int(FtCount);
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (guided)
    #endif
    for(int cf=0;cf<ftcount;cf++){
      //-Get Floating object values / Obtiene datos de floating.
      const StFloatingData fobj=FtObjs[cf];
      //-Compute force face / Calculo de face.
      const float fmass=fobj.mass;
      tfloat3 face=FtoForces[cf].face;
      face.x=(face.x+fmass*Gravity.x)/fmass;
      face.y=(face.y+fmass*Gravity.y)/fmass;
      face.z=(face.z+fmass*Gravity.z)/fmass;
      //-Compute fomega / Calculo de fomega.
      tfloat3 fomega=fobj.fomega;
      {
        const tfloat3 omega=FtoForces[cf].fomegavel;
        fomega.x=float(dt*omega.x+fomega.x);
        fomega.y=float(dt*omega.y+fomega.y);
        fomega.z=float(dt*omega.z+fomega.z);
      }
      tfloat3 fvel=fobj.fvel;
      //-Zero components for 2-D simulation / Anula componentes para 2D.
      if(Simulate2D){ face.y=0; fomega.x=0; fomega.z=0; fvel.y=0; }
      //-Compute fcenter / Calculo de fcenter.
      tdouble3 fcenter=fobj.center;
      fcenter.x+=dt*fvel.x;
      fcenter.y+=dt*fvel.y;
      fcenter.z+=dt*fvel.z;
      //-Compute fvel / Calculo de fvel.
      fvel.x=float(dt*face.x+fvel.x);
      fvel.y=float(dt*face.y+fvel.y);
      fvel.z=float(dt*face.z+fvel.z);

      //-Updates floating particles.
      const float fradius=fobj.radius;
      const unsigned fpini=fobj.begin-CaseNpb;
      const unsigned fpfin=fpini+fobj.count;
      for(unsigned fp=fpini;fp<fpfin;fp++){
        const int p=FtRidp[fp];
        if(p!=UINT_MAX){
          tfloat4 *velrhop=Velrhopc+p;
          //-Compute and record position displacement / Calcula y graba desplazamiento de posicion.
          const double dx=dt*double(velrhop->x);
          const double dy=dt*double(velrhop->y);
          const double dz=dt*double(velrhop->z);
          UpdatePos(Posc[p],dx,dy,dz,false,p,Posc,Dcellc,Codec);
          //-Compute and record new velocity / Calcula y graba nueva velocidad.
          tfloat3 dist=(PeriActive? FtPeriodicDist(Posc[p],fcenter,fradius): ToTFloat3(Posc[p]-fcenter)); 
          velrhop->x=fvel.x+(fomega.y*dist.z-fomega.z*dist.y);
          velrhop->y=fvel.y+(fomega.z*dist.x-fomega.x*dist.z);
          velrhop->z=fvel.z+(fomega.x*dist.y-fomega.y*dist.x);
        }
      }

      //-Stores floating data.
      if(!predictor){
        const tdouble3 centerold=FtObjs[cf].center;
        FtObjs[cf].center=(PeriActive? UpdatePeriodicPos(fcenter): fcenter);
        FtObjs[cf].fvel=fvel;
        FtObjs[cf].fomega=fomega;
      }
    }
    TmcStop(Timers,TMC_SuFloating);
  }
}

//==============================================================================
/// Inicia proceso de simulacion.
/// Initial processing of simulation.
//==============================================================================
void JSphCpuSingle::Run(std::string appname,JCfgRun *cfg,JLog2 *log){
  const char* met="Run";
  if(!cfg||!log)return;
  AppName=appname; Log=log;

  //-Configure timers / Configura timers
  //-------------------
  TmcCreation(Timers,cfg->SvTimers);
  TmcStart(Timers,TMC_Init);
  if(cfg->SvTimersStep>0){
    TimersStep=new JTimersStep(cfg->DirOut,cfg->SvTimersStep,0,0);
    for(unsigned ct=0;ct<TimerGetCount();ct++)if(TimerIsActive(ct))TimersStep->AddTimer(TimerGetName(ct),TimerGetPtrValue(ct));
  }

  //-Load parameters and values of input / Carga de parametros y datos de entrada
  //-----------------------------------------
  LoadConfig(cfg);
  LoadCaseParticles();
  ConfigConstants(Simulate2D);
  ConfigDomain();
  ConfigRunMode(cfg);

  //-Initialization of execution variables / Inicializacion de variables de ejecucion
  //-------------------------------------------
  InitRun();
  UpdateMaxValues();
  PrintAllocMemory(GetAllocMemoryCpu());
  SaveData(); 
  TmcResetValues(Timers);
  TmcStop(Timers,TMC_Init);
  PartNstep=-1; Part++;

  //-Main Loop / Bucle principal
  //------------------
  bool partoutstop=false;
  TimerSim.Start();
  TimerPart.Start();
  Log->Print(string("\n[Initialising simulation (")+RunCode+")  "+fun::GetDateTime()+"]");
  PrintHeadPart();

  //-finding dummy particle relations to wall particles
  count=1;
  FindIrelation(); 
  while(TimeStep<TimeMax){
    clock_t start = clock(); 
    //if(ViscoTime)Visco=ViscoTime->GetVisco(float(TimeStep));
    double stepdt=ComputeStep_Sym();
    if(PartDtMin>stepdt)PartDtMin=stepdt; if(PartDtMax<stepdt)PartDtMax=stepdt;
    if(CaseNmoving)RunMotion(stepdt);
    //RunCellDivide(true);
    TimeStep+=stepdt;
    partoutstop=(Np<NpMinimum || !Np);
    if((TimeStep-TimeStepIni)-TimePart*((Part-PartIni)-1)>=TimePart || partoutstop){
      if(partoutstop){
        Log->Print("\n**** Particles OUT limit reached...\n");
        TimeMax=TimeStep;
      }
      SaveData();
      Part++;
      PartNstep=Nstep;
      TimeStepM1=TimeStep;
      TimerPart.Start();
    }
    UpdateMaxValues();
    Nstep++;
    if(TimersStep&&TimersStep->Check(float(TimeStep)))SaveTimersStep(Np,Npb,NpbOk,CellDivSingle->GetNct());
    //if(Nstep>=3)break;
    clock_t stop = clock();   
    double dif = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    cout<<"Timestep Time = " << dif << "ms\n";
  }
  TimerSim.Stop(); TimerTot.Stop();

  //-End of Simulation / Fin de simulacion
  //--------------------
  FinishRun(partoutstop);
}

//==============================================================================
/// Genera los ficheros de salida de datos
/// Generate out file data
//==============================================================================
void JSphCpuSingle::SaveData(){
  const bool save=(SvData!=SDAT_None&&SvData!=SDAT_Info);
  const unsigned npsave=Np-NpbPer-NpfPer; //-Keep the periodic particles if they exist / Resta las periodicas si las hubiera.
  TmcStart(Timers,TMC_SuSavePart);
  //-Collect particle values in original order / Recupera datos de particulas en orden original.
  unsigned *idp=NULL;
  tdouble3 *pos=NULL;
  tfloat3 *vel=NULL;
  float *rhop=NULL;
  if(save){
    //-Assign memory and collect particle values / Asigna memoria y recupera datos de las particulas.
    idp=ArraysCpu->ReserveUint();
    pos=ArraysCpu->ReserveDouble3();
    vel=ArraysCpu->ReserveFloat3();
    rhop=ArraysCpu->ReserveFloat();
    unsigned npnormal=GetParticlesData(Np,0,true,PeriActive!=0,idp,pos,vel,rhop,NULL);
    if(npnormal!=npsave)RunException("SaveData","The number of particles is invalid.");
  }
  //-Gather additional information / Reune informacion adicional.
  StInfoPartPlus infoplus;
  memset(&infoplus,0,sizeof(StInfoPartPlus));
  if(SvData&SDAT_Info){
    infoplus.nct=CellDivSingle->GetNct();
    infoplus.npbin=NpbOk;
    infoplus.npbout=Npb-NpbOk;
    infoplus.npf=Np-Npb;
    infoplus.npbper=NpbPer;
    infoplus.npfper=NpfPer;
    infoplus.memorycpualloc=this->GetAllocMemoryCpu();
    infoplus.gpudata=false;
    TimerSim.Stop();
    infoplus.timesim=TimerSim.GetElapsedTimeD()/1000.;
  }

  //-Record particle values / Graba datos de particulas.
  const tdouble3 vdom[2]={OrderDecode(CellDivSingle->GetDomainLimits(true)),OrderDecode(CellDivSingle->GetDomainLimits(false))};
  JSph::SaveData(npsave,idp,pos,vel,rhop,1,vdom,&infoplus);
  //-Libera memoria para datos de particulas.
  ArraysCpu->Free(idp);
  ArraysCpu->Free(pos);
  ArraysCpu->Free(vel);
  ArraysCpu->Free(rhop);
  //-Record execution information / Graba informacion de ejecucion.
  if(TimersStep)TimersStep->SaveData();
  TmcStop(Timers,TMC_SuSavePart);
}

//==============================================================================
/// Muestra y graba resumen final de ejecucion.
/// Show and record final overview of execution.
//==============================================================================
void JSphCpuSingle::FinishRun(bool stop){
  float tsim=TimerSim.GetElapsedTimeF()/1000.f,ttot=TimerTot.GetElapsedTimeF()/1000.f;
  if(TimersStep)TimersStep->SaveData();
  JSph::ShowResume(stop,tsim,ttot,true,"");
  string hinfo=";RunMode",dinfo=string(";")+RunMode;
  if(SvTimers){
    ShowTimers();
    GetTimersInfo(hinfo,dinfo);
  }
  Log->Print(" ");
  if(SvRes)SaveRes(tsim,ttot,hinfo,dinfo);
}

void JSphCpuSingle::FindIrelation(){
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  tuint3 ncells=CellDivSingle->GetNcells();
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const unsigned cellfluid=nc.w*nc.z+1;
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
  const unsigned *begincell = CellDivSingle->GetBeginCell();
  JSphCpu::FindIrelation(Npb,0,Posc,Idpc,Irelationc,Codec); 
}

void JSphCpuSingle::BoundaryVelocity(TpSlipCond TSlipCond){
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  tuint3 ncells=CellDivSingle->GetNcells();
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const unsigned cellfluid=nc.w*nc.z+1;
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
  const unsigned *begincell = CellDivSingle->GetBeginCell();
  JSphCpu::Boundary_Velocity(TSlipCond,Psimple,NpbOk,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,Codec);
}

void JSphCpuSingle::KernelCorrection(){ 
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  tuint3 ncells=CellDivSingle->GetNcells();
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const unsigned cellfluid=nc.w*nc.z+1;
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
  const unsigned *begincell = CellDivSingle->GetBeginCell();

  const unsigned np=Np;
  const unsigned npbok=NpbOk;
  const unsigned npb=Npb;
  const unsigned npf=np-npb;

  dWxCorr=ArraysCpu->ReserveDouble3();
  dWzCorr=ArraysCpu->ReserveDouble3();

  memset(dWxCorr,0,sizeof(tdouble3)*np);						
  memset(dWzCorr,0,sizeof(tdouble3)*np);								

  JSphCpu::KernelCorrection(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,dWxCorr,dWzCorr); //-Fluid-Fluid
  JSphCpu::KernelCorrection(Psimple,npf,npb,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,dWxCorr,dWzCorr); //-Fluid-Bound
  JSphCpu::KernelCorrection(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,dWxCorr,dWzCorr); //-Bound-Fluid
  JSphCpu::KernelCorrection(Psimple,npbok,0,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,dWxCorr,dWzCorr); //-Bound-Bound
  JSphCpu::InverseCorrection(npf,npb,dWxCorr,dWzCorr);
  JSphCpu::InverseCorrection(npbok,0,dWxCorr,dWzCorr);
}
//==============================================================================
/// PPE Solver
//==============================================================================
#include <fstream>
#include <sstream>
#include <iomanip>
void JSphCpuSingle::SolvePPE(double dt){ 
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  tuint3 ncells=CellDivSingle->GetNcells();
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const unsigned cellfluid=nc.w*nc.z+1;
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
  const unsigned *begincell = CellDivSingle->GetBeginCell();

  const unsigned np=Np;
  const unsigned npb=Npb;
  const unsigned npbok=NpbOk;
  const unsigned npf=np-npb;
  unsigned PPEDim=0;
  unsigned PPEDimDummy=0;

  //Matrix Order and Free Surface
  POrder=ArraysCpu->ReserveUint(); memset(POrder,np,sizeof(unsigned)*np);
  Divr=ArraysCpu->ReserveFloat(); memset(Divr,0,sizeof(float)*np);

  FreeSurfaceFind(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,Codec); //-Fluid-Fluid
  FreeSurfaceFind(Psimple,npf,npb,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,Codec); //-Fluid-Bound
  FreeSurfaceFind(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,Codec); //-Bound-Fluid
  FreeSurfaceFind(Psimple,npbok,0,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,Codec); //-Bound-Bound
  MatrixOrder(np,0,POrder,Idpc,Irelationc,Codec,PPEDim);
  b.resize(PPEDim,0);
  rowInd.resize(PPEDim+1,0);
  unsigned Nnz=0;
  //Organising storage for parallelism
  MatrixStorage(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);//-Fluid-Fluid
  MatrixStorage(Psimple,npf,npb,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);//-Fluid-Bound
  MatrixStorage(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);//-Bound-Fluid
  MatrixStorage(Psimple,npbok,0,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);//-Bound-Bound
  MatrixASetup(PPEDim,Nnz,rowInd); 
  stencil.resize((Nnz-PPEDim),1);
  //FindStencilFluid(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,stencil,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);
  //FindStencilFluid(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Divr,stencil,rowInd,POrder,Idpc,Codec,PPEDim,FreeSurface);
  //RHS
  PopulateMatrixB(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,dWxCorr,dWzCorr,b,POrder,Idpc,dt,PPEDim,Divr,FreeSurface,stencil,rowInd); //-Fluid-Fluid
  PopulateMatrixB(Psimple,npf,npb,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,dWxCorr,dWzCorr,b,POrder,Idpc,dt,PPEDim,Divr,FreeSurface,stencil,rowInd); //-Fluid-Bound
  PopulateMatrixB(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,dWxCorr,dWzCorr,b,POrder,Idpc,dt,PPEDim,Divr,FreeSurface,stencil,rowInd); //-Bound-Fluid
  PopulateMatrixB(Psimple,npbok,0,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,dWxCorr,dWzCorr,b,POrder,Idpc,dt,PPEDim,Divr,FreeSurface,stencil,rowInd); //-Bound-Bound
  colInd.resize(Nnz,PPEDim); 
  a.resize(Nnz,0);
  //LHS
  PopulateMatrixAFluid(Psimple,npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,Divr,a,rowInd,colInd,POrder,Irelationc,b,Idpc,Codec,PPEDim,FreeSurface,Gravity,RhopZero,stencil);//-Fluid-Fluid
  //PopulateMatrixAInteractBound(Psimple,npf,npb,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,Divr,a,rowInd,colInd,b,POrder,Idpc,Codec,Irelationc,PPEDim,FreeSurface);//-Fluid-Bound
  PopulateMatrixABoundary(Psimple,npbok,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,Divr,a,rowInd,colInd,POrder,Irelationc,b,Idpc,Codec,PPEDim,FreeSurface,Gravity,RhopZero); //-Fluid-Fluid
  //PopulateMatrixAInteractBound(Psimple,npbok,0,nc,hdiv,0,begincell,cellzero,Dcellc,Posc,PsPosc,Velrhopc,Divr,a,rowInd,colInd,b,POrder,Idpc,Codec,Irelationc,PPEDim,FreeSurface); //-Fluid-Bound
  FreeSurfaceMark(npf,npb,Divr,a,b,rowInd,POrder,Idpc,Codec,PPEDim);
  FreeSurfaceMark(npbok,0,Divr,a,b,rowInd,POrder,Idpc,Codec,PPEDim);
  // allocate vectors
  x.resize(PPEDim,0);
  
 
   /* ofstream FileOutput;
    string TimeFile;

    ostringstream TimeNum;
    TimeNum << count;
    ostringstream FileNum;
    FileNum << count;

    TimeFile =  "CPU Fluid Properties_" + FileNum.str() + ", T = " + TimeNum.str() + ".txt";

    FileOutput.open(TimeFile.c_str());

  for(int i=0;i<npbok;i++){
    if(POrder[i]!=np){
      FileOutput << fixed << setprecision(19) << "particle "<< Idpc[i] << "\t Order " << POrder[i] << "\t b " << b[POrder[i]] << "\n";
      for(int j=rowInd[POrder[i]];j<rowInd[POrder[i]+1];j++) FileOutput << fixed << setprecision(16) << j << "\t" << a[j] << "\t" << colInd[j] << "\n";
    }
  }

  for(int i=npb;i<np;i++){
    if(POrder[i]!=np){
      FileOutput << fixed << setprecision(20) <<"particle "<< Idpc[i] << "\t Order " << POrder[i] << "\t b " << b[POrder[i]] << "\n";
      for(int j=rowInd[POrder[i]];j<rowInd[POrder[i]+1];j++) FileOutput << fixed << setprecision(16) << j << "\t" << a[j] << "\t" << colInd[j] << "\n";
    }
  }
  FileOutput.close();  
  
  count++;
  //solvers
#ifndef _WITHGPU
  solveVienna(TPrecond,TAMGInter,Tolerance,Iterations,StrongConnection,JacobiWeight,Presmooth,Postsmooth,CoarseCutoff,a,b,x,rowInd,colInd,PPEDim,Nnz); 
#endif
  /*TimeFile =  "Pressure" + FileNum.str() + ", T = " + TimeNum.str() + ".txt";

    FileOutput.open(TimeFile.c_str());

  for(int i=0;i<PPEDim;i++){
    FileOutput << fixed << setprecision(10) << i << "\t" << x[i] <<"\n";
  }

  FileOutput.close();

  count++; */
  PressureAssign(Psimple,np,0,Posc,PsPosc,Velrhopc,Idpc,Irelationc,POrder,x,Codec,npb,Divr,Gravity);
  //for(int i=0;i<Np;i++)Velrhopc[i].w=Divr[i];
  b.clear();
  a.clear();
  x.clear();
  rowInd.clear();
  colInd.clear();
  stencil.clear();
}

//==============================================================================
/// SHIFTING
//==============================================================================
void JSphCpuSingle::RunShifting(double dt){
  unsigned int np=Np;
  unsigned int npb=Npb;
  unsigned int npf=np-npb;
  //-Assign memory to variables Pre / Asigna memoria a variables Pre.
  PosPrec=ArraysCpu->ReserveDouble3();
  VelrhopPrec=ArraysCpu->ReserveFloat4();

  ShiftPosc=ArraysCpu->ReserveFloat3();
  Divr=ArraysCpu->ReserveFloat();
  memset(ShiftPosc,0,sizeof(tfloat3)*np);               //ShiftPosc[]=0
  memset(Divr,0,sizeof(float)*np);           //Divr[]=0   

  #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int i=0;i<int(Np);i++){
      PosPrec[i]=Posc[i];
      VelrhopPrec[i]=Velrhopc[i];
  }

  RunCellDivide(true);

  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  tuint3 ncells=CellDivSingle->GetNcells();
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const unsigned cellfluid=nc.w*nc.z+1;
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
  const unsigned *begincell = CellDivSingle->GetBeginCell();

  PreparePosSimple();

  JSphCpu::Interaction_Shifting(Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellc,PsPosc,Velrhopc,Idpc,Codec,ShiftPosc,Divr,TensileN,TensileR);

  JSphCpu::RunShifting(dt);

  Shift(dt);

  ArraysCpu->Free(PosPrec);      PosPrec=NULL;
  ArraysCpu->Free(PsPosc);       PsPosc=NULL;
  ArraysCpu->Free(ShiftPosc);    ShiftPosc=NULL;
  ArraysCpu->Free(Divr); Divr=NULL;
  ArraysCpu->Free(VelrhopPrec);  VelrhopPrec=NULL;
}