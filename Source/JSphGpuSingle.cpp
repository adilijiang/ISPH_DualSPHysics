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

#include "JSphGpuSingle.h"
#include "JCellDivGpuSingle.h"
#include "JArraysGpu.h"
#include "Functions.h"
#include "JXml.h"
#include "JSphMotion.h"
#include "JPartsLoad4.h"
#include "JSphVisco.h"
#include "JWaveGen.h"
#include "JSphGpu_ker.h"
#include "JPtxasInfo.h"
#include <time.h>

#include <iostream>

using namespace std;
//==============================================================================
/// Constructor.
//==============================================================================
JSphGpuSingle::JSphGpuSingle():JSphGpu(false){
  ClassName="JSphGpuSingle";
  CellDivSingle=NULL;
  PartsLoaded=NULL;
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphGpuSingle::~JSphGpuSingle(){
  delete CellDivSingle; CellDivSingle=NULL;
  delete PartsLoaded;   PartsLoaded=NULL;
}

//==============================================================================
/// Devuelve la memoria reservada en cpu.
/// Returns the memory allocated to the CPU.
//==============================================================================
llong JSphGpuSingle::GetAllocMemoryCpu()const{  
  llong s=JSphGpu::GetAllocMemoryCpu();
  //Reservada en otros objetos
  //Allocated in other objects
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryCpu();
  if(PartsLoaded)s+=PartsLoaded->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Devuelve la memoria reservada en gpu.
/// Returns the memory allocated to the GPU.
//==============================================================================
llong JSphGpuSingle::GetAllocMemoryGpu()const{  
  llong s=JSphGpu::GetAllocMemoryGpu();
  //Reservada en otros objetos
  //Allocated in other objects.
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryGpu();
  return(s);
}

//==============================================================================
/// Devuelve la memoria gpu reservada o usada para particulas.
/// Returns the gpu memory allocated or used for particles
//==============================================================================
llong JSphGpuSingle::GetMemoryGpuNp()const{
  llong s=JSphGpu::GetAllocMemoryGpu();
  //Reservada en otros objetos
  //Allocated in other objects
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryGpuNp();
  return(s);
}

//==============================================================================
/// Devuelve la memoria gpu reservada o usada para celdas.
/// Returns the gpu memory allocated or used for cells.
//==============================================================================
llong JSphGpuSingle::GetMemoryGpuNct()const{
  llong s=CellDivSingle->GetAllocMemoryGpuNct();
  return(CellDivSingle->GetAllocMemoryGpuNct());
}

//==============================================================================
/// Actualiza los valores maximos de memory, particles y cells.
/// Updates the maximum values of memory, particles and cells.
//==============================================================================
void JSphGpuSingle::UpdateMaxValues(){
  MaxParticles=max(MaxParticles,Np);
  if(CellDivSingle)MaxCells=max(MaxCells,CellDivSingle->GetNct());
  llong m=GetAllocMemoryCpu();
  MaxMemoryCpu=max(MaxMemoryCpu,m);
  m=GetAllocMemoryGpu();
  MaxMemoryGpu=max(MaxMemoryGpu,m);
}

//==============================================================================
/// Carga la configuracion de ejecucion.
/// Loads the configuration of the execution.
//==============================================================================
void JSphGpuSingle::LoadConfig(JCfgRun *cfg){
  const char met[]="LoadConfig";
  PtxasFile=cfg->PtxasFile;
  //-Carga configuracion basica general
  //-Loads general configuration
  JSph::LoadConfig(cfg);
  //-Checks compatibility of selected options.
  if(RenCorrection && UseDEM)RunException(met,"Ren correction is not implemented with Floatings-DEM.");
  Log->Print("**Special case configuration is loaded");
}

//==============================================================================
/// Carga particulas del caso a procesar.
/// Loads particles of the case to be processed.
//==============================================================================
void JSphGpuSingle::LoadCaseParticles(){
  Log->Print("Loading initial state of particles...");
  PartsLoaded=new JPartsLoad4;
  PartsLoaded->LoadParticles(DirCase,CaseName,PartBegin,PartBeginDir);
  PartsLoaded->CheckConfig(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,PeriX,PeriY,PeriZ);
  Log->Printf("Loaded particles: %u",PartsLoaded->GetCount());
  //-Recupera informacion de las particulas cargadas.
  //-Retrieves information of the loaded particles
  Simulate2D=PartsLoaded->GetSimulate2D();
  if(Simulate2D&&PeriY)RunException("LoadCaseParticles","Can not use periodic conditions in Y with 2D simulations");
  CasePosMin=PartsLoaded->GetCasePosMin();
  CasePosMax=PartsLoaded->GetCasePosMax();

  //-Calcula limites reales de la simulacion.
  //-Computes the current limits of the simulation.
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

  //-Configura limites de ejes periodicos
  //-Configures the limits of the periodic axis
  if(PeriX)PeriXinc.x=-MapRealSize.x;
  if(PeriY)PeriYinc.y=-MapRealSize.y;
  if(PeriZ)PeriZinc.z=-MapRealSize.z;
  //-Calcula limites de simulacion con bordes periodicos.
  //-Computes the limits of simulations with periodic edges.
  Map_PosMin=MapRealPosMin; Map_PosMax=MapRealPosMax;
  float dosh=float(H*2);
  if(PeriX){ Map_PosMin.x=Map_PosMin.x-dosh;  Map_PosMax.x=Map_PosMax.x+dosh; }
  if(PeriY){ Map_PosMin.y=Map_PosMin.y-dosh;  Map_PosMax.y=Map_PosMax.y+dosh; }
  if(PeriZ){ Map_PosMin.z=Map_PosMin.z-dosh;  Map_PosMax.z=Map_PosMax.z+dosh; }
  Map_Size=Map_PosMax-Map_PosMin;
}

//==============================================================================
/// Configuracion del dominio actual.
/// Configuration of the current domain.
//==============================================================================
void JSphGpuSingle::ConfigDomain(){
  const char* met="ConfigDomain";
  //-Calcula numero de particulas.
  //Computes the number of particles.
  Np=PartsLoaded->GetCount(); Npb=CaseNpb; NpbOk=Npb;
  //-Reserva memoria fija para moving y floating.
  //-Allocates memory for arrays with fixed size (motion and floating bodies)
  AllocGpuMemoryFixed();
  //-Reserva memoria en Gpu para particulas.
  //-Allocates GPU memory for particles.
  AllocGpuMemoryParticles(Np,0);
  //-Reserva memoria en Cpu.
  //-Allocates memory on the CPU.
  AllocCpuMemoryParticles(Np);

  //-Copia datos de particulas.
  //-Copies particle data.
  memcpy(AuxPos,PartsLoaded->GetPos(),sizeof(tdouble3)*Np);
  memcpy(Idp,PartsLoaded->GetIdp(),sizeof(unsigned)*Np);
  memcpy(Velrhop,PartsLoaded->GetVelRhop(),sizeof(tfloat4)*Np);

  //-Calcula radio de floatings.
  //-Computes radius of floating bodies.
  if(CaseNfloat && PeriActive!=0)CalcFloatingRadius(Np,AuxPos,Idp);

  //-Carga code de particulas.
  //-Loads Code of the particles.
  LoadCodeParticles(Np,Idp,Code);

  //-Libera memoria de PartsLoaded.
  //-Releases memory of PartsLoaded.
  delete PartsLoaded; PartsLoaded=NULL;
  //-Aplica configuracion de CellOrder.
  //-Applies configuration of CellOrder.
  ConfigCellOrder(CellOrder,Np,AuxPos,Velrhop);

  //-Configura division celdas.
  //-Configure cell division.
  ConfigCellDivision();
  //-Establece dominio de simulacion local dentro de Map_Cells y calcula DomCellCode.
  //-Sets local domain of the simulation within Map_Cells and computes DomCellCode.
  SelecDomain(TUint3(0,0,0),Map_Cells);
  //-Calcula celda inicial de particulas y comprueba si hay excluidas inesperadas.
  //-Computes inital cell of the particles and checks if there are unexpected excluded particles
  LoadDcellParticles(Np,Code,AuxPos,Dcell);

  //-Sube datos de particulas a la GPU.
  //-Uploads particle data on the GPU.
  ReserveBasicArraysGpu();
  for(unsigned p=0;p<Np;p++){ Posxy[p]=TDouble2(AuxPos[p].x,AuxPos[p].y); Posz[p]=AuxPos[p].z; }
  ParticlesDataUp(Np);
  //-Sube constantes a la GPU.
  //-Uploads constants on the GPU.
  ConstantDataUp();

  //-Crea objeto para divide en Gpu y selecciona un cellmode valido.
  //-Creates object for Celldiv on the GPU and selects a valid cellmode.
  CellDivSingle=new JCellDivGpuSingle(Stable,FtCount!=0,PeriActive,CellOrder,CellMode,Scell,Map_PosMin,Map_PosMax,Map_Cells,CaseNbound,CaseNfixed,CaseNpb,Log,DirOut);
  CellDivSingle->DefineDomain(DomCellCode,DomCelIni,DomCelFin,DomPosMin,DomPosMax);
  ConfigCellDiv((JCellDivGpu*)CellDivSingle);

  ConfigBlockSizes(false,PeriActive!=0);
  ConfigSaveData(0,1,"");

  //-Reordena particulas por celda.
  //-Reorders particles according to cells
  BoundChanged=true;
  RunCellDivide(true);
}

//==============================================================================
/// ES:
/// Redimensiona el espacio reservado para particulas en CPU y GPU midiendo el
/// tiempo consumido con TMG_SuResizeNp. Al terminar actualiza el divide.
///
/// EN:
/// Resizes the allocated space for particles on the CPU and the GPU measuring
/// the time spent with TMG_SuResizeNp. At the end updates the division.
//==============================================================================
void JSphGpuSingle::ResizeParticlesSize(unsigned newsize,float oversize,bool updatedivide){
  TmgStart(Timers,TMG_SuResizeNp);
  newsize+=(oversize>0? unsigned(oversize*newsize): 0);
  FreeCpuMemoryParticles();
  CellDivSingle->FreeMemoryGpu();
  ResizeGpuMemoryParticles(newsize);
  AllocCpuMemoryParticles(newsize);
  TmgStop(Timers,TMG_SuResizeNp);
  if(updatedivide)RunCellDivide(true);
}

//==============================================================================
/// ES:
/// Crea particulas duplicadas de condiciones periodicas.
/// Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
/// Las nuevas periodicas se situan a partir del Np de entrada, primero las NpbPer
/// de contorno y despues las NpfPer fluidas. El Np de salida contiene tambien las
/// nuevas periodicas.
///
/// EN:
/// Creates duplicate particles for periodic conditions
/// Creates new periodic particles and marks the old ones to be ignored.
/// The new particles are lccated from the value of Np, first the NpbPer
/// for boundaries and then the NpfPer for the fluids. The Np output also contains 
/// the new periodic particles.
//==============================================================================
void JSphGpuSingle::RunPeriodic(){
  const char met[]="RunPeriodic";
  TmgStart(Timers,TMG_SuPeriodic);
  //-Guarda numero de periodicas actuales.
  //-Stores the current number of periodic particles.
  NpfPerM1=NpfPer;
  NpbPerM1=NpbPer;
  //-Marca periodicas actuales para ignorar.
  //-Marks current periodic particles to be ignored.
  cusph::PeriodicIgnore(Np,Codeg);
  //-Crea las nuevas periodicas.
  //-Creates new periodic particles.
  const unsigned npb0=Npb;
  const unsigned npf0=Np-Npb;
  const unsigned np0=Np;
  NpbPer=NpfPer=0;
  BoundChanged=true;
  for(unsigned ctype=0;ctype<2;ctype++){//-0:bound, 1:fluid+floating.
    //-Calcula rango de particulas a examinar (bound o fluid).
	//-Computes the particle range to be examined (bound and fluid).
    const unsigned pini=(ctype? npb0: 0);
    const unsigned num= (ctype? npf0: npb0);
    //-Busca periodicas en cada eje (X, Y e Z).
	//-Searches for periodic zones on each axis (X, Y and Z).
    for(unsigned cper=0;cper<3;cper++)if((cper==0 && PeriActive&1) || (cper==1 && PeriActive&2) || (cper==2 && PeriActive&4)){
      tdouble3 perinc=(cper==0? PeriXinc: (cper==1? PeriYinc: PeriZinc));
      //-Primero busca en la lista de periodicas nuevas y despues en la lista inicial de particulas (necesario para periodicas en mas de un eje).
	  //-First searches in the list of new periodic particles and then in the initial particle list (necessary for periodic zones in more than one axis)
      for(unsigned cblock=0;cblock<2;cblock++){//-0:periodicas nuevas, 1:particulas originales //-0:new periodic particles, 1:original periodic particles
        const unsigned nper=(ctype? NpfPer: NpbPer); //-Numero de periodicas nuevas del tipo a procesar. //-number of new periodic particles for the type currently computed (bound or fluid) 
        const unsigned pini2=(cblock? pini: Np-nper);
        const unsigned num2= (cblock? num:  nper);
        //-Repite la busqueda si la memoria disponible resulto insuficiente y hubo que aumentarla.
		//-Repeats search if the available memory was insufficient and had to be increased.
        bool run=true;
        while(run && num2){
          //-Reserva memoria para crear lista de particulas periodicas.
		  //-Allocates memory to create the periodic particle list.
          unsigned* listpg=ArraysGpu->ReserveUint();
          unsigned nmax=GpuParticlesSize-1; //-Numero maximo de particulas que caben en la lista. //-maximum number of particles that can be included in the list
          //-Genera lista de nuevas periodicas.
		  //-Generates list of new periodic particles
          if(Np>=0x80000000)RunException(met,"The number of particles is too big.");//-Pq el ultimo bit se usa para marcar el sentido en que se crea la nueva periodica. //-Because the last bit is used to mark the reason the new periodical is created
          unsigned count=cusph::PeriodicMakeList(num2,pini2,Stable,nmax,Map_PosMin,Map_PosMax,perinc,Posxyg,Poszg,Codeg,listpg);
          //-Redimensiona memoria para particulas si no hay espacio suficiente y repite el proceso de busqueda.
		  //-Resizes the memory size for the particles if there is not sufficient space and repeats the serach process.
          if(count>nmax || count+Np>GpuParticlesSize){
            ArraysGpu->Free(listpg); listpg=NULL;
            TmgStop(Timers,TMG_SuPeriodic);
            ResizeParticlesSize(Np+count,PERIODIC_OVERMEMORYNP,false);
            TmgStart(Timers,TMG_SuPeriodic);
          }
          else{
            run=false;
            //-Crea nuevas particulas periodicas duplicando las particulas de la lista.
			//-Create new periodic particles duplicating the particles from the list
            if(TStep==STEP_Verlet)cusph::PeriodicDuplicateVerlet(count,Np,DomCells,perinc,listpg,Idpg,Codeg,Dcellg,Posxyg,Poszg,Velrhopg,SpsTaug,VelrhopM1g);
            if(TStep==STEP_Symplectic){
              if((PosxyPreg || PoszPreg || VelrhopPreg) && (!PosxyPreg || !PoszPreg || !VelrhopPreg))RunException(met,"Symplectic data is invalid.") ;
              cusph::PeriodicDuplicateSymplectic(count,Np,DomCells,perinc,listpg,Idpg,Codeg,Dcellg,Posxyg,Poszg,Velrhopg,SpsTaug,PosxyPreg,PoszPreg,VelrhopPreg);
            }
            //-Libera lista y actualiza numero de particulas.
			//-Releases memory and updates the particle number.
            ArraysGpu->Free(listpg); listpg=NULL;
            Np+=count;
            //-Actualiza numero de periodicas nuevas.
			//-Updated number of new periodic particles.
            if(!ctype)NpbPer+=count;
            else NpfPer+=count;
          }
        }
      }
    }
  }
  TmgStop(Timers,TMG_SuPeriodic);
  CheckCudaError(met,"Failed in creation of periodic particles.");
}

//==============================================================================
/// Ejecuta divide de particulas en celdas.
/// Executes the division of particles in cells.
//==============================================================================
void JSphGpuSingle::RunCellDivide(bool updateperiodic){
  const char met[]="RunCellDivide";
  //-Crea nuevas particulas periodicas y marca las viejas para ignorarlas.
  //-Creates new periodic particles and marks the old ones to be ignored.
  if(updateperiodic && PeriActive)RunPeriodic();
  //-Inicia Divide.
  //-Initiates Divide.
  CellDivSingle->Divide(Npb,Np-Npb-NpbPer-NpfPer,NpbPer,NpfPer,BoundChanged,Dcellg,Codeg,Timers,Posxyg,Poszg,Idpg);
  //-Ordena datos de particulas
  //-Sorts particle data
  TmgStart(Timers,TMG_NlSortData);
  {
    unsigned* idpg=ArraysGpu->ReserveUint();
    word*     codeg=ArraysGpu->ReserveWord();
    unsigned* dcellg=ArraysGpu->ReserveUint();
    double2*  posxyg=ArraysGpu->ReserveDouble2();
    double*   poszg=ArraysGpu->ReserveDouble();
    float4*   velrhopg=ArraysGpu->ReserveFloat4();
    CellDivSingle->SortBasicArrays(Idpg,Codeg,Dcellg,Posxyg,Poszg,Velrhopg,idpg,codeg,dcellg,posxyg,poszg,velrhopg);
    swap(Idpg,idpg);           ArraysGpu->Free(idpg);
    swap(Codeg,codeg);         ArraysGpu->Free(codeg);
    swap(Dcellg,dcellg);       ArraysGpu->Free(dcellg);
    swap(Posxyg,posxyg);       ArraysGpu->Free(posxyg);
    swap(Poszg,poszg);         ArraysGpu->Free(poszg);
    swap(Velrhopg,velrhopg);   ArraysGpu->Free(velrhopg);
  }
  if(TStep==STEP_Verlet){
    float4* velrhopg=ArraysGpu->ReserveFloat4();
    CellDivSingle->SortDataArrays(VelrhopM1g,velrhopg);
    swap(VelrhopM1g,velrhopg);   ArraysGpu->Free(velrhopg);
  }
  else if(TStep==STEP_Symplectic && (PosxyPreg || PoszPreg || VelrhopPreg)){//En realidad solo es necesario en el divide del corrector, no en el predictor??? //In reality, only necessary in the corrector not the predictor step?
    if(!PosxyPreg || !PoszPreg || !VelrhopPreg)RunException(met,"Symplectic data is invalid.") ;
    double2* posxyg=ArraysGpu->ReserveDouble2();
    double* poszg=ArraysGpu->ReserveDouble();
    float4* velrhopg=ArraysGpu->ReserveFloat4();
    CellDivSingle->SortDataArrays(PosxyPreg,PoszPreg,VelrhopPreg,posxyg,poszg,velrhopg);
    swap(PosxyPreg,posxyg);      ArraysGpu->Free(posxyg);
    swap(PoszPreg,poszg);        ArraysGpu->Free(poszg);
    swap(VelrhopPreg,velrhopg);  ArraysGpu->Free(velrhopg);
  }
  if(TVisco==VISCO_LaminarSPS){
    tsymatrix3f *spstaug=ArraysGpu->ReserveSymatrix3f();
    CellDivSingle->SortDataArrays(SpsTaug,spstaug);
    swap(SpsTaug,spstaug);  ArraysGpu->Free(spstaug);
  }

  //-Recupera datos del divide.
  //-Retrieves division data.
  Np=CellDivSingle->GetNpFinal();
  Npb=CellDivSingle->GetNpbFinal();
  NpbOk=Npb-CellDivSingle->GetNpbIgnore();
  //-Recupera posiciones de floatings.
  //-Retrieves positions of floating bodies.
  if(CaseNfloat)cusph::CalcRidp(PeriActive!=0,Np-Npb,Npb,CaseNpb,CaseNpb+CaseNfloat,Codeg,Idpg,FtRidpg);
  TmgStop(Timers,TMG_NlSortData);

  //-Gestion de particulas excluidas (contorno y fluid).
  //-Manages excluded particles (boundary and fluid).
  TmgStart(Timers,TMG_NlOutCheck);
  unsigned npout=CellDivSingle->GetNpOut();
  if(npout){
    ParticlesDataDown(npout,Np,true,true,false);
    CellDivSingle->CheckParticlesOut(npout,Idp,AuxPos,AuxRhop,Code);
    AddParticlesOut(npout,Idp,AuxPos,AuxVel,AuxRhop,CellDivSingle->GetNpfOutRhop(),CellDivSingle->GetNpfOutMove());
  }
  TmgStop(Timers,TMG_NlOutCheck);
  BoundChanged=false;
}

//==============================================================================
/// Aplica correccion de Ren a la presion y densidad del contorno.
/// Applies Ren et al. correction to the pressure and desnity of the boundaries.
//==============================================================================
/*void JSphGpuSingle::RunRenCorrection(){
  //-Calcula presion en contorno a partir de fluido.
  //-Computes pressure in the boundary from the fluid
  float *presskf=ArraysGpu->ReserveFloat();
  cusph::Interaction_Ren(Psimple,WithFloating,CellMode,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,FtoMasspg,Gravity,presskf);
  //-Recalcula valores de presion y densidad en contorno segun RenBeta.
  //-Computes corrected values of pressure and density on the boundary accordng to RenBeta
  cusph::ComputeRenPress(Psimple,NpbOk,RenCorrection,presskf,Velrhopg,PsPospressg);
  ArraysGpu->Free(presskf); presskf=NULL;
}*/

//==============================================================================
/// Interaccion para el calculo de fuerzas.
/// Interaction for force computation.
//==============================================================================
void JSphGpuSingle::Interaction_Forces(TpInter tinter,double dt){
  const char met[]="Interaction_Forces";
  PreInteraction_Forces(tinter,dt);
  TmgStart(Timers,TMG_CfForces);
  //if(RenCorrection)RunRenCorrection();

  const bool lamsps=(TVisco==VISCO_LaminarSPS);
  const unsigned bsfluid=BlockSizes.forcesfluid;
  const unsigned bsbound=BlockSizes.forcesbound;

  //-Interaccion Fluid-Fluid/Bound & Bound-Fluid.
  //-Interaction Fluid-Fluid/Bound & Bound-Fluid.
  
  cusph::Interaction_Forces(Psimple,WithFloating,UseDEM,CellMode,Visco*ViscoBoundFactor,Visco,bsbound,bsfluid,tinter,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,dWxCorrg,dWzCorrg,FtoMasspg,SpsTaug,SpsGradvelg,ViscDtg,Arg,Aceg,Simulate2D);
  if(tinter==1)cudaMemset(Velrhopg,0,sizeof(float4)*Npb);
  //-Interaccion DEM Floating-Bound & Floating-Floating //(DEM)
  //-Interaction DEM Floating-Bound & Floating-Floating //(DEM)
//  if(UseDEM)cusph::Interaction_ForcesDem(Psimple,CellMode,BlockSizes.forcesdem,CaseNfloat,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,FtRidpg,DemDatag,float(DemDtForce),Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,ViscDtg,Aceg);

  //-Calculo de Tau para Laminar+SPS
  //-Computes Tau for Laminar+SPS
  //if(lamsps)cusph::ComputeSpsTau(Np,Npb,SpsSmag,SpsBlin,Velrhopg,SpsGradvelg,SpsTaug);

  //-Para simulaciones 2D anula siempre la 2º componente
  //-For 2D simulations always overrides the 2nd component (Y axis)
  if(Simulate2D)cusph::Resety(Np-Npb,Npb,Aceg);

  //if(Deltag)cusph::AddDelta(Np-Npb,Deltag+Npb,Arg+Npb);//-Añade correccion de Delta-SPH a Arg[]. //-Adds the Delta-SPH correction for the density
  CheckCudaError(met,"Failed while executing kernels of interaction.");

  //-Calculates maximum value of ViscDt.
  //if(Np)ViscDtMax=cusph::ReduMaxFloat(Np,0,ViscDtg,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np)));
  //-Calculates maximum value of Ace.
  //AceMax=ComputeAceMax(ViscDtg); 

  TmgStop(Timers,TMG_CfForces);
  CheckCudaError(met,"Failed in reduction of viscdt.");
}

//==============================================================================
/// Devuelve valor maximo de (ace.x^2 + ace.y^2 + ace.z^2) a partir de Acec[].
/// Returns the maximum value of  (ace.x^2 + ace.y^2 + ace.z^2) from Acec[].
//==============================================================================
double JSphGpuSingle::ComputeAceMax(float *auxmem){
  float acemax=0;
  const unsigned npf=Np-Npb;
  if(!PeriActive)cusph::ComputeAceMod(npf,Aceg+Npb,auxmem);//-Sin condiciones periodicas. //-Without periodic conditions.
  else cusph::ComputeAceMod(npf,Codeg+Npb,Aceg+Npb,auxmem);//-Con condiciones periodicas ignora las particulas periodicas. //-With periodic conditions ignores the periodic particles.
  if(npf)acemax=cusph::ReduMaxFloat(npf,0,auxmem,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(npf)));
  return(sqrt(double(acemax)));
}

//==============================================================================
/// ES:
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Verlet.
///
/// EN:
/// Particle interaction and update of particle data according to
/// the computed forces using the Verlet time stepping scheme
//==============================================================================
/*double JSphGpuSingle::ComputeStep_Ver(){
  Interaction_Forces(INTER_Forces);     //-Interaccion //-Interaction
  const double dt=DtVariable(true);     //-Calcula nuevo dt //-Calculate new dt
  DemDtForce=dt;                        //(DEM)
  if(TShifting)RunShifting(dt);         //-Shifting
  ComputeVerlet(dt);                    //-Actualiza particulas usando Verlet //-Update particle data using Verlet
  if(CaseNfloat)RunFloating(dt,false); //-Gestion de floating bodies //-Management of floating bodies
  PosInteraction_Forces();              //-Libera memoria de interaccion //-Releases memory of interaction
  return(dt);
}*/

//==============================================================================
/// ES:
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Symplectic.
///
/// EN:
/// Particle interaction and update of particle data according to
/// the computed forces using the Symplectic time stepping scheme
//==============================================================================
double JSphGpuSingle::ComputeStep_Sym(){
  const double dt=DtPre;
  //-Predictor
  //----------- 
  InitAdvection(dt);
  RunCellDivide(true);
  Interaction_Forces(INTER_Forces,dt);       //-Interaccion //-Interaction
  //const double ddt_p=DtVariable(false);   //-Calcula dt del predictor //-Computes dt in the predictor step
  //if(TShifting)RunShifting(dt*.5);        //-Shifting
  ComputeSymplecticPre(dt);               //-Aplica Symplectic-Predictor a las particulas //Applies Symplectic-Predictor to the particles
  //if(CaseNfloat)RunFloating(dt*.5,true);  //-Gestion de floating bodies //-Management of the floating bodies
  PosInteraction_Forces(INTER_Forces);                //-Libera memoria de interaccion //-Releases memory of the interaction
  //-Pressure Poisson equation
  //-----------
  KernelCorrection();
  
  SolvePPE(dt); //-Solve pressure Poisson equation
  //-Corrector
  //-----------
  //DemDtForce=dt;                          //(DEM)
  Interaction_Forces(INTER_ForcesCorr,dt);   //-Interaccion //-interaction
  //const double ddt_c=DtVariable(true);    //-Calcula dt del corrector //Computes dt in the corrector step
  ComputeSymplecticCorr(dt);              //-Aplica Symplectic-Corrector a las particulas //Applies Symplectic-Corrector to the particles
  //if(CaseNfloat)RunFloating(dt,false);    //-Gestion de floating bodies //-Management of the floating bodies
  PosInteraction_Forces(INTER_ForcesCorr);                //-Libera memoria de interaccion //-Releases memory of the interaction
  if(TShifting)RunShifting(dt);           //-Shifting
  //DtPre=min(ddt_p,ddt_c);                 //-Calcula el dt para el siguiente ComputeStep //-Computes dt for the next ComputeStep
  count++;
  return(dt);
}

//==============================================================================
/// Procesa floating objects.
/// Processes floating objects.
//==============================================================================
void JSphGpuSingle::RunFloating(double dt,bool predictor){
  const char met[]="RunFloating";
  if(TimeStep>=FtPause){//-Se usa >= pq si FtPause es cero en symplectic-predictor no entraria. //-Used because if FtPause is zero we don't enter the predictor.
    TmgStart(Timers,TMG_SuFloating);
    //-Calcula fuerzas sobre floatings.
	//-Computes forces for the floating objects
   // cusph::FtCalcForces(PeriActive!=0,FtCount,Gravity,FtoDatag,FtoMasspg,FtoCenterg,FtRidpg,Posxyg,Poszg,Aceg,FtoForcesg);
    //-Aplica movimiento sobre floatings.
	//-Applies movement to the floating objects
    cusph::FtUpdate(PeriActive!=0,predictor,Simulate2D,FtCount,dt,Gravity,FtoDatag,FtRidpg,FtoForcesg,FtoCenterg,FtoVelg,FtoOmegag,Posxyg,Poszg,Dcellg,Velrhopg,Codeg);
    TmgStop(Timers,TMG_SuFloating);
  }
}

//==============================================================================
/// Inicia proceso de simulacion.
/// Runs the simulation
//==============================================================================
void JSphGpuSingle::Run(std::string appname,JCfgRun *cfg,JLog2 *log){
  const char* met="Run";
  if(!cfg||!log)return;
  AppName=appname; Log=log;

  //-Seleccion de GPU 
  //-Selection of GPU
  //-------------------
  SelecDevice(cfg->GpuId);

  //-Configura timers 
  //-Configures timers
  //-------------------
  TmgCreation(Timers,cfg->SvTimers);
  TmgStart(Timers,TMG_Init);
  if(cfg->SvTimersStep>0){
    TimersStep=new JTimersStep(cfg->DirOut,cfg->SvTimersStep,0,0);
    for(unsigned ct=0;ct<TimerGetCount();ct++)if(TimerIsActive(ct))TimersStep->AddTimer(TimerGetName(ct),TimerGetPtrValue(ct));
  }

  //-Carga de parametros y datos de entrada
  //-Loads parameters and input data
  //-----------------------------------------
  LoadConfig(cfg);
  LoadCaseParticles();
  ConfigConstants(Simulate2D);
  ConfigDomain();
  ConfigRunMode("Single-Gpu");

  //-Inicializacion de variables de ejecucion
  //-Initialisation of the execution variables
  //-------------------------------------------
  InitRun();
  UpdateMaxValues();
  PrintAllocMemory(GetAllocMemoryCpu(),GetAllocMemoryGpu());
  SaveData(); 
  TmgResetValues(Timers);
  TmgStop(Timers,TMG_Init);
  PartNstep=-1; Part++;

  //-Bucle principal
  //-Main Loop
  //------------------
  bool partoutstop=false;
  TimerSim.Start();
  TimerPart.Start();
  Log->Print(string("\n[Initialising simulation (")+RunCode+")  "+fun::GetDateTime()+"]");
  PrintHeadPart();
  //-finding dummy particle relations to wall particles
  FindIrelation(); count =1;
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
    //if(Nstep>=1)break;
    clock_t stop = clock();   
    double dif = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    cout<<"Timestep Time = " << dif << "ms\n";
  }
  TimerSim.Stop(); TimerTot.Stop();

  //-Fin de simulacion
  //-End of the simulation.
  //--------------------
  FinishRun(partoutstop);
}

//==============================================================================
/// Genera los ficheros de salida de datos
/// Generates output files for particle data
//==============================================================================
void JSphGpuSingle::SaveData(){
  const bool save=(SvData!=SDAT_None&&SvData!=SDAT_Info);
  const unsigned npsave=Np-NpbPer-NpfPer; //-Resta las periodicas si las hubiera. //-Subtracts periodic particles if any.
  //-Recupera datos de particulas en GPU.
  //-Retrieves particle data from the GPU.
  if(save){
    TmgStart(Timers,TMG_SuDownData);
    unsigned npnormal=ParticlesDataDown(Np,0,false,true,PeriActive!=0);
    if(npnormal!=npsave)RunException("SaveData","The number of particles is invalid.");
    TmgStop(Timers,TMG_SuDownData);
  }
  //-Recupera datos de floatings en GPU.
  //-Retrieve floating object data from the GPU
  if(FtCount){
    TmgStart(Timers,TMG_SuDownData);
    cudaMemcpy(FtoCenter,FtoCenterg,sizeof(double3)*FtCount,cudaMemcpyDeviceToHost);
    for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].center=FtoCenter[cf];
    tfloat3 *aux=(tfloat3 *)FtoCenter;
    cudaMemcpy(aux,FtoVelg,sizeof(float3)*FtCount,cudaMemcpyDeviceToHost);
    for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].fvel=aux[cf];
    cudaMemcpy(aux,FtoOmegag,sizeof(float3)*FtCount,cudaMemcpyDeviceToHost);
    for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].fomega=aux[cf];
    TmgStop(Timers,TMG_SuDownData);
  }
  //-Reune informacion adicional.
  //-Collects additional information
  TmgStart(Timers,TMG_SuSavePart);
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
    infoplus.gpudata=true;
    infoplus.memorynctalloc=infoplus.memorynctused=GetMemoryGpuNct();
    infoplus.memorynpalloc=infoplus.memorynpused=GetMemoryGpuNp();
    TimerSim.Stop();
    infoplus.timesim=TimerSim.GetElapsedTimeD()/1000.;
  }
  //-Graba datos de particulas.
  //-Stores particle data.
  const tdouble3 vdom[2]={OrderDecode(CellDivSingle->GetDomainLimits(true)),OrderDecode(CellDivSingle->GetDomainLimits(false))};
  JSph::SaveData(npsave,Idp,AuxPos,AuxVel,AuxRhop,1,vdom,&infoplus);
  //-Graba informacion de ejecucion.
  //-Stores information of the execution.
  if(TimersStep)TimersStep->SaveData();
  TmgStop(Timers,TMG_SuSavePart);
}

//==============================================================================
/// Muestra y graba resumen final de ejecucion.
/// Displays and stores final summary of the execution.
//==============================================================================
void JSphGpuSingle::FinishRun(bool stop){
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

//==============================================================================
/// Irelation - Dummy particles' respective Wall particle
//==============================================================================
void JSphGpuSingle::FindIrelation(){
  const unsigned bsbound=BlockSizes.forcesbound;
  cusph::FindIrelation(bsbound,Npb,Posxyg,Poszg,Codeg,Idpg,Irelationg);
}

//==============================================================================
/// Kernel Gradient Correction
//==============================================================================
void JSphGpuSingle::KernelCorrection(){
  const unsigned bsfluid=BlockSizes.forcesfluid;
  const unsigned bsbound=BlockSizes.forcesbound;

  dWxCorrg=ArraysGpu->ReserveDouble3();
  dWzCorrg=ArraysGpu->ReserveDouble3();

  cudaMemset(dWxCorrg,0,sizeof(double3)*Np);						
  cudaMemset(dWzCorrg,0,sizeof(double3)*Np);
  
  cusph::KernelCorrection(Psimple,CellMode,bsfluid,bsbound,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,PsPospressg,Velrhopg,dWxCorrg,dWzCorrg,Codeg);
}

//==============================================================================
/// Initial advection
//==============================================================================
void JSphGpuSingle::InitAdvection(double dt){
    PosxyPreg=ArraysGpu->ReserveDouble2();
    PoszPreg=ArraysGpu->ReserveDouble();
    VelrhopPreg=ArraysGpu->ReserveFloat4();
    
    unsigned np=Np;
    unsigned npb=Npb;
    unsigned npf=np-npb;
    //-Cambia datos a variables Pre para calcular nuevos datos.
    //-Changes data of predictor variables for calculating the new data
    cudaMemcpy(PosxyPreg,Posxyg,sizeof(double2)*np,cudaMemcpyDeviceToDevice);     //Es decir... PosxyPre[] <= Posxy[] //i.e. PosxyPre[] <= Posxy[]
    cudaMemcpy(PoszPreg,Poszg,sizeof(double)*np,cudaMemcpyDeviceToDevice);        //Es decir... PoszPre[] <= Posz[] //i.e. PoszPre[] <= Posz[]
    cudaMemcpy(VelrhopPreg,Velrhopg,sizeof(float4)*np,cudaMemcpyDeviceToDevice); //Es decir... VelrhopPre[] <= Velrhop[] //i.e. VelrhopPre[] <= Velrhop[]
	  cudaMemset(Velrhopg,0,sizeof(float4)*npb);
    //-Copia posicion anterior del contorno.
    //-Copies previous position of the boundaries.
    //cudaMemcpy(Posxyg,PosxyPreg,sizeof(double2)*Npb,cudaMemcpyDeviceToDevice);
    //cudaMemcpy(Poszg,PoszPreg,sizeof(double)*Npb,cudaMemcpyDeviceToDevice);
    
    double2 *movxyg=ArraysGpu->ReserveDouble2();  cudaMemset(movxyg,0,sizeof(double2)*np);
    double *movzg=ArraysGpu->ReserveDouble();     cudaMemset(movzg,0,sizeof(double)*np);
    
    cusph::ComputeRStar(WithFloating,npf,npb,VelrhopPreg,dt,Codeg,movxyg,movzg);
	  cusph::ComputeStepPos2(PeriActive,WithFloating,npf,npb,PosxyPreg,PoszPreg,movxyg,movzg,Posxyg,Poszg,Dcellg,Codeg);

    ArraysGpu->Free(movxyg);   movxyg=NULL;
    ArraysGpu->Free(movzg);    movzg=NULL; 
}

//==============================================================================
/// PPE Solver
//==============================================================================
#include <fstream>
#include <sstream>
#include <iomanip>
void JSphGpuSingle::SolvePPE(double dt){ 
  const char met[]="SolvePPE";
  tuint3 ncells=CellDivSingle->GetNcells();
  const int2 *begincell=CellDivSingle->GetBeginCell();
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  
  const unsigned *dcell=Dcellg;
  const unsigned np=Np;
  const unsigned npb=Npb;
  const unsigned npbok=NpbOk;
  const unsigned npf = np - npb;
  unsigned PPEDim=0;
  const unsigned bsbound=BlockSizes.forcesbound;
  const unsigned bsfluid=BlockSizes.forcesfluid;
  
  POrderg=ArraysGpu->ReserveUint(); cusph::InitArrayPOrder(np,POrderg,np);
  Divrg=ArraysGpu->ReserveDouble(); cudaMemset(Divrg,0,sizeof(double)*np);
  CheckCudaError(met,"Memory Assignment PORder");
  MatrixOrder(np,0,bsbound,bsfluid,POrderg,ncells,begincell,cellmin,dcell,Idpg,Irelationg,Codeg,PPEDim);
  CheckCudaError(met,"MatrixOrder");
  cusph::FreeSurfaceFind(Psimple,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,Divrg);
  CheckCudaError(met,"FreeSurfaceFind");
  //Create matrix
  cudaMalloc((void**)&b,sizeof(double)*PPEDim); cudaMemset(b,0,sizeof(double)*PPEDim);
  cudaMalloc((void**)&X,sizeof(double)*PPEDim); cudaMemset(X,0,sizeof(double)*PPEDim);
  cudaMalloc((void**)&rowInd,sizeof(unsigned int)*(PPEDim+1)); cudaMemset(rowInd,0,sizeof(unsigned int)*(PPEDim+1));
  CheckCudaError(met,"Memory Assignment b");
  cusph::PopulateMatrixB(Psimple,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Posxyg,Poszg,PsPospressg,Velrhopg,dWxCorrg,dWzCorrg,b,POrderg,Idpg,dt,PPEDim,Divrg,Codeg,FreeSurface);
  cusph::MatrixStorage(Psimple,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,Divrg,POrderg,rowInd,FreeSurface);
  CheckCudaError(met,"MatrixStorage");
  unsigned Nnz=MatrixASetup(PPEDim,rowInd);
  cudaMalloc((void**)&a,sizeof(double)*Nnz); cudaMemset(a,0,sizeof(double)*Nnz);
  cudaMalloc((void**)&colInd,sizeof(unsigned int)*Nnz); cusph::InitArrayCol(Nnz,colInd,int(PPEDim));
  CheckCudaError(met,"Memory Assignment a");
  cusph::PopulateMatrixA(Psimple,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,GravityDbl,Posxyg,Poszg,PsPospressg,Velrhopg,a,b,rowInd,colInd,POrderg,Idpg,PPEDim,Divrg,Codeg,Irelationg,FreeSurface);

  cusph::FreeSurfaceMark(Psimple,bsbound,bsfluid,np,npb,npbok,Divrg,a,b,rowInd,POrderg,Codeg,PI,FreeSurface);
  CheckCudaError(met,"Free Surface");
 /* unsigned int *POrderCPU; POrderCPU=new unsigned int[np]; cudaMemcpy(POrderCPU,POrderg,sizeof(unsigned int)*np,cudaMemcpyDeviceToHost);
  unsigned int *IdCPU; IdCPU=new unsigned int[np]; cudaMemcpy(IdCPU,Idpg,sizeof(unsigned int)*np,cudaMemcpyDeviceToHost);
  double *bcpu; bcpu=new double[PPEDim]; cudaMemcpy(bcpu,b,sizeof(double)*PPEDim,cudaMemcpyDeviceToHost);
  double *acpu; acpu=new double[Nnz]; cudaMemcpy(acpu,a,sizeof(double)*Nnz,cudaMemcpyDeviceToHost);
  unsigned int *rowcpu; rowcpu=new unsigned int[PPEDim+1]; cudaMemcpy(rowcpu,rowInd,sizeof(unsigned int)*(PPEDim+1),cudaMemcpyDeviceToHost);
  unsigned int *colcpu; colcpu=new unsigned int[Nnz]; cudaMemcpy(colcpu,colInd,sizeof(unsigned int)*(Nnz),cudaMemcpyDeviceToHost);
  
  /*for(int i=0;i<(PPEDim+1);i++){
    std::cout<<i<<"\t"<<rowcpu[i]<<"\n";
    system("PAUSE");
  }*/
  
 /* ofstream FileOutput;
    string TimeFile;

    ostringstream TimeNum;
    TimeNum << count;
    ostringstream FileNum;
    FileNum << count;

    TimeFile =  "GPU Fluid Properties_" + FileNum.str() + ", T = " + TimeNum.str() + ".txt";

    FileOutput.open(TimeFile.c_str());
    FileOutput << fixed << setprecision(19) << Gravity.z << "\n";
    FileOutput << fixed << setprecision(19) << GravityDbl.z << "\n";
  for(int i=0;i<npbok;i++){
    FileOutput << fixed << setprecision(19) << "particle "<< IdCPU[i] << "\t Order " << POrderCPU[i] << "\t b " << bcpu[POrderCPU[i]] << "\n";
    if(POrderCPU[i]!=np)for(int j=rowcpu[POrderCPU[i]];j<rowcpu[POrderCPU[i]+1];j++) FileOutput << fixed << setprecision(16) << j << "\t" << acpu[j] << "\t" << colcpu[j] << "\n";
  }

  for(int i=npb;i<np;i++){
    FileOutput << fixed << setprecision(20) << "particle "<< IdCPU[i] << "\t Order " << POrderCPU[i] << "\t b " << bcpu[POrderCPU[i]] << "\n";
    if(POrderCPU[i]!=np)for(int j=rowcpu[POrderCPU[i]];j<rowcpu[POrderCPU[i]+1];j++) FileOutput << fixed << setprecision(16) << j << "\t" << acpu[j] << "\t" << colcpu[j] << "\n";
  }
  FileOutput.close();*/
   
  //count++;
  //cusph::solveViennaCPU(a,b,X,rowInd,colInd,PPEDim,Nnz); 

  cusph::solveVienna(TPrecond,TAMGInter,Tolerance,Iterations,StrongConnection,JacobiWeight,Presmooth,Postsmooth,CoarseCutoff,a,X,b,rowInd,colInd,Nnz,PPEDim); 

  /*double *xcpu; xcpu=new double[PPEDim]; cudaMemcpy(xcpu,X,sizeof(double)*PPEDim,cudaMemcpyDeviceToHost);

    TimeFile =  "Pressure" + FileNum.str() + ", T = " + TimeNum.str() + ".txt";

    FileOutput.open(TimeFile.c_str());

  for(int i=0;i<PPEDim;i++){
    FileOutput << fixed << setprecision(10) << i << "\t" << xcpu[i] <<"\n";
  }
  FileOutput.close();
    delete[] POrderCPU; POrderCPU=NULL;
  delete[] IdCPU; IdCPU=NULL;
  delete[] acpu; acpu=NULL;
  delete[] bcpu; bcpu=NULL;
  delete[] xcpu; xcpu=NULL;
  delete[] rowcpu; rowcpu=NULL;
  delete[] colcpu; colcpu=NULL;*/
  cusph::PressureAssign(Psimple,bsbound,bsfluid,np,npb,npbok,Gravity,Posxyg,Poszg,PsPospressg,Velrhopg,X,POrderg,Idpg,Codeg,Irelationg,Divrg);
  CheckCudaError(met,"pressure assign");
  ArraysGpu->Free(POrderg);       POrderg=NULL;
  ArraysGpu->Free(Divrg);		      Divrg=NULL;
  cudaFree(b); cudaFree(X); cudaFree(a); cudaFree(rowInd); cudaFree(colInd);
  CheckCudaError(met,"free");
}

//==============================================================================
/// Shifting
//==============================================================================
void JSphGpuSingle::RunShifting(double dt){ 
  const unsigned np=Np;
  const unsigned npb=Npb;
  const unsigned npbok=NpbOk;
  const unsigned npf = np - npb;

  PosxyPreg=ArraysGpu->ReserveDouble2();
  PoszPreg=ArraysGpu->ReserveDouble();
  VelrhopPreg=ArraysGpu->ReserveFloat4();

  ShiftPosg=ArraysGpu->ReserveDouble3();
  Divrg=ArraysGpu->ReserveDouble();
  cudaMemset(ShiftPosg,0,sizeof(float3)*np);       //ShiftPosg[]=0
  cudaMemset(Divrg,0,sizeof(double)*np);        //Divrg[]=0

  //-Cambia datos a variables Pre para calcular nuevos datos.
  //-Changes data of predictor variables for calculating the new data
  cudaMemcpy(PosxyPreg,Posxyg,sizeof(double2)*np,cudaMemcpyDeviceToDevice);     //Es decir... PosxyPre[] <= Posxy[] //i.e. PosxyPre[] <= Posxy[]
  cudaMemcpy(PoszPreg,Poszg,sizeof(double)*np,cudaMemcpyDeviceToDevice);        //Es decir... PoszPre[] <= Posz[] //i.e. PoszPre[] <= Posz[]
  cudaMemcpy(VelrhopPreg,Velrhopg,sizeof(float4)*np,cudaMemcpyDeviceToDevice); //Es decir... VelrhopPre[] <= Velrhop[] //i.e. VelrhopPre[] <= Velrhop[]

  RunCellDivide(true);

  tuint3 ncells=CellDivSingle->GetNcells();
  const int2 *begincell=CellDivSingle->GetBeginCell();
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();

  const unsigned *dcell=Dcellg;
  const unsigned bsbound=BlockSizes.forcesbound;
  const unsigned bsfluid=BlockSizes.forcesfluid;

  PsPospressg=ArraysGpu->ReserveFloat4();
  cusph::PreInteractionSimple(Np,Posxyg,Poszg,Velrhopg,PsPospressg,CteB,Gamma);

  cusph::Interaction_Shifting(Psimple,WithFloating,UseDEM,CellMode,Visco*ViscoBoundFactor,Visco,bsfluid,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,FtoMasspg,TShifting,ShiftPosg,Divrg,TensileN,TensileR);

  JSphGpu::RunShifting(dt);

  Shift(dt,bsfluid);
  
  ArraysGpu->Free(PosxyPreg);     PosxyPreg=NULL;
  ArraysGpu->Free(PoszPreg);      PoszPreg=NULL;
  ArraysGpu->Free(VelrhopPreg);   VelrhopPreg=NULL;
  ArraysGpu->Free(ShiftPosg);     ShiftPosg=NULL;
  ArraysGpu->Free(Divrg);         Divrg=NULL;
  ArraysGpu->Free(PsPospressg);   PsPospressg=NULL; 
}

