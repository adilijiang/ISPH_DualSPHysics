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
#include "JTimeOut.h"
#include "JSphGpu_ker.h"

#include "JBlockSizeAuto.h"

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
  //-Carga configuracion basica general
  //-Loads general configuration
  JSph::LoadConfig(cfg);
	BlockSizeMode=cfg->BlockSizeMode;
  //-Checks compatibility of selected options.
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
  if(Simulate2D&&PeriY)RunException("LoadCaseParticles","Cannot use periodic conditions in Y with 2D simulations");
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
  if(CaseNfloat && PeriActive!=0 && !PartBegin)CalcFloatingRadius(Np,AuxPos,Idp);

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
            if(TStep==STEP_Symplectic){
              if((PosxyPreg || PoszPreg || VelrhopPreg) && (!PosxyPreg || !PoszPreg || !VelrhopPreg))RunException(met,"Symplectic data is invalid.") ;
            //  cusph::PeriodicDuplicateSymplectic(count,Np,DomCells,perinc,listpg,Idpg,Codeg,Dcellg,Posxyg,Poszg,Velrhopg,PosxyPreg,PoszPreg,VelrhopPreg);
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
		double3*	velocityg=ArraysGpu->ReserveDouble3();
		double*		pressureg=ArraysGpu->ReserveDouble();
    CellDivSingle->SortBasicArrays(Idpg,Codeg,Dcellg,Posxyg,Poszg,Velrhopg,Velocity,Pressureg,idpg,codeg,dcellg,posxyg,poszg,velrhopg,velocityg,pressureg);
    swap(Idpg,idpg);           ArraysGpu->Free(idpg);
    swap(Codeg,codeg);         ArraysGpu->Free(codeg);
    swap(Dcellg,dcellg);       ArraysGpu->Free(dcellg);
    swap(Posxyg,posxyg);       ArraysGpu->Free(posxyg);
    swap(Poszg,poszg);         ArraysGpu->Free(poszg);
    swap(Velrhopg,velrhopg);   ArraysGpu->Free(velrhopg);
		swap(Velocity,velocityg);  ArraysGpu->Free(velocityg);
		swap(Pressureg,pressureg);  ArraysGpu->Free(pressureg);
  }
  if(TStep==STEP_Symplectic && (PosxyPreg || PoszPreg || VelrhopPreg)){//En realidad solo es necesario en el divide del corrector, no en el predictor??? //In reality, only necessary in the corrector not the predictor step?
    if(!PosxyPreg || !PoszPreg || !VelrhopPreg)RunException(met,"Symplectic data is invalid.") ;
    double2* posxyg=ArraysGpu->ReserveDouble2();
    double* poszg=ArraysGpu->ReserveDouble();
    float4* velrhopg=ArraysGpu->ReserveFloat4();
		double3* velocitypre=ArraysGpu->ReserveDouble3();
    CellDivSingle->SortDataArrays(PosxyPreg,PoszPreg,VelrhopPreg,VelocityPre,posxyg,poszg,velrhopg,velocitypre);
    swap(PosxyPreg,posxyg);      ArraysGpu->Free(posxyg);
    swap(PoszPreg,poszg);        ArraysGpu->Free(poszg);
    swap(VelrhopPreg,velrhopg);  ArraysGpu->Free(velrhopg);
		swap(VelocityPre,velocitypre);  ArraysGpu->Free(velocitypre);
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
/// Interaccion para el calculo de fuerzas.
/// Interaction for force computation.
//==============================================================================
void JSphGpuSingle::Interaction_Forces(TpInter tinter,double dt){
  const char met[]="Interaction_Forces";
  TmgStart(Timers,TMG_CfForces);
  PreInteraction_Forces(tinter,dt);
  const unsigned bsfluid=BlockSizes.forcesfluid;
  const unsigned bsbound=BlockSizes.forcesbound;
	CheckCudaError(met,"Failed checkin.");

	//-Interaccion Fluid-Fluid/Bound & Bound-Fluid.
  cusph::Interaction_Forces(TKernel,WithFloating,UseDEM,TSlipCond,Schwaiger,CellMode,Visco*ViscoBoundFactor,Visco,bsbound,bsfluid,tinter,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,Velocity,Codeg,Idpg,dWxCorrg,dWyCorrg,dWzCorrg,FtoMasspg,Aceg,Simulate2D,Divrg,MirrorPosg,MirrorCellg,MLSg,rowIndg,sumFrg,taog,BoundaryFS,FreeSurface,PistonPosX,NULL,NULL,Pressureg,PistonVel,RightWall,Gravity);	
	
  //-Interaccion DEM Floating-Bound & Floating-Floating //(DEM)
  //-Interaction DEM Floating-Bound & Floating-Floating //(DEM)
  //if(UseDEM)cusph::Interaction_ForcesDem(Psimple,CellMode,BlockSizes.forcesdem,CaseNfloat,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,FtRidpg,DemDatag,float(DemDtForce),Posxyg,Poszg,PsPospressg,Velrhopg,Codeg,Idpg,ViscDtg,Aceg);
  //-Para simulaciones 2D anula siempre la 2º componente
  //-For 2D simulations always overrides the 2nd component (Y axis)
  if(Simulate2D)cusph::ResetAcey(Np-Npb,Aceg);
  TmgStop(Timers,TMG_CfForces);
  CheckCudaError(met,"Failed while executing kernels of interaction.");
}

//==============================================================================
/// Devuelve valor maximo de (ace.x^2 + ace.y^2 + ace.z^2) a partir de Acec[].
/// Returns the maximum value of  (ace.x^2 + ace.y^2 + ace.z^2) from Acec[].
//==============================================================================
double JSphGpuSingle::ComputeAceMax(float *auxmem){
  float acemax=0;
  const unsigned npf=Np-Npb;
//  if(!PeriActive)cusph::ComputeAceMod(npf,Aceg+Npb,auxmem);//-Sin condiciones periodicas. //-Without periodic conditions.
  //else cusph::ComputeAceMod(npf,Codeg+Npb,Aceg+Npb,auxmem);//-Con condiciones periodicas ignora las particulas periodicas. //-With periodic conditions ignores the periodic particles.
  if(npf)acemax=cusph::ReduMaxFloat(npf,0,auxmem,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(npf)));
  return(sqrt(double(acemax)));
}

//==============================================================================
/// ES:
/// Realiza interaccion y actualizacion de particulas segun las fuerzas 
/// calculadas en la interaccion usando Symplectic.
///
/// EN:
/// Particle interaction and update of particle data according to
/// the computed forces using the Symplectic time stepping scheme
//==============================================================================
double JSphGpuSingle::ComputeStep_Sym(double dt){
  //SaveVtkData("InitSymplectic.vtk",Nstep,Np,Posxyg,Poszg,Idpg,Velrhopg);
  //-Predictor
  //----------- 
	TmgStart(Timers,TMG_Stage1);
  InitAdvection(dt);
	RunCellDivide(true);
	Interaction_Forces(INTER_Forces,dt);        //-Interaction
	ComputeSymplecticPre(dt);                   //-Applies Symplectic-Predictor to the particles
	//-Pressure Poisson equation
  //-----------
	TmgStop(Timers,TMG_Stage1);
	SolvePPE(dt);                               //-Solve pressure Poisson equation
  //-Corrector
  //-----------
	TmgStart(Timers,TMG_Stage3);
  Interaction_Forces(INTER_ForcesCorr,dt);    //-Interaccion //-interaction
  ComputeSymplecticCorr(dt);                  //Applies Symplectic-Corrector to the particles
	TmgStop(Timers,TMG_Stage3);
	//-Shifting
	//-----------
	TmgStart(Timers,TMG_Stage4);
  if(TShifting)RunShifting(dt);               //-Shifting
	TmgStop(Timers,TMG_Stage4);
//	if(VariableTimestep) dt=ComputeVariable();
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
//    cusph::FtCalcForces(PeriActive!=0,FtCount,Gravity,FtoDatag,FtoMasspg,FtoCenterg,FtRidpg,Posxyg,Poszg,Aceg,FtoForcesg);
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
	double stepdt=DtPre;
  TimerSim.Start();
  TimerPart.Start();
  Log->Print(string("\n[Initialising simulation (")+RunCode+")  "+fun::GetDateTime()+"]");
  PrintHeadPart();
	//cudaMemset(MirrorPosg,0,sizeof(double3)*Npb);
	//MirrorBoundary();
	count=1;
	cudaMemset(Velocity,0,sizeof(double3)*Np);
	PistonPosX=Dp*0.5;
	PistonVel=0;
	while(TimeStep<TimeMax){
		if(CaseNmoving)RunMotion(stepdt);
    stepdt=ComputeStep_Sym(stepdt);
    if(PartDtMin>stepdt)PartDtMin=stepdt; if(PartDtMax<stepdt)PartDtMax=stepdt;
    TimeStep+=stepdt;
    partoutstop=(Np<NpMinimum || !Np);
    if(TimeStep>=TimePartNext || partoutstop){ 
			if(partoutstop){
        Log->Print("\n**** Particles OUT limit reached...\n");
        TimeMax=TimeStep;
      }
      SaveData();
			SaveVtkData("InitSymplectic.vtk",Nstep,Np,Posxyg,Poszg,Idpg,Velocity,Pressureg);
      Part++;
      PartNstep=Nstep;
      TimeStepM1=TimeStep;
			TimePartNext=TimeOut->GetNextTime(TimeStep);
      TimerPart.Start();
    }
    UpdateMaxValues();
    Nstep++;
    //if(Nstep>=2)break;
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
#include "JFormatFiles2.h"
void JSphGpuSingle::SaveVtkData(std::string fname,unsigned fnum,unsigned np,const double2 *posxy,const double *posz,const unsigned *idp,const double3 *velrhop,const double *pressure)const{
  //-Allocate memory.
  tdouble2 *pxy=new tdouble2[np];
  double *pz=new double[np];
  tfloat3 *pos=new tfloat3[np];
  unsigned *idph=new unsigned[np];
  double3 *vel=new double3[np];
  //float *rhop=new float[np];
	double *pres=new double[np];
	//float *divr=new float[np];

  //-Copies memory to CPU.
  cudaMemcpy(pxy,posxy,sizeof(double2)*np,cudaMemcpyDeviceToHost);
  cudaMemcpy(pz,posz,sizeof(double)*np,cudaMemcpyDeviceToHost);
  cudaMemcpy(idph,idp,sizeof(unsigned)*np,cudaMemcpyDeviceToHost);
  cudaMemcpy(vel,velrhop,sizeof(double3)*np,cudaMemcpyDeviceToHost);
	cudaMemcpy(pres,pressure,sizeof(double)*np,cudaMemcpyDeviceToHost);
	//cudaMemcpy(divr,divrg,sizeof(float)*np,cudaMemcpyDeviceToHost);
  for(unsigned p=0;p<np;p++){
    pos[p]=ToTFloat3(TDouble3(pxy[p].x,pxy[p].y,pz[p]));
  }

  //-Creates VTK file.
  JFormatFiles2::StScalarData fields[20];
  unsigned nfields=0;
  if(idph){  fields[nfields]=JFormatFiles2::DefineField("Id"  ,JFormatFiles2::UInt32 ,1,idph); nfields++; }
  if(vel){   fields[nfields]=JFormatFiles2::DefineField("Vel" ,JFormatFiles2::Double64,3,vel);  nfields++; }
  //if(rhop){  fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,rhop); nfields++; }
	if(pres){  fields[nfields]=JFormatFiles2::DefineField("Pressure",JFormatFiles2::Double64,1,pres); nfields++; }
	//if(divr){  fields[nfields]=JFormatFiles2::DefineField("Divr",JFormatFiles2::Float32,1,divr); nfields++; }
  string file=Log->GetDirOut()+fun::FileNameSec(fname,fnum);
  JFormatFiles2::SaveVtk(file,np,pos,nfields,fields);

  //-Frees memory.
  delete[] pxy;
  delete[] pz;
  delete[] pos;
  delete[] idph;
  delete[] vel;
  //delete[] rhop;
	delete[] pres;
	//delete[]divr;
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
  TmgStop(Timers,TMG_SuSavePart);
}

//==============================================================================
/// Muestra y graba resumen final de ejecucion.
/// Displays and stores final summary of the execution.
//==============================================================================
void JSphGpuSingle::FinishRun(bool stop){
  float tsim=TimerSim.GetElapsedTimeF()/1000.f,ttot=TimerTot.GetElapsedTimeF()/1000.f;
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
/// Initial advection
//==============================================================================
void JSphGpuSingle::InitAdvection(double dt){
    const char met[]="SolvePPE";
    PosxyPreg=ArraysGpu->ReserveDouble2();
    PoszPreg=ArraysGpu->ReserveDouble();
    VelrhopPreg=ArraysGpu->ReserveFloat4();
    VelocityPre=ArraysGpu->ReserveDouble3();
    unsigned np=Np;
    unsigned npb=Npb;
    unsigned npf=np-npb;
    //-Cambia datos a variables Pre para calcular nuevos datos.
    //-Changes data of predictor variables for calculating the new data
    cudaMemcpy(PosxyPreg,Posxyg,sizeof(double2)*np,cudaMemcpyDeviceToDevice);     //Es decir... PosxyPre[] <= Posxy[] //i.e. PosxyPre[] <= Posxy[]
    cudaMemcpy(PoszPreg,Poszg,sizeof(double)*np,cudaMemcpyDeviceToDevice);        //Es decir... PoszPre[] <= Posz[] //i.e. PoszPre[] <= Posz[]
		cudaMemcpy(VelrhopPreg,Velrhopg,sizeof(float4)*np,cudaMemcpyDeviceToDevice); //Es decir... VelrhopPre[] <= Velrhop[] //i.e. VelrhopPre[] <= Velrhop[]
		cudaMemcpy(VelocityPre,Velocity,sizeof(double3)*np,cudaMemcpyDeviceToDevice);

    double2 *movxyg=ArraysGpu->ReserveDouble2();  cudaMemset(movxyg,0,sizeof(double2)*np);
    double *movzg=ArraysGpu->ReserveDouble();     cudaMemset(movzg,0,sizeof(double)*np);
    
    cusph::ComputeRStar(BlockSizes.forcesfluid,WithFloating,npf,npb,VelocityPre,dt,Codeg,movxyg,movzg);
		cusph::Moveparticles(BlockSizes.forcesfluid,np,npb,PosxyPreg,PoszPreg,movxyg,movzg,Posxyg,Poszg);
	  //cusph::ComputeStepPos2(BlockSizes.forcesfluid,PeriActive,WithFloating,np,npb,PosxyPreg,PoszPreg,movxyg,movzg,Posxyg,Poszg,Dcellg,Codeg);

    ArraysGpu->Free(movxyg);   movxyg=NULL;
    ArraysGpu->Free(movzg);    movzg=NULL; 
    CheckCudaError(met,"Initial Advection");
}

//==============================================================================
/// PPE Solver
//==============================================================================
#include <fstream>
#include <sstream>
#include <iomanip>
void JSphGpuSingle::SolvePPE(double dt){ 
  const char met[]="SolvePPE";
	TmgStart(Timers,TMG_Stage2a);
  tuint3 ncells=CellDivSingle->GetNcells();
  const int2 *begincell=CellDivSingle->GetBeginCell();
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();
  
  const unsigned *dcell=Dcellg;
  const unsigned np=Np;
  const unsigned npb=Npb;
  const unsigned npbok=NpbOk;
  const unsigned npf=np-npb;
	const unsigned PPEDim=npf;
  const unsigned bsbound=BlockSizes.forcesbound;
  const unsigned bsfluid=BlockSizes.forcesfluid;
	unsigned Nnz=0;
	unsigned Numfreesurface=0;

	//Create matrix
  bg=ArraysGpu->ReserveDouble(); cudaMemset(bg,0,sizeof(double)*PPEDim);

  MatrixASetup(np,npb,npbok,PPEDim,rowIndg,Divrg,FreeSurface,Nnz,Numfreesurface);	

	if(Nnz>MatrixMemory*np) RunException(met,fun::PrintStr("MatrixMemory too small"));
	CheckCudaError(met,"Nnz");

  cudaMemset(colIndg,0,sizeof(int)*Nnz);
	cudaMemset(ag,0,sizeof(double)*Nnz);

  cusph::PopulateMatrix(TKernel,Schwaiger,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Gravity,Posxyg,Poszg,Velocity,dWxCorrg,dWyCorrg,dWzCorrg,ag,bg,rowIndg,colIndg,Idpg,Divrg,Codeg,FreeSurface,MirrorPosg,MirrorCellg,MLSg,dt,sumFrg,taog,BoundaryFS,Pressureg,PistonPosX,PistonVel,RightWall);
	
	/*std::cout<<Nnz<<"\n";
		unsigned *rowInd=new unsigned[PPEDim+1]; cudaMemcpy(rowInd,rowIndg,sizeof(unsigned)*(PPEDim+1),cudaMemcpyDeviceToHost);
		unsigned *colInd=new unsigned[Nnz]; cudaMemcpy(colInd,colIndg,sizeof(unsigned)*Nnz,cudaMemcpyDeviceToHost);
		double *b=new double[PPEDim]; cudaMemcpy(b,bg,sizeof(double)*PPEDim,cudaMemcpyDeviceToHost);
		double *a=new double[Nnz]; cudaMemcpy(a,ag,sizeof(double)*Nnz,cudaMemcpyDeviceToHost);
		unsigned *Idpc=new unsigned[Np]; cudaMemcpy(Idpc,Idpg,sizeof(unsigned)*Np,cudaMemcpyDeviceToHost);
	
		ofstream FileOutput;
			string TimeFile;
		
			ostringstream TimeNum;
			TimeNum << count;
			ostringstream FileNum;
			FileNum << count;

			TimeFile =  "CPU Fluid Properties_" + FileNum.str() + ", T = " + TimeNum.str() + ".txt";

			FileOutput.open(TimeFile.c_str());

		for(int i=npb;i<np;i++){
			FileOutput << fixed << setprecision(20) <<"particle "<< Idpc[i] << "\t Order " << (i-npb) << "\t b " << b[(i-npb)] << "\n";
			for(int j=rowInd[(i-npb)];j<rowInd[(i-npb)+1];j++){
				FileOutput << fixed << setprecision(16) << j << "\t" << a[j] << "\t" << colInd[j] << "\t"<<Idpc[colInd[j]+npb]<<"\n";
			}
		}
		FileOutput.close();
		
		delete[] rowInd; delete[] colInd; delete[] b; delete[]a; delete[] Idpc;
	
	count++;*/

	if(PeriActive){
		//CellDivSingle->MatrixMirrorDCellSingle(bsbound,bsfluid,npf,npb,npbok,Posxyg,Poszg,Codeg,Idpg,rowIndg,colIndg,DomRealPosMin,DomRealPosMax,DomPosMin,Scell,DomCellCode,PeriActive,MapRealPosMin,MapRealSize,PeriXinc,PeriYinc,PeriZinc);
		//cusph::PopulatePeriodic(CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Posxyg,Poszg,a,rowInd,colInd,Idpg,Codeg,MirrorCellg);
	}

	CheckCudaError(met,"Matrix Setup");

	//cusph::FreeSurfaceMark(bsbound,bsfluid,np,npb,npbok,Divrg,ag,bg,rowIndg,Codeg,PI,FreeSurface,ShiftOffset);
  CheckCudaError(met,"FreeSurfaceMark");
	TmgStop(Timers,TMG_Stage2a);
	TmgStart(Timers,TMG_Stage2b);
  //cusph::solveVienna(TPrecond,TAMGInter,Tolerance,Iterations,Restart,StrongConnection,JacobiWeight,Presmooth,Postsmooth,CoarseCutoff,CoarseLevels,ag,Pressureg,bg,rowIndg,colIndg,Nnz,PPEDim,Numfreesurface); 
	cusph::PreBiCGSTAB(Tolerance,Iterations,ag,Pressureg,bg,rowIndg,colIndg,Nnz,PPEDim);
	CheckCudaError(met,"Matrix Solve");

  cusph::PressureAssign(bsbound,bsfluid,np,npb,npbok,Gravity,Poszg,Velrhopg,Pressureg,Idpg,Codeg,NegativePressureBound,MirrorPosg,bg);
	
  CheckCudaError(met,"Pressure assign");
  
  ArraysGpu->Free(bg);             bg=NULL;
	TmgStop(Timers,TMG_Stage2b);
}

//==============================================================================
/// Shifting
//==============================================================================
void JSphGpuSingle::RunShifting(double dt){ 
  const char met[]="Shifting";
  const unsigned np=Np;
  const unsigned npb=Npb;
  const unsigned npbok=NpbOk;
  const unsigned npf=np-npb;

  PosxyPreg=ArraysGpu->ReserveDouble2();
  PoszPreg=ArraysGpu->ReserveDouble();
  VelrhopPreg=ArraysGpu->ReserveFloat4();
	VelocityPre=ArraysGpu->ReserveDouble3();

	Divrg=ArraysGpu->ReserveFloat();
	cudaMemset(Divrg,0,sizeof(float)*np);
  cudaMemset(ShiftPosg,0,sizeof(double3)*npf);
	cudaMemset(Tensileg,0,sizeof(double3)*npf);
	cudaMemset(dWxCorrg,0,sizeof(double3)*npf);
	cudaMemset(dWyCorrg,0,sizeof(double3)*npf); 
	cudaMemset(dWzCorrg,0,sizeof(double3)*npf);
	cudaMemset(MLSg,0,sizeof(double4)*npb);
  cudaMemset(Aceg,0,sizeof(double3)*npf);

  //-Cambia datos a variables Pre para calcular nuevos datos.
  //-Changes data of predictor variables for calculating the new data
  cudaMemcpy(PosxyPreg,Posxyg,sizeof(double2)*np,cudaMemcpyDeviceToDevice);     //Es decir... PosxyPre[] <= Posxy[] //i.e. PosxyPre[] <= Posxy[]
  cudaMemcpy(PoszPreg,Poszg,sizeof(double)*np,cudaMemcpyDeviceToDevice);        //Es decir... PoszPre[] <= Posz[] //i.e. PoszPre[] <= Posz[]
  //cudaMemcpy(VelrhopPreg,Velrhopg,sizeof(float4)*np,cudaMemcpyDeviceToDevice); //Es decir... VelrhopPre[] <= Velrhop[] //i.e. VelrhopPre[] <= Velrhop[]
	cudaMemcpy(VelocityPre,Velocity,sizeof(double3)*np,cudaMemcpyDeviceToDevice);

  RunCellDivide(true);

  TmgStart(Timers,TMG_SuShifting);
  tuint3 ncells=CellDivSingle->GetNcells();
  const int2 *begincell=CellDivSingle->GetBeginCell();
  tuint3 cellmin=CellDivSingle->GetCellDomainMin();

  const unsigned *dcell=Dcellg;
  const unsigned bsbound=BlockSizes.forcesbound;
  const unsigned bsfluid=BlockSizes.forcesfluid;

  cusph::Interaction_Shifting(TKernel,TSlipCond,Simulate2D,WithFloating,UseDEM,CellMode,Visco*ViscoBoundFactor,Visco,bsfluid,bsbound,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetBeginCell(),CellDivSingle->GetCellDomainMin(),Dcellg,Posxyg,Poszg,Velocity,Codeg,FtoMasspg,TShifting,ShiftPosg,Divrg,TensileN,TensileR,Tensileg,FreeSurface,BoundaryFS,Idpg,MirrorPosg,MirrorCellg,dWxCorrg,dWyCorrg,dWzCorrg,MLSg,rowIndg,PistonPosX,PistonVel,RightWall,Gravity,Pressureg);

  CheckCudaError(met,"Failed in calculating concentration");

  JSphGpu::RunShifting(dt);
  TmgStop(Timers,TMG_SuShifting);
  CheckCudaError(met,"Failed in calculating shifting distance");
	const bool wavegen=(WaveGen? true:false);
	cusph::CorrectShiftVelocity(wavegen,TKernel,CellMode,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,Posxyg,Poszg,Velocity,dWxCorrg,dWyCorrg,dWzCorrg,Idpg,Divrg,Codeg,BoundaryFS,ShiftPosg,Aceg,DampingPointX,DampingLengthX,PistonPosX,PistonVel,RightWall,Gravity,Pressureg);
	Shift(dt,bsfluid);
	//cusph::ResetBoundVel(Npb,bsbound,Velocity,VelocityPre);
  ArraysGpu->Free(PosxyPreg);     PosxyPreg=NULL;
  ArraysGpu->Free(PoszPreg);      PoszPreg=NULL;
  ArraysGpu->Free(VelrhopPreg);   VelrhopPreg=NULL;
	ArraysGpu->Free(VelocityPre);   VelocityPre=NULL;
  ArraysGpu->Free(Divrg);         Divrg=NULL;
}

