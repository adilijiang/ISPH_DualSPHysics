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

#ifndef _WITHGPU
  #define VIENNACL_WITH_OPENMP
  #include "viennacl/vector.hpp"
  #include "viennacl/compressed_matrix.hpp"
  #include "viennacl/linalg/bicgstab.hpp"
  #include "viennacl/linalg//jacobi_precond.hpp"
  #include "viennacl/linalg/norm_2.hpp"
  #include "viennacl/tools/matrix_generation.hpp"

  #include "viennacl/linalg/amg.hpp" 
  #include "viennacl/tools/timer.hpp"
  #include "viennacl/io/matrix_market.hpp"
#endif

#include "JSphCpu.h"
#include "JCellDivCpu.h"
#include "JPartFloatBi4.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JArraysCpu.h"
#include "JSphDtFixed.h"
#include "JWaveGen.h"
#include "JXml.h"
#include "JSaveDt.h"
#include "JSphVarAcc.h"

#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#ifdef _WITHOMP
  #include <omp.h>  //Activate tb in Properties config -> C/C++ -> Language -> OpenMp

#else
  #define omp_get_thread_num() 0
  #define omp_get_max_threads() 1
#endif

using namespace std;


//==============================================================================
// Constructor.
//==============================================================================
JSphCpu::JSphCpu(bool withmpi):JSph(true,withmpi){
  ClassName="JSphCpu";
  CellDiv=NULL;
  ArraysCpu=new JArraysCpu;
  InitVars();
  TmcCreation(Timers,false);
}

//==============================================================================
// Destructor.
//==============================================================================
JSphCpu::~JSphCpu(){
  FreeCpuMemoryParticles();
  FreeCpuMemoryFixed();
  delete ArraysCpu;
  TmcDestruction(Timers);
}

//==============================================================================
// Initialization of variables.
//==============================================================================
void JSphCpu::InitVars(){
  RunMode="";
  OmpThreads=1;

  Np=Npb=NpbOk=0;
  NpbPer=NpfPer=0;
  WithFloating=false;

  Idpc=NULL; Codec=NULL; Dcellc=NULL; Posc=NULL; Velrhopc=NULL;
  PosPrec=NULL; VelrhopPrec=NULL; //-Symplectic
  Acec=NULL; Deltac=NULL; Divr=NULL; POrder=NULL; PPEDim=0;
  dWxCorr=NULL; dWzCorr=NULL;
  ShiftPosc=NULL; 
  RidpMove=NULL; 
  FtRidp=NULL;
  FtoForces=NULL;
  Irelationc=NULL;
  a.clear();
  b.clear();
  x.clear();
  colInd.clear();
  rowInd.clear();
  FreeCpuMemoryParticles();
  FreeCpuMemoryFixed();
}

//==============================================================================
/// Libera memoria fija en cpu para moving y floating.
/// Release fixed memory on CPU for moving and floating bodies.
//==============================================================================
void JSphCpu::FreeCpuMemoryFixed(){
  MemCpuFixed=0;
  delete[] RidpMove;  RidpMove=NULL;
  delete[] FtRidp;    FtRidp=NULL;
  delete[] FtoForces; FtoForces=NULL;
  delete[] Irelationc; Irelationc=NULL; 
}

//==============================================================================
/// Allocates memory for arrays with fixed size (motion and floating bodies).
//==============================================================================
void JSphCpu::AllocCpuMemoryFixed(){
  MemCpuFixed=0;

  try{
    Irelationc=new unsigned[Npb]; MemCpuFixed+=(sizeof(unsigned)*Npb);
    //-Allocates memory for moving objects.
    if(CaseNmoving){
      RidpMove=new unsigned[CaseNmoving];  MemCpuFixed+=(sizeof(unsigned)*CaseNmoving);
    }
    //-Allocates memory for floating bodies.
    if(CaseNfloat){
      FtRidp=new unsigned[CaseNfloat];     MemCpuFixed+=(sizeof(unsigned)*CaseNfloat);
      FtoForces=new StFtoForces[FtCount];  MemCpuFixed+=(sizeof(StFtoForces)*FtCount);
    }
  }
  catch(const std::bad_alloc){
    RunException("AllocMemoryFixed","Could not allocate the requested memory.");
  }
}

//==============================================================================
/// Libera memoria en cpu para particulas.
/// Release memory in CPU for particles.
//==============================================================================
void JSphCpu::FreeCpuMemoryParticles(){
  CpuParticlesSize=0;
  MemCpuParticles=0;
  ArraysCpu->Reset();
}

//==============================================================================
/// Reserva memoria en Cpu para las particulas. 
/// Reserve memory on CPU for the particles. 
//==============================================================================
void JSphCpu::AllocCpuMemoryParticles(unsigned np,float over){
  const char* met="AllocCpuMemoryParticles";
  FreeCpuMemoryParticles();
  //-Calculate number of partices with reserved memory / Calcula numero de particulas para las que se reserva memoria.
  const unsigned np2=(over>0? unsigned(over*np): np);
  CpuParticlesSize=np2;
  //-Calculate which arrays / Calcula cuantos arrays.
  ArraysCpu->SetArraySize(np2);
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_2B,2);  ///<-code
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_4B,6);  ///<-idp,dcell,POrder,POrderOld
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_8B,1);  ///<-divr
  //if(TDeltaSph==DELTA_DynamicExt)ArraysCpu->AddArrayCount(JArraysCpu::SIZE_4B,1);  ///<-delta
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_12B,1); ///<-ace
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_16B,1); ///<-velrhop
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_24B,5); ///<-pos, dWxCorr, dWyCorr, dWzCorr
  if(TStep==STEP_Symplectic){
    ArraysCpu->AddArrayCount(JArraysCpu::SIZE_24B,1); ///<-pospre
    ArraysCpu->AddArrayCount(JArraysCpu::SIZE_16B,1); ///<-velrhoppre
  }
  if(TShifting!=SHIFT_None){
    ArraysCpu->AddArrayCount(JArraysCpu::SIZE_12B,1); ///<-shiftpos
  }
  //-Show reserved memory / Muestra la memoria reservada.
  MemCpuParticles=ArraysCpu->GetAllocMemoryCpu();
  PrintSizeNp(np2,MemCpuParticles);
}

//==============================================================================
/// Resizes space in CPU memory for particles.
//==============================================================================
void JSphCpu::ResizeCpuMemoryParticles(unsigned npnew){
  //-Saves current data from CPU.
  unsigned    *idp       =SaveArrayCpu(Np,Idpc);
  word        *code      =SaveArrayCpu(Np,Codec);
  unsigned    *dcell     =SaveArrayCpu(Np,Dcellc);
  tdouble3    *pos       =SaveArrayCpu(Np,Posc);
  tfloat4     *velrhop   =SaveArrayCpu(Np,Velrhopc);
  tdouble3    *pospre    =SaveArrayCpu(Np,PosPrec);
  tfloat4     *velrhoppre=SaveArrayCpu(Np,VelrhopPrec);

  //-Frees pointers.
  ArraysCpu->Free(Idpc);
  ArraysCpu->Free(Codec);
  ArraysCpu->Free(Dcellc);
  ArraysCpu->Free(Posc);
  ArraysCpu->Free(Velrhopc);
  ArraysCpu->Free(PosPrec);
  ArraysCpu->Free(VelrhopPrec);
  //-Resizes CPU memory allocation.
  const double mbparticle=(double(MemCpuParticles)/(1024*1024))/CpuParticlesSize; //-MB por particula.
  Log->Printf("**JSphCpu: Requesting gpu memory for %u particles: %.1f MB.",npnew,mbparticle*npnew);
  ArraysCpu->SetArraySize(npnew);
  //-Reserve pointers.
  Idpc    =ArraysCpu->ReserveUint();
  Codec   =ArraysCpu->ReserveWord();
  Dcellc  =ArraysCpu->ReserveUint();
  Posc    =ArraysCpu->ReserveDouble3();
  Velrhopc=ArraysCpu->ReserveFloat4();
  if(pospre)    PosPrec    =ArraysCpu->ReserveDouble3();
  if(velrhoppre)VelrhopPrec=ArraysCpu->ReserveFloat4();
  //-Restore data in CPU memory.
  RestoreArrayCpu(Np,idp,Idpc);
  RestoreArrayCpu(Np,code,Codec);
  RestoreArrayCpu(Np,dcell,Dcellc);
  RestoreArrayCpu(Np,pos,Posc);
  RestoreArrayCpu(Np,velrhop,Velrhopc);
  RestoreArrayCpu(Np,pospre,PosPrec);
  RestoreArrayCpu(Np,velrhoppre,VelrhopPrec);
  //-Updates values.
  CpuParticlesSize=npnew;
  MemCpuParticles=ArraysCpu->GetAllocMemoryCpu();
}


//==============================================================================
/// Saves a CPU array in CPU memory. 
//==============================================================================
template<class T> T* JSphCpu::TSaveArrayCpu(unsigned np,const T *datasrc)const{
  T *data=NULL;
  if(datasrc){
    try{
      data=new T[np];
    }
    catch(const std::bad_alloc){
      RunException("TSaveArrayCpu","Could not allocate the requested memory.");
    }
    memcpy(data,datasrc,sizeof(T)*np);
  }
  return(data);
}

//==============================================================================
/// Restores a GPU array from CPU memory. 
//==============================================================================
template<class T> void JSphCpu::TRestoreArrayCpu(unsigned np,T *data,T *datanew)const{
  if(data&&datanew)memcpy(datanew,data,sizeof(T)*np);
  delete[] data;
}
//==============================================================================
void JSphCpu::RestoreArrayCpu_Uint(unsigned np,unsigned *data,unsigned *datanew)const{
  if(data&&datanew)memcpy(datanew,data,sizeof(unsigned)*np);
  delete[] data;
}

//==============================================================================
/// Arrays para datos basicos de las particulas. 
/// Arrays for basic particle data. 
//==============================================================================
void JSphCpu::ReserveBasicArraysCpu(){
  Idpc=ArraysCpu->ReserveUint();
  Codec=ArraysCpu->ReserveWord();
  Dcellc=ArraysCpu->ReserveUint();
  Posc=ArraysCpu->ReserveDouble3();
  Velrhopc=ArraysCpu->ReserveFloat4();
}

//==============================================================================
/// Devuelve la memoria reservada en cpu.
/// Return memory reserved on CPU.
//==============================================================================
llong JSphCpu::GetAllocMemoryCpu()const{  
  llong s=JSph::GetAllocMemoryCpu();
  //Reserved in AllocCpuMemoryParticles() / Reservada en AllocCpuMemoryParticles()
  s+=MemCpuParticles;
  //Reserved in AllocCpuMemoryFixed() / Reservada en AllocCpuMemoryFixed()
  s+=MemCpuFixed;
  //Reserved in other objects / Reservada en otros objetos
  return(s);
}

//==============================================================================
/// Visualiza la memoria reservada
/// Visualize the reserved memory
//==============================================================================
void JSphCpu::PrintAllocMemory(llong mcpu)const{
  Log->Printf("Allocated memory in CPU: %lld (%.2f MB)",mcpu,double(mcpu)/(1024*1024));
}

//==============================================================================
/// (ES):
/// Recupera datos de un rango de particulas y devuelve el numero de particulas que
/// sera menor que n si se eliminaron las periodicas.
/// - cellorderdecode: Reordena componentes de pos y vel segun CellOrder.
/// - onlynormal: Solo se queda con las normales, elimina las particulas periodicas.
/// (EN):
/// Collect data from a range of particles and return the number of particles that 
/// will be less than n and eliminate the periodic ones
/// - cellorderdecode: Reorder components of position (pos) and velocity (vel) according to CellOrder.
/// - onlynormal: Only keep the normal ones and eliminate the periodic particles.
//==============================================================================
unsigned JSphCpu::GetParticlesData(unsigned n,unsigned pini,bool cellorderdecode,bool onlynormal
  ,unsigned *idp,tdouble3 *pos,tfloat3 *vel,float *rhop,word *code)
{
  const char met[]="GetParticlesData";
  unsigned num=n;
  //-Copy selected values / Copia datos seleccionados.
  if(code)memcpy(code,Codec+pini,sizeof(word)*n);
  if(idp)memcpy(idp,Idpc+pini,sizeof(unsigned)*n);
  if(pos)memcpy(pos,Posc+pini,sizeof(tdouble3)*n);
  if(vel && rhop){
    for(unsigned p=0;p<n;p++){
      tfloat4 vr=Velrhopc[p+pini];
      vel[p]=TFloat3(vr.x,vr.y,vr.z);
      rhop[p]=vr.w;
    }
  }
  else{
    if(vel) for(unsigned p=0;p<n;p++){ tfloat4 vr=Velrhopc[p+pini]; vel[p]=TFloat3(vr.x,vr.y,vr.z); }
    if(rhop)for(unsigned p=0;p<n;p++)rhop[p]=Velrhopc[p+pini].w;
  }
  //-Eliminate non-normal particles (periodic & others) / Elimina particulas no normales (periodicas y otras).
  if(onlynormal){
    if(!idp || !pos || !vel || !rhop)RunException(met,"Pointers without data.");
    word *code2=code;
    if(!code2){
      code2=ArraysCpu->ReserveWord();
      memcpy(code2,Codec+pini,sizeof(word)*n);
    }
    unsigned ndel=0;
    for(unsigned p=0;p<n;p++){
      bool normal=(CODE_GetSpecialValue(code2[p])==CODE_NORMAL);
      if(ndel && normal){
        const unsigned pdel=p-ndel;
        idp[pdel]  =idp[p];
        pos[pdel]  =pos[p];
        vel[pdel]  =vel[p];
        rhop[pdel] =rhop[p];
        code2[pdel]=code2[p];
      }
      if(!normal)ndel++;
    }
    num-=ndel;
    if(!code)ArraysCpu->Free(code2);
  }
  //-Reorder components in their original order / Reordena componentes en su orden original.
  if(cellorderdecode)DecodeCellOrder(n,pos,vel);
  return(num);
}

//==============================================================================
/// Carga la configuracion de ejecucion con OpenMP.
/// Load the execution configuration with OpenMP.
//==============================================================================
void JSphCpu::ConfigOmp(const JCfgRun *cfg){
#ifdef _WITHOMP
  //-Determine number of threads for host with OpenMP / Determina numero de threads por host con OpenMP
  if(Cpu && cfg->OmpThreads!=1){
    OmpThreads=cfg->OmpThreads;
    if(OmpThreads<=0)OmpThreads=max(omp_get_num_procs(),1);
    if(OmpThreads>MAXTHREADS_OMP)OmpThreads=MAXTHREADS_OMP;
    omp_set_num_threads(OmpThreads);
    Log->Printf("Threads by host for parallel execution: %d",omp_get_max_threads());
  }
  else{
    OmpThreads=1;
    omp_set_num_threads(OmpThreads);
  }
#else
  OmpThreads=1;
#endif
}

//==============================================================================
/// Configura modo de ejecucion en CPU.
/// Configure execution mode on CPU.
//==============================================================================
void JSphCpu::ConfigRunMode(const JCfgRun *cfg,std::string preinfo){
  #ifndef WIN32
    const int len=128; char hname[len];
    gethostname(hname,len);
    if(!preinfo.empty())preinfo=preinfo+", ";
    preinfo=preinfo+"HostName:"+hname;
  #endif
  Hardware="Cpu";
  if(OmpThreads==1)RunMode="Single core";
  else RunMode=string("OpenMP(Threads:")+fun::IntStr(OmpThreads)+")";
  if(!preinfo.empty())RunMode=preinfo+", "+RunMode;
  if(Stable)RunMode=string("Stable, ")+RunMode;
  else RunMode=string("Pos-Double, ")+RunMode;
  Log->Print(fun::VarStr("RunMode",RunMode));
}

//==============================================================================
/// Inicializa vectores y variables para la ejecucion.
/// Initialize vectors and variables for execution.
//==============================================================================
void JSphCpu::InitRun(){
  const char met[]="InitRun";
  WithFloating=(CaseNfloat>0);
  if(TStep==STEP_Symplectic)DtPre=DtIni;
  if(UseDEM)DemDtForce=DtIni; //(DEM)
  if(CaseNfloat)InitFloating();

  //-Adjust paramaters to start.
  PartIni=PartBeginFirst;
  TimeStepIni=(!PartIni? 0: PartBeginTimeStep);
  //-Adjust motion for the instant of the loaded PART.
  if(CaseNmoving){
    MotionTimeMod=(!PartIni? PartBeginTimeStep: 0);
    Motion->ProcesTime(0,TimeStepIni+MotionTimeMod);
  }

  //-Uses Inlet information from PART read.
  if(PartBeginTimeStep && PartBeginTotalNp){
    TotalNp=PartBeginTotalNp;
    IdMax=unsigned(TotalNp-1);
  }

  //-Prepares WaveGen configuration.
  if(WaveGen){
    Log->Printf("\nWave paddles configuration:");
    WaveGen->Init(TimeMax,Gravity,Simulate2D,CellOrder,MassFluid,Dp,Dosh,Scell,Hdiv,DomPosMin,DomRealPosMin,DomRealPosMax);
    WaveGen->VisuConfig(""," ");
  }

  //-Process Special configurations in XML.
  JXml xml; xml.LoadFile(FileXml);
  //-Configuration of SaveDt.
  if(xml.GetNode("case.execution.special.savedt",false)){
    SaveDt=new JSaveDt(Log);
    SaveDt->Config(&xml,"case.execution.special.savedt",TimeMax,TimePart);
    SaveDt->VisuConfig("\nSaveDt configuration:"," ");
  }

  Part=PartIni; Nstep=0; PartNstep=0; PartOut=0;
  TimeStep=TimeStepIni; TimeStepM1=TimeStep;
  if(DtFixed)DtIni=DtFixed->GetDt(TimeStep,DtIni);
  if(TimersStep)TimersStep->SetInitialTime(float(TimeStep));
}

//==============================================================================
/// Adds variable acceleration from input files.
//==============================================================================
void JSphCpu::AddVarAcc(){
  for(unsigned c=0;c<VarAcc->GetCount();c++){
    unsigned mkfluid;
    tdouble3 acclin,accang,centre,velang,vellin;
    bool setgravity;
    VarAcc->GetAccValues(c,TimeStep,mkfluid,acclin,accang,centre,velang,vellin,setgravity);
    const bool withaccang=(accang.x!=0||accang.y!=0||accang.z!=0);
    const word codesel=word(mkfluid);
    const int npb=int(Npb),np=int(Np);
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int p=npb;p<np;p++){//-Iterates through the fluid particles.
      //-Checks if the current particle is part of the particle set by its MK.
      if(CODE_GetTypeValue(Codec[p])==codesel){
        tdouble3 acc=ToTDouble3(Acec[p]);
        acc=acc+acclin;                             //-Adds linear acceleration.
        if(!setgravity)acc=acc-ToTDouble3(Gravity); //-Subtract global gravity from the acceleration if it is set in the input file
        if(withaccang){                             //-Adds angular acceleration.
          const tdouble3 dc=Posc[p]-centre;
          const tdouble3 vel=TDouble3(Velrhopc[p].x-vellin.x,Velrhopc[p].y-vellin.y,Velrhopc[p].z-vellin.z);//-Get the current particle's velocity

          //-Calculate angular acceleration ((Dw/Dt) x (r_i - r)) + (w x (w x (r_i - r))) + (2w x (v_i - v))
          //(Dw/Dt) x (r_i - r) (term1)
          acc.x+=(accang.y*dc.z)-(accang.z*dc.y);
          acc.y+=(accang.z*dc.x)-(accang.x*dc.z);
          acc.z+=(accang.x*dc.y)-(accang.y*dc.x);

          //Centripetal acceleration (term2)
          //First find w x (r_i - r))
          const double innerx=(velang.y*dc.z)-(velang.z*dc.y);
          const double innery=(velang.z*dc.x)-(velang.x*dc.z);
          const double innerz=(velang.x*dc.y)-(velang.y*dc.x);
          //Find w x inner
          acc.x+=(velang.y*innerz)-(velang.z*innery);
          acc.y+=(velang.z*innerx)-(velang.x*innerz);
          acc.z+=(velang.x*innery)-(velang.y*innerx);

          //Coriolis acceleration 2w x (v_i - v) (term3)
          acc.x+=((2.0*velang.y)*vel.z)-((2.0*velang.z)*vel.y);
          acc.y+=((2.0*velang.z)*vel.x)-((2.0*velang.x)*vel.z);
          acc.z+=((2.0*velang.x)*vel.y)-((2.0*velang.y)*vel.x);
        }
        //-Stores the new acceleration value.
        Acec[p]=ToTFloat3(acc);
      }
    }
  }
}

//==============================================================================
/// Prepara variables para interaccion "INTER_Forces" o "INTER_ForcesCorr".
/// Prepare variables for interaction functions "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphCpu::PreInteractionVars_Forces(TpInter tinter,unsigned np,unsigned npb){
  //-Initialize Arrays / Inicializa arrays.
  const unsigned npf=np-npb;
  if(Deltac)memset(Deltac,0,sizeof(float)*np);                       //Deltac[]=0
 
  memset(Acec,0,sizeof(tfloat3)*np);

  //-Apply the extra forces to the correct particle sets.
  if(VarAcc)AddVarAcc();

  //-Prepare values of rhop for interaction / Prepara datos derivados de rhop para interaccion.
  /*const int n=int(np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(n>LIMIT_PREINTERACTION_OMP)
  #endif
  for(int p=0;p<n;p++){
    const float rhop=Velrhopc[p].w,rhop_r0=rhop/RhopZero;
    Pressc[p]=CteB*(pow(rhop_r0,Gamma)-1.0f);
  }*/
}

//==============================================================================
/// Prepara variables para interaccion "INTER_Forces" o "INTER_ForcesCorr".
/// Prepare variables for interaction functions "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphCpu::PreInteraction_Forces(TpInter tinter){
  TmcStart(Timers,TMC_CfPreForces);
	const unsigned np=Np;
	const unsigned npb=Npb;
  if(tinter==1){
    //-Assign memory to variables Pre / Asigna memoria a variables Pre.
    PosPrec=ArraysCpu->ReserveDouble3();
    VelrhopPrec=ArraysCpu->ReserveFloat4();
    //-Change data to variables Pre to calculate new data / Cambia datos a variables Pre para calcular nuevos datos.
   #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int i=0;i<int(np);i++){
      PosPrec[i]=Posc[i];
      VelrhopPrec[i]=Velrhopc[i]; //Put value of Velrhop[] in VelrhopPre[] / Es decir... VelrhopPre[] <= Velrhop[]
    }
		
		dWxCorr=ArraysCpu->ReserveDouble3(); memset(dWxCorr,0,sizeof(tdouble3)*np);
		dWyCorr=ArraysCpu->ReserveDouble3(); memset(dWyCorr,0,sizeof(tdouble3)*np);
		dWzCorr=ArraysCpu->ReserveDouble3(); memset(dWzCorr,0,sizeof(tdouble3)*np);

		Divr=ArraysCpu->ReserveFloat(); memset(Divr,0,sizeof(float)*np);
		//Matrix Order and Free Surface
  }
  //-Assign memory / Asigna memoria.
  Acec=ArraysCpu->ReserveFloat3();

  if(tinter==1){//Initial Advection
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int p=int(npb);p<int(np);p++){
 	    const tdouble3 pos=PosPrec[p];
	    const tfloat4 v=VelrhopPrec[p];
	    Posc[p].x=pos.x+DtPre*v.x;
	    Posc[p].y=pos.y+DtPre*v.y;
	    Posc[p].z=pos.z+DtPre*v.z;
    }
  }
  
  //-Initialize Arrays / Inicializa arrays.
  PreInteractionVars_Forces(tinter,np,npb);

  //-Calcula VelMax: Se incluyen las particulas floatings y no afecta el uso de condiciones periodicas.
  //-Calculate VelMax: Floating object particles are included and do not affect use of periodic condition.
  const unsigned pini=(DtAllParticles? 0: npb);
  float velmax=0;
  for(unsigned p=pini;p<np;p++){
    const tfloat4 v=Velrhopc[p];
    const float v2=v.x*v.x+v.y*v.y+v.z*v.z;
    velmax=max(velmax,v2);
  }
  VelMax=sqrt(velmax);
  ViscDtMax=0;
  TmcStop(Timers,TMC_CfPreForces);
}

//==============================================================================
/// Libera memoria asignada de ArraysGpu.
/// Free memory assigned to ArraysGpu.
//==============================================================================
void JSphCpu::PosInteraction_Forces(TpInter tinter){
  //-Free memory assinged in PreInteraction_Forces() / Libera memoria asignada en PreInteraction_Forces().
  ArraysCpu->Free(Acec);         Acec=NULL;
  if(tinter==2){
	  ArraysCpu->Free(Divr);		 Divr=NULL;
    ArraysCpu->Free(POrder);    POrder=NULL;
	  ArraysCpu->Free(dWxCorr);	 dWxCorr=NULL;
    ArraysCpu->Free(dWyCorr);	 dWyCorr=NULL;
	  ArraysCpu->Free(dWzCorr);	 dWzCorr=NULL;
  }
}

//==============================================================================
/// Devuelve valores de kernel gradients: frx, fry y frz.
/// Return values of kernel gradients: frx, fry and frz.
//==============================================================================
void JSphCpu::GetKernel(float rr2,float drx,float dry,float drz,float &frx,float &fry,float &frz)const{
  const float rad=sqrt(rr2);
  const float qq=rad/H;
  //-Wendland kernel
  const float wqq1=1.f-0.5f*qq;
  const float fac=Bwen*qq*wqq1*wqq1*wqq1/rad;
  /*float fac;
  if(qq<1.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f)-75.0f*powf(1.0f-qq,4.0f));
  else if(qq<2.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f));
  else if(qq<3.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f));
  else fac=0;

  fac=fac/rad;*/
  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

void JSphCpu::GetKernelDouble(double rr2,double drx,double dry,double drz,double &frx,double &fry,double &frz)const{
  const double rad=sqrt(rr2);
  const double qq=rad/H;
  //-Wendland kernel
  const double wqq1=1.0-0.5*qq;
  const double fac=Bwen*qq*wqq1*wqq1*wqq1/rad;
  /*double fac;

  if(qq<1.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0)+30.0*pow(2.0-qq,4.0)-75.0*pow(1.0-qq,4.0));
  else if(qq<2.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0)+30.0*powf(2.0-qq,4.0));
  else if(qq<3.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0));
  else fac=0;

  fac=fac/rad;*/
  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

//==============================================================================
/// Devuelve valores de kernel: Wab = W(q) con q=r/H.
/// Return values of kernel: Wab = W(q) where q=r/H.
//==============================================================================
float JSphCpu::GetKernelWab(float rr2)const{
  const float qq=sqrt(rr2)/H;
  //-Wendland kernel.
  /*float wab;

  if(qq<1.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f)+15.0f*powf(1.0f-qq,5.0f));
  else if(qq<2.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f));
  else if(qq<3.0f)wab=Awen*(powf(3.0f-qq,5.0f));
  else wab=0;
  return(wab);*/
  const float wqq=2.f*qq+1.f;
  const float wqq1=1.f-0.5f*qq;

  const float wqq2=wqq1*wqq1;
  return(Awen*wqq*wqq2*wqq2);
}

//==============================================================================
/// Devuelve limites de celdas para interaccion.
/// Return cell limits for interaction.
//==============================================================================
void JSphCpu::GetInteractionCells(unsigned rcell
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
/// Slip Conditions and Boundary interactions
//=============================================================================

void JSphCpu::Boundary_Velocity(TpSlipCond TSlipCond,unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,tfloat4 *velrhop,const word *code,float *divr,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr)const{

  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++){
		unsigned codep1=CODE_GetTypeValue(Codec[p1]);
		if(codep1!=1){
			//-Obtain data of particle p1 / Obtiene datos de particula p1.
			const tdouble3 posp1=pos[p1];
		
			tfloat3 Sum1=TFloat3(0);
			float Sum2=0.0;
			float divrp1=0.0;
			tfloat3 wallVelocity=TFloat3(0);

			tdouble3 dwxp1=TDouble3(0); tdouble3 dwyp1=TDouble3(0); tdouble3 dwzp1=TDouble3(0);

			//-Obtain interaction limits / Obtiene limites de interaccion
			int cxini,cxfin,yini,yfin,zini,zfin;
			GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

			//-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
			for(int z=zini;z<zfin;z++){
				const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
				for(int y=yini;y<yfin;y++){
					int ymod=zmod+nc.x*y;
					const unsigned pini=beginendcell[cxini+ymod];
					const unsigned pfin=beginendcell[cxfin+ymod];

					//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
					float massp2=MassFluid; //-Contiene masa de particula segun sea bound o fluid.
					const float volume=massp2/RhopZero; //Volume of particle j
		
					//-Interactions
					//------------------------------------------------
					for(unsigned p2=pini;p2<pfin;p2++){
						const double drx=posp1.x-pos[p2].x;
						const double dry=posp1.y-pos[p2].y;
						const double drz=posp1.z-pos[p2].z;
						const double rr2=drx*drx+dry*dry+drz*drz;
						if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
							//-Wendland kernel.
							double frx,fry,frz;
							GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);
							const float rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
							divrp1-=volume*rDivW;

							dwxp1.x-=volume*frx*drx; dwxp1.y-=volume*fry*drx; dwxp1.z-=volume*frz*drx;
							dwyp1.x-=volume*frx*dry; dwyp1.y-=volume*fry*dry; dwyp1.z-=volume*frz*dry;
							dwzp1.x-=volume*frx*drz; dwzp1.y-=volume*fry*drz; dwzp1.z-=volume*frz*drz;

							if(codep1<10&&TSlipCond){
								const float W=GetKernelWab(rr2);
								Sum1.x+=W*velrhop[p2].x;
								Sum1.y+=W*velrhop[p2].y;
								Sum1.z+=W*velrhop[p2].z;
								Sum2+=W;
							}
						}
					}
				}
			}

			for(int z=zini;z<zfin;z++){
				const int zmod=(nc.w)*z; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
				for(int y=yini;y<yfin;y++){
					int ymod=zmod+nc.x*y;
					const unsigned pini=beginendcell[cxini+ymod];
					const unsigned pfin=beginendcell[cxfin+ymod];

					//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
					float massp2=MassFluid; //-Contiene masa de particula segun sea bound o fluid.
					const float volume=massp2/RhopZero; //Volume of particle j
		
					//-Interactions
					//------------------------------------------------
					for(unsigned p2=pini;p2<pfin;p2++){
						const double drx=posp1.x-pos[p2].x;
						const double dry=posp1.y-pos[p2].y;
						const double drz=posp1.z-pos[p2].z;
						const double rr2=drx*drx+dry*dry+drz*drz;
						if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
							if(CODE_GetTypeValue(Codec[p2])==codep1+10){
								wallVelocity.x=velrhop[p2].x;
								wallVelocity.y=velrhop[p2].y;
								wallVelocity.z=velrhop[p2].z;
							}

							//-Wendland kernel.
							double frx,fry,frz;
							GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);
							const float rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
							divrp1-=volume*rDivW;
						
							dwxp1.x-=volume*frx*drx; dwxp1.y-=volume*fry*drx; dwxp1.z-=volume*frz*drx;
							dwyp1.x-=volume*frx*dry; dwyp1.y-=volume*fry*dry; dwyp1.z-=volume*frz*dry;
							dwzp1.x-=volume*frx*drz; dwzp1.y-=volume*fry*drz; dwzp1.z-=volume*frz*drz;
						}
					}
				}
			}
			
			if(Sum2){
				if(TSlipCond==SLIPCOND_Slip){
					velrhop[p1].x=(wallVelocity.x-(Sum1.x/Sum2))/2.0f;
					velrhop[p1].y=(wallVelocity.y-(Sum1.y/Sum2))/2.0f;
					velrhop[p1].z=(wallVelocity.z-(Sum1.z/Sum2))/2.0f;
				}
				else if(TSlipCond==SLIPCOND_NoSlip){
					velrhop[p1].x=(wallVelocity.x+(Sum1.x/Sum2))/2.0f;
					velrhop[p1].y=(wallVelocity.y+(Sum1.y/Sum2))/2.0f;
					velrhop[p1].z=(wallVelocity.z+(Sum1.z/Sum2))/2.0f;
				}
			}

			if(divrp1) divr[p1]=divrp1;
			//if(Idpc[p1]==3130||Idpc[p1]==3337||Idpc[p1]==3544||Idpc[p1]==8467)divr[p1]=-1;
			if(dwxp1.x||dwxp1.y||dwxp1.z
				||dwyp1.x||dwyp1.y||dwyp1.z
				||dwzp1.x||dwzp1.y||dwzp1.z){
			
				dwxcorr[p1]=dwxcorr[p1]+dwxp1;
				dwycorr[p1]=dwycorr[p1]+dwyp1;
				dwzcorr[p1]=dwzcorr[p1]+dwzp1;
			}
		}
  }
}

//==============================================================================
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template<TpFtMode ftmode> void JSphCpu::InteractionForcesFluid
  (TpInter tinter, unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,float visco
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const tdouble3 *pos,const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code,const unsigned *idp
  ,tfloat3 *ace,float *divr)const
{
  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  //-Initialize viscth to calculate viscdt maximo con OpenMP / Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[MAXTHREADS_OMP*STRIDE_OMP];
  for(int th=0;th<OmpThreads;th++)viscth[th*STRIDE_OMP]=0;
  //-Initial execution with OpenMP / Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++){
    //float visc=0,arp1=0,deltap1=0;
		float divrp1=0;
    tfloat3 acep1=TFloat3(0);
		tdouble3 dwxp1=TDouble3(0); tdouble3 dwyp1=TDouble3(0); tdouble3 dwzp1=TDouble3(0);

    //-Obtain data of particle p1 in case of floating objects / Obtiene datos de particula p1 en caso de existir floatings.
    /*bool ftp1=false;     //-Indicate if it is floating / Indica si es floating.
    float ftmassp1=1.f;  //-Contains floating particle mass or 1.0f if it is fluid / Contiene masa de particula floating o 1.0f si es fluid.
    if(USE_FLOATING){
      ftp1=(CODE_GetType(code[p1])==CODE_TYPE_FLOATING);
      if(ftp1)ftmassp1=FtObjs[CODE_GetTypeValue(code[p1])].massp;
      //if(ftp1 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
      //if(ftp1 && shift)shiftposp1.x=FLT_MAX;  //-For floating objects do not calculate shifting / Para floatings no se calcula shifting.
    }*/
    
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    //const float rhopp1=velrhop[p1].w;
    const tdouble3 posp1=pos[p1];
    const float pressp1=velrhop[p1].w;
    
    //-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
    //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
    for(int z=zini;z<zfin;z++){
      const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        const unsigned pini=beginendcell[cxini+ymod];
        const unsigned pfin=beginendcell[cxfin+ymod];

        //-Interaction of Fluid with type Fluid or Bound / Interaccion de Fluid con varias Fluid o Bound.
        //------------------------------------------------
        for(unsigned p2=pini;p2<pfin;p2++){
          const double drx=posp1.x-pos[p2].x;
          const double dry=posp1.y-pos[p2].y;
          const double drz=posp1.z-pos[p2].z;
          const double rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
            //-Wendland kernel.
            double frx,fry,frz;
            GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);
			
            //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
            float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			      const float volumep2=massp2/RhopZero; //Volume of particle j
            //bool ftp2=false;    //-Indicate if it is floating / Indica si es floating.
            bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound /  Se desactiva cuando se usa DEM y es float-float o float-bound.
            /*if(USE_FLOATING){
              ftp2=(CODE_GetType(code[p2])==CODE_TYPE_FLOATING);
              if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
              //if(ftp2 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
              //if(ftp2 && shift && tshifting==SHIFT_NoBound)shiftposp1.x=FLT_MAX; //-With floating objects do not use shifting / Con floatings anula shifting.
              //compute=!(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound / Se desactiva cuando se usa DEM y es float-float o float-bound.
            }*/

            //===== Acceleration from viscous forces && FreeSurface detection && KernelGradientCorrection===== 
            if(compute && tinter==1){
			        const float rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
			        const float temp=volumep2*2.0f*visco*rDivW/(rr2+Eta2);
			        const float dvx=velp1.x-velrhop[p2].x, dvy=velp1.y-velrhop[p2].y, dvz=velp1.z-velrhop[p2].z;
              acep1.x+=temp*dvx; acep1.y+=temp*dvy; acep1.z+=temp*dvz;

							divrp1-=volumep2*rDivW;

							dwxp1.x-=volumep2*frx*drx; dwxp1.y-=volumep2*fry*drx; dwxp1.z-=volumep2*frz*drx;
							dwyp1.x-=volumep2*frx*dry; dwyp1.y-=volumep2*fry*dry; dwyp1.z-=volumep2*frz*dry;
							dwzp1.x-=volumep2*frx*drz; dwzp1.y-=volumep2*fry*drz; dwzp1.z-=volumep2*frz*drz;
            }

			      //===== Acceleration from pressure gradient ===== 
            if(compute && tinter==2){
			        const float temp_x=frx*dwxcorr[p1].x+fry*dwycorr[p1].x+frz*dwzcorr[p1].x;
              const float temp_y=frx*dwxcorr[p1].y+fry*dwycorr[p1].y+frz*dwzcorr[p1].y;
			        const float temp_z=frx*dwxcorr[p1].z+fry*dwycorr[p1].z+frz*dwzcorr[p1].z;
			        const float temp=volumep2*(velrhop[p2].w-pressp1);
              acep1.x+=temp*temp_x; acep1.y+=temp*temp_y; acep1.z+=temp*temp_z;
			      }
          }
        }
      }
    }
	
    //-Sum results together / Almacena resultados.
    if(acep1.x||acep1.y||acep1.z||divrp1
			||dwxp1.x||dwxp1.y||dwxp1.z
			||dwyp1.x||dwyp1.y||dwyp1.z
			||dwzp1.x||dwzp1.y||dwzp1.z){
			if(tinter==1){ 
				ace[p1]=ace[p1]+acep1; 
				divr[p1]+=divrp1;
				dwxcorr[p1]=dwxcorr[p1]+dwxp1;
				dwycorr[p1]=dwycorr[p1]+dwyp1;
				dwzcorr[p1]=dwzcorr[p1]+dwzp1;
			}
	    if(tinter==2) ace[p1]=ace[p1]+acep1/RhopZero;
    }
  }
  //-Keep max value in viscdt / Guarda en viscdt el valor maximo.
  //for(int th=0;th<OmpThreads;th++)if(viscdt<viscth[th*STRIDE_OMP])viscdt=viscth[th*STRIDE_OMP];
}

//==============================================================================
/// Realiza interaccion DEM entre particulas Floating-Bound & Floating-Floating //(DEM)
/// Perform DEM interaction between particles Floating-Bound & Floating-Floating //(DEM)
//==============================================================================
template<bool psimple> void JSphCpu::InteractionForcesDEM
  (unsigned nfloat,tint4 nc,int hdiv,unsigned cellfluid
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const unsigned *ftridp,const StDemData* demobjs
  ,const tdouble3 *pos,const tfloat3 *pspos,const tfloat4 *velrhop,const word *code,const unsigned *idp
  ,float &viscdt,tfloat3 *ace)const
{
  //-Initialize demdtth to calculate max demdt with OpenMP / Inicializa demdtth para calcular demdt maximo con OpenMP.
  float demdtth[MAXTHREADS_OMP*STRIDE_OMP];
  for(int th=0;th<OmpThreads;th++)demdtth[th*STRIDE_OMP]=-FLT_MAX;
  //-Initial execution with OpenMP / Inicia ejecucion con OpenMP.
  const int nft=int(nfloat);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int cf=0;cf<nft;cf++){
    const unsigned p1=ftridp[cf];
    if(p1!=UINT_MAX){
      float demdtp1=0;
      tfloat3 acep1=TFloat3(0);

      //-Get data of particle p1 / Obtiene datos de particula p1.
      const tfloat3 psposp1=(psimple? pspos[p1]: TFloat3(0));
      const tdouble3 posp1=(psimple? TDouble3(0): pos[p1]);
      const word tavp1=CODE_GetTypeAndValue(code[p1]);
      const float masstotp1=demobjs[tavp1].mass;
      const float taup1=demobjs[tavp1].tau;
      const float kfricp1=demobjs[tavp1].kfric;
      const float restitup1=demobjs[tavp1].restitu;

      //-Get interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Search for neighbours in adjacent cells (first bound and then fluid+floating) / Busqueda de vecinos en celdas adyacentes (primero bound y despues fluid+floating).
      for(unsigned cellinitial=0;cellinitial<=cellfluid;cellinitial+=cellfluid){
        for(int z=zini;z<zfin;z++){
          const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            const unsigned pini=beginendcell[cxini+ymod];
            const unsigned pfin=beginendcell[cxfin+ymod];

            //-Interaction of Floating Object particles with type Fluid or Bound / Interaccion de Floating con varias Fluid o Bound.
            //------------------------------------------------
            for(unsigned p2=pini;p2<pfin;p2++)if(CODE_GetType(code[p2])!=CODE_TYPE_FLUID && tavp1!=CODE_GetTypeAndValue(code[p2])){
              const float drx=(psimple? psposp1.x-pspos[p2].x: float(posp1.x-pos[p2].x));
              const float dry=(psimple? psposp1.y-pspos[p2].y: float(posp1.y-pos[p2].y));
              const float drz=(psimple? psposp1.z-pspos[p2].z: float(posp1.z-pos[p2].z));
              const float rr2=drx*drx+dry*dry+drz*drz;
              const float rad=sqrt(rr2);

              //-Calculate max value of demdt / Calcula valor maximo de demdt.
              const word tavp2=CODE_GetTypeAndValue(code[p2]);
              const float masstotp2=demobjs[tavp2].mass;
              const float taup2=demobjs[tavp2].tau;
              const float kfricp2=demobjs[tavp2].kfric;
              const float restitup2=demobjs[tavp2].restitu;
              //const StDemData *demp2=demobjs+CODE_GetTypeAndValue(code[p2]);

              const float nu_mass=(!cellinitial? masstotp1/2: masstotp1*masstotp2/(masstotp1+masstotp2)); //-Con boundary toma la propia masa del floating 1.
              const float kn=4/(3*(taup1+taup2))*sqrt(float(Dp)/4); //generalized rigidity - Lemieux 2008
              const float demvisc=float(PI)/(sqrt( kn/nu_mass ))*40.f;              
              if(demdtp1<demvisc)demdtp1=demvisc;

              const float over_lap=1.0f*float(Dp)-rad; //-(ri+rj)-|dij|
              if(over_lap>0.0f){ //-Contact
                const float dvx=velrhop[p1].x-velrhop[p2].x, dvy=velrhop[p1].y-velrhop[p2].y, dvz=velrhop[p1].z-velrhop[p2].z; //vji
                const float nx=drx/rad, ny=dry/rad, nz=drz/rad; //normal_ji               
                const float vn=dvx*nx+dvy*ny+dvz*nz; //vji.nji      
                //normal
                const float eij=(restitup1+restitup2)/2;
                const float gn=-(2.0f*log(eij)*sqrt(nu_mass*kn))/(sqrt(float(PI)+log(eij)*log(eij))); //generalized damping - Cummins 2010
                //const float gn=0.08f*sqrt(nu_mass*sqrt(float(Dp)/2)/((taup1+taup2)/2)); //generalized damping - Lemieux 2008
                float rep=kn*pow(over_lap,1.5f);
                float fn=rep-gn*pow(over_lap,0.25f)*vn;                   
                acep1.x+=(fn*nx); acep1.y+=(fn*ny); acep1.z+=(fn*nz); //-Force is applied in the normal between the particles
                //tangential
                float dvxt=dvx-vn*nx, dvyt=dvy-vn*ny, dvzt=dvz-vn*nz; //Vji_t
                float vt=sqrt(dvxt*dvxt + dvyt*dvyt + dvzt*dvzt);
                float tx=0, ty=0, tz=0; //Tang vel unit vector
                if(vt!=0){ tx=dvxt/vt; ty=dvyt/vt; tz=dvzt/vt; }
                float ft_elast=2*(kn*float(DemDtForce)-gn)*vt/7;   //Elastic frictional string -->  ft_elast=2*(kn*fdispl-gn*vt)/7; fdispl=dtforce*vt;
                const float kfric_ij=(kfricp1+kfricp2)/2;
                float ft=kfric_ij*fn*tanh(8*vt);  //Coulomb
                ft=(ft<ft_elast? ft: ft_elast);   //not above yield criteria, visco-elastic model
                acep1.x+=(ft*tx); acep1.y+=(ft*ty); acep1.z+=(ft*tz);
              } 
            }
          }
        }
      }
      //-Sum results together / Almacena resultados.
      if(acep1.x||acep1.y||acep1.z){
        ace[p1]=ace[p1]+acep1;
        const int th=omp_get_thread_num();
        if(demdtth[th*STRIDE_OMP]<demdtp1)demdtth[th*STRIDE_OMP]=demdtp1;
      }
    }
  }
  //-Update viscdt with max value of viscdt or demdt* / Actualiza viscdt con el valor maximo de viscdt y demdt*.
  float demdt=demdtth[0];
  for(int th=1;th<OmpThreads;th++)if(demdt<demdtth[th*STRIDE_OMP])demdt=demdtth[th*STRIDE_OMP];
  if(viscdt<demdt)viscdt=demdt;
}

//==============================================================================
/// Seleccion de parametros template para Interaction_ForcesFluidT.
/// Selection of template parameters for Interaction_ForcesFluidT.
//==============================================================================
template<TpFtMode ftmode> void JSphCpu::Interaction_ForcesT
  (TpInter tinter, unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
  ,const tdouble3 *pos,const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code,const unsigned *idp
  ,tfloat3 *ace,float *divr)const
{
  const unsigned npf=np-npb;
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);

  if(npf){
		Boundary_Velocity(TSlipCond,NpbOk,0,nc,hdiv,cellfluid,begincell,cellzero,Dcellc,Posc,Velrhopc,Codec,divr,dwxcorr,dwycorr,dwzcorr);
    //-Interaction Fluid-Fluid / Interaccion Fluid-Fluid
    InteractionForcesFluid<ftmode> (tinter,npf,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr);

	  //-Interaction Fluid-Bound / Interaccion Fluid-Bound
    InteractionForcesFluid<ftmode> (tinter,npf,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr);

    //-Interaction of DEM Floating-Bound & Floating-Floating / Interaccion DEM Floating-Bound & Floating-Floating //(DEM)
    //if(USE_DEM)InteractionForcesDEM<psimple> (CaseNfloat,nc,hdiv,cellfluid,begincell,cellzero,dcell,FtRidp,DemObjs,pos,pspos,velrhop,code,idp,viscdt,ace);
  
		if(Simulate2D){
			JSphCpu::InverseCorrection(npf,npb,dWxCorr,dWzCorr);
			JSphCpu::InverseCorrection(npbok,0,dWxCorr,dWzCorr);
		}
		else{
			JSphCpu::InverseCorrection3D(npf,npb,dWxCorr,dWyCorr,dWzCorr);
			JSphCpu::InverseCorrection3D(npbok,0,dWxCorr,dWyCorr,dWzCorr);
		}
	}
 /* if(npbok){
    //-Interaction of type Bound-Fluid / Interaccion Bound-Fluid
    InteractionForcesBound      <psimple,ftmode> (npbok,0,nc,hdiv,cellfluid,begincell,cellzero,dcell,pos,pspos,velrhop,code,idp,viscdt,ar);
  }*/
}

//==============================================================================
/// Seleccion de parametros template para Interaction_ForcesX.
/// Selection of template parameters for Interaction_ForcesX.
//==============================================================================
void JSphCpu::Interaction_Forces(TpInter tinter,unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
  ,const tdouble3 *pos,const tfloat4 *velrhop,const unsigned *idp,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code
  ,tfloat3 *ace,float *divr)const
{
  if(!WithFloating) Interaction_ForcesT<FTMODE_None> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr);
  else if(!UseDEM)  Interaction_ForcesT<FTMODE_Sph> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr);
  else              Interaction_ForcesT<FTMODE_Dem> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr);
}

//==============================================================================
/// (ES):
/// Actualiza pos, dcell y code a partir del desplazamiento indicado.
/// El valor de outrhop indica si esta fuera de los limites de densidad.
/// Comprueba los limites en funcion de MapRealPosMin y MapRealSize esto es valido
/// para single-cpu pq DomRealPos y MapRealPos son iguales. Para multi-cpu seria 
/// necesario marcar las particulas q salgan del dominio sin salir del mapa.
/// (EN):
/// Update pos, dcell and code to move with indicated displacement.
/// The value of outrhop indicates is it outside of the density limits.
/// Check the limits in funcion of MapRealPosMin & MapRealSize that this is valid
/// for single-cpu because DomRealPos & MapRealPos are equal. For multi-cpu it will be 
/// necessary to mark the particles that leave the domain without leaving the map.
//==============================================================================
void JSphCpu::UpdatePos(tdouble3 rpos,double movx,double movy,double movz
  ,bool outrhop,unsigned p,tdouble3 *pos,unsigned *cell,word *code)const
{
  //-Check validity of displacement / Comprueba validez del desplazamiento.
  bool outmove=(fabs(float(movx))>MovLimit || fabs(float(movy))>MovLimit || fabs(float(movz))>MovLimit);
  //-Aplica desplazamiento.
  rpos.x+=movx; rpos.y+=movy; rpos.z+=movz;
  //-Check limits of real domain / Comprueba limites del dominio reales.
  double dx=rpos.x-MapRealPosMin.x;
  double dy=rpos.y-MapRealPosMin.y;
  double dz=rpos.z-MapRealPosMin.z;
  bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=MapRealSize.x || dy>=MapRealSize.y || dz>=MapRealSize.z);
  //-Adjust position according to periodic conditions and compare domain limits / Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
  if(PeriActive && out){
    bool xperi=((PeriActive&1)!=0),yperi=((PeriActive&2)!=0),zperi=((PeriActive&4)!=0);
    if(xperi){
      if(dx<0)             { dx-=PeriXinc.x; dy-=PeriXinc.y; dz-=PeriXinc.z; }
      if(dx>=MapRealSize.x){ dx+=PeriXinc.x; dy+=PeriXinc.y; dz+=PeriXinc.z; }
    }
    if(yperi){
      if(dy<0)             { dx-=PeriYinc.x; dy-=PeriYinc.y; dz-=PeriYinc.z; }
      if(dy>=MapRealSize.y){ dx+=PeriYinc.x; dy+=PeriYinc.y; dz+=PeriYinc.z; }
    }
    if(zperi){
      if(dz<0)             { dx-=PeriZinc.x; dy-=PeriZinc.y; dz-=PeriZinc.z; }
      if(dz>=MapRealSize.z){ dx+=PeriZinc.x; dy+=PeriZinc.y; dz+=PeriZinc.z; }
    }
    bool outx=!xperi && (dx<0 || dx>=MapRealSize.x);
    bool outy=!yperi && (dy<0 || dy>=MapRealSize.y);
    bool outz=!zperi && (dz<0 || dz>=MapRealSize.z);
    out=(outx||outy||outz);
    rpos=TDouble3(dx,dy,dz)+MapRealPosMin;
  }
  //-Keep currnt position / Guarda posicion actualizada.
  pos[p]=rpos;
  //-Keep cell and check / Guarda celda y check.
  if(outrhop || outmove || out){//-Particle out
    word rcode=code[p];
    if(outrhop)rcode=CODE_SetOutRhop(rcode);
    else if(out)rcode=CODE_SetOutPos(rcode);
    else rcode=CODE_SetOutMove(rcode);
    code[p]=rcode;
    cell[p]=0xFFFFFFFF;
  }
  else{//-Particle in
    if(PeriActive){
      dx=rpos.x-DomPosMin.x;
      dy=rpos.y-DomPosMin.y;
      dz=rpos.z-DomPosMin.z;
    }
    unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
    cell[p]=PC__Cell(DomCellCode,cx,cy,cz);
  }
}

//==============================================================================
/// Actualizacion de particulas segun fuerzas y dt usando Symplectic-Predictor.
/// Update of particles according to forces and dt using Symplectic-Predictor.
//==============================================================================
void JSphCpu::ComputeSymplecticPre(double dt){
  if(TShifting)ComputeSymplecticPreT<true> (dt);
  else         ComputeSymplecticPreT<false>(dt);
}
//==============================================================================
template<bool shift> void JSphCpu::ComputeSymplecticPreT(double dt){
  TmcStart(Timers,TMC_SuComputeStep);
  //-Calculate new values of particles / Calcula nuevos datos de particulas.
  //const double dt05=dt*.5;
  //-Calculate new density for boundary and copy velocity / Calcula nueva densidad para el contorno y copia velocidad.
  const int npb=int(Npb);
  /*#ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(npb>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=0;p<npb;p++){
    const tfloat4 vr=VelrhopPrec[p];
    //const float rhopnew=float(double(vr.w)+dt05*Arc[p]);
    Velrhopc[p]=TFloat4(vr.x,vr.y,vr.z,(rhopnew<RhopZero? RhopZero: rhopnew));//-Avoid fluid particles being absorbed by boundary ones / Evita q las boundary absorvan a las fluidas.
  }*/

  //-Calculate new values of fluid / Calcula nuevos datos del fluido.
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=npb;p<np;p++){
    //-Calcula densidad.
    //const float rhopnew=float(double(VelrhopPrec[p].w)+dt05*Arc[p]);
    if(!WithFloating || CODE_GetType(Codec[p])==CODE_TYPE_FLUID){//-Fluid Particles / Particulas: Fluid
      /*//-Calculate displacement & update position / Calcula desplazamiento y actualiza posicion.
      double dx=double(VelrhopPrec[p].x)*dt05;
      double dy=double(VelrhopPrec[p].y)*dt05;
      double dz=double(VelrhopPrec[p].z)*dt05;
      if(shift){
        dx+=double(ShiftPosc[p].x);
        dy+=double(ShiftPosc[p].y);
        dz+=double(ShiftPosc[p].z);
      }
      bool outrhop=(rhopnew<RhopOutMin||rhopnew>RhopOutMax);
      UpdatePos(PosPrec[p],dx,dy,dz,outrhop,p,Posc,Dcellc,Codec);*/
      //-Update velocity & density / Actualiza velocidad y densidad.
      Velrhopc[p].x=float(double(VelrhopPrec[p].x)+double(Acec[p].x)* dt);
      Velrhopc[p].y=float(double(VelrhopPrec[p].y)+double(Acec[p].y)* dt);
      Velrhopc[p].z=float(double(VelrhopPrec[p].z)+double(Acec[p].z)* dt);
      //Velrhopc[p].w=rhopnew;
    }
    /*else{//-Floating Particles / Particulas: Floating
      Velrhopc[p]=VelrhopPrec[p];
      Velrhopc[p].w=(rhopnew<RhopZero? RhopZero: rhopnew); //-Avoid fluid particles being absorbed by floating ones / Evita q las floating absorvan a las fluidas.
      //-Copy position / Copia posicion.
      Posc[p]=PosPrec[p];
    }*/
  }

  //-Copy previous position of boundary / Copia posicion anterior del contorno.
  memcpy(Posc,PosPrec,sizeof(tdouble3)*Npb);
  TmcStop(Timers,TMC_SuComputeStep);
}

//==============================================================================
/// Actualizacion de particulas segun fuerzas y dt usando Symplectic-Corrector.
/// Update particles according to forces and dt using Symplectic-Corrector.
//==============================================================================
void JSphCpu::ComputeSymplecticCorr(double dt){
  if(TShifting)ComputeSymplecticCorrT<true> (dt);
  else         ComputeSymplecticCorrT<false>(dt);
}
//==============================================================================
template<bool shift> void JSphCpu::ComputeSymplecticCorrT(double dt){
  TmcStart(Timers,TMC_SuComputeStep);
  
  //-Calculate rhop of boudary and set velocity=0 / Calcula rhop de contorno y vel igual a cero.
  const int npb=int(Npb);
  /*#ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(npb>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=0;p<npb;p++){
    const double epsilon_rdot=(-double(Arc[p])/double(Velrhopc[p].w))*dt;
    const float rhopnew=float(double(VelrhopPrec[p].w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
    Velrhopc[p]=TFloat4(0,0,0,(rhopnew<RhopZero? RhopZero: rhopnew));//-Avoid fluid particles being absorbed by boundary ones / Evita q las boundary absorvan a las fluidas.
  }*/

  //-Calculate fluid values / Calcula datos de fluido.
  const double dt05=dt*0.5;
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=npb;p<np;p++){
    //const double epsilon_rdot=(-double(Arc[p])/double(Velrhopc[p].w))*dt;
    //const float rhopnew=float(double(VelrhopPrec[p].w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
    if(!WithFloating || CODE_GetType(Codec[p])==CODE_TYPE_FLUID){//-Particulas: Fluid
      //-Update velocity & density / Actualiza velocidad y densidad.
      Velrhopc[p].x-=float((Acec[p].x-Gravity.x)*dt); 
      Velrhopc[p].y-=float((Acec[p].y-Gravity.y)*dt);  
      Velrhopc[p].z-=float((Acec[p].z-Gravity.z)*dt);
      //Velrhopc[p].w=rhopnew;
      //-Calculate displacement and update position / Calcula desplazamiento y actualiza posicion.
      double dx=(double(VelrhopPrec[p].x)+double(Velrhopc[p].x))*dt05; 
      double dy=(double(VelrhopPrec[p].y)+double(Velrhopc[p].y))*dt05; 
      double dz=(double(VelrhopPrec[p].z)+double(Velrhopc[p].z))*dt05;
      bool outrhop=false;//(rhopnew<RhopOutMin||rhopnew>RhopOutMax);
      UpdatePos(PosPrec[p],dx,dy,dz,outrhop,p,Posc,Dcellc,Codec);
    }
    else{//-Floating Particles / Particulas: Floating
      Velrhopc[p]=VelrhopPrec[p];
      //Velrhopc[p].w=(rhopnew<RhopZero? RhopZero: rhopnew); //-Avoid fluid particles being absorbed by floating ones / Evita q las floating absorvan a las fluidas.
      //-Copy position / Copia posicion.
      Posc[p]=PosPrec[p];
    }
  }

  ArraysCpu->Free(PosPrec);      PosPrec=NULL;
  ArraysCpu->Free(VelrhopPrec);  VelrhopPrec=NULL;
  TmcStop(Timers,TMC_SuComputeStep);
}

//==============================================================================
/// Calcula un Dt variable.
/// Calculat variable Dt.
//==============================================================================
double JSphCpu::DtVariable(bool final){
  //-dt1 depends on force per unit mass.
  const double dt1=(AceMax? (sqrt(double(H)/AceMax)): DBL_MAX); 
  //-dt2 combines the Courant and the viscous time-step controls.
  const double dt2=double(H)/(max(Cs0,VelMax*10.)+double(H)*ViscDtMax);
  //-dt new value of time step.
  double dt=double(CFLnumber)*min(dt1,dt2);
  if(DtFixed)dt=DtFixed->GetDt(float(TimeStep),float(dt));
  if(dt<double(DtMin)){ dt=double(DtMin); DtModif++; }
  if(SaveDt && final)SaveDt->AddValues(TimeStep,dt,dt1*CFLnumber,dt2*CFLnumber,AceMax,ViscDtMax,VelMax);
  return(dt);
}

//==============================================================================
/// Calcula Shifting final para posicion de particulas.
/// Calculate final Shifting for particles' position.
//==============================================================================
void JSphCpu::RunShifting(double dt){
  TmcStart(Timers,TMC_SuShifting);
  const double coeftfs=(Simulate2D? 2.0: 3.0)-FreeSurface;
  const double ShiftOffset=0.2;
  const int pini=int(Npb),pfin=int(Np),npf=int(Np-Npb);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(npf>LIMIT_COMPUTELIGHT_OMP)
  #endif
  for(int p=pini;p<pfin;p++){
    tfloat3 rshiftpos=ShiftPosc[p];
    float divrp1=Divr[p];
    double umagn=-double(ShiftCoef)*double(H)*double(H);

 	  tfloat3 norm=TFloat3(-rshiftpos.x,-rshiftpos.y,-rshiftpos.z);
	  tfloat3 tang=TFloat3(0);
	  tfloat3 bitang=TFloat3(0);

	  //-tangent and bitangent calculation
	  tang.x=norm.z+norm.y;		
	  if(!Simulate2D)tang.y=-(norm.x+norm.z);	
	  tang.z=-norm.x+norm.y;
	  bitang.x=tang.y*norm.z-norm.y*tang.z;
	  if(!Simulate2D)bitang.y=norm.x*tang.z-tang.x*norm.z;
	  bitang.z=tang.x*norm.y-norm.x*tang.y;

	  //-unit normal vector
	  float temp=norm.x*norm.x+norm.y*norm.y+norm.z*norm.z;
	  temp=sqrt(temp);
	  norm.x=norm.x/temp; norm.y=norm.y/temp; norm.z=norm.z/temp;
	  if(!temp){norm.x=0.f; norm.y=0.f; norm.z=0.f;}

	  //-unit tangent vector
	  temp=tang.x*tang.x+tang.y*tang.y+tang.z*tang.z;
	  temp=sqrt(temp);
	  tang.x=tang.x/temp; tang.y=tang.y/temp; tang.z=tang.z/temp;
	  if(!temp){tang.x=0.f; tang.y=0.f; tang.z=0.f;}

	  //-unit bitangent vector
	  temp=bitang.x*bitang.x+bitang.y*bitang.y+bitang.z*bitang.z;
	  temp=sqrt(temp);
	  bitang.x=bitang.x/temp; bitang.y=bitang.y/temp; bitang.z=bitang.z/temp;
	  if(!temp){bitang.x=0.f; bitang.y=0.f; bitang.z=0.f;}

	  //-gradient calculation
	  float dcds=tang.x*rshiftpos.x+tang.z*rshiftpos.z+tang.y*rshiftpos.y;
	  float dcdn=norm.x*rshiftpos.x+norm.z*rshiftpos.z+norm.y*rshiftpos.y;
	  float dcdb=bitang.x*rshiftpos.x+bitang.z*rshiftpos.z+bitang.y*rshiftpos.y;

    if(divrp1<FreeSurface){
      rshiftpos.x=dcds*tang.x+dcdb*bitang.x;
      rshiftpos.y=dcds*tang.y+dcdb*bitang.y;
      rshiftpos.z=dcds*tang.z+dcdb*bitang.z;
    }
    else if(divrp1>=FreeSurface && divrp1<=FreeSurface+ShiftOffset){ 
      double FactorShift=0.5*(1-cos(PI*double(divrp1-FreeSurface)/0.2));
      rshiftpos.x=dcds*tang.x+dcdb*bitang.x+dcdn*norm.x*FactorShift;
      rshiftpos.y=dcds*tang.y+dcdb*bitang.y+dcdn*norm.y*FactorShift;
      rshiftpos.z=dcds*tang.z+dcdb*bitang.z+dcdn*norm.z*FactorShift;
    }

    rshiftpos.x=float(double(rshiftpos.x)*umagn);
    rshiftpos.y=float(double(rshiftpos.y)*umagn);
    rshiftpos.z=float(double(rshiftpos.z)*umagn);
    ShiftPosc[p]=rshiftpos; //particles in fluid bulk, normal shifting

    //Max Shifting
		if(TShifting==SHIFT_Max){
			double Maxx=abs(Velrhopc[p].x*dt);
			double Maxy=abs(Velrhopc[p].y*dt);
			double Maxz=abs(Velrhopc[p].z*dt);
			if(abs(ShiftPosc[p].x)>Maxx){
				if(ShiftPosc[p].x>0) ShiftPosc[p].x=Maxx;
				else ShiftPosc[p].x=-Maxx;
			}
			if(abs(ShiftPosc[p].z)>Maxz){
				if(ShiftPosc[p].z>0) ShiftPosc[p].z=Maxz;
				else ShiftPosc[p].z=-Maxz;
			}
			if(abs(ShiftPosc[p].y)>Maxy){
				if(ShiftPosc[p].y>0) ShiftPosc[p].y=Maxy;
				else ShiftPosc[p].y=-Maxy;
			}
		}
  }
  TmcStop(Timers,TMC_SuShifting);
}

//==============================================================================
/// (ES):
/// Calcula posicion de particulas segun idp[]. Cuando no la encuentra es UINT_MAX.
/// Cuando periactive es False supone que no hay particulas duplicadas (periodicas)
/// y todas son CODE_NORMAL.
/// (EN):
/// Calculate position of particles according to idp[]. When it is not met set as UINT_MAX.
/// When periactive is False assume that there are no duplicate particles (periodic ones)
/// and all are set as CODE_NORMAL.
//==============================================================================
void JSphCpu::CalcRidp(bool periactive,unsigned np,unsigned pini,unsigned idini,unsigned idfin,const word *code,const unsigned *idp,unsigned *ridp)const{
  //-Assign values UINT_MAX / Asigna valores UINT_MAX
  const unsigned nsel=idfin-idini;
  memset(ridp,255,sizeof(unsigned)*nsel); 

  //-Calculate position according to id / Calcula posicion segun id.
  const unsigned pfin=pini+np;
  if(periactive){//-Calculate position according to id checking that the particles are normal (i.e. not periodic) /Calcula posicion segun id comprobando que las particulas son normales (no periodicas).
    for(unsigned p=pini;p<pfin;p++){
      const unsigned id=idp[p];
      if(idini<=id && id<idfin){
        if(CODE_GetSpecialValue(code[p])==CODE_NORMAL)ridp[id-idini]=p;
      }
    }
  }
  else{//-Calculate position according to id assuming that all the particles are normal (i.e. not periodic) / Calcula posicion segun id suponiendo que todas las particulas son normales (no periodicas).
    for(unsigned p=pini;p<pfin;p++){
      const unsigned id=idp[p];
      if(idini<=id && id<idfin)ridp[id-idini]=p;
    }
  }
}

//==============================================================================
/// Aplica un movimiento lineal a un conjunto de particulas.
/// Apply a linear movement to a group of particles.
//==============================================================================
void JSphCpu::MoveLinBound(unsigned np,unsigned ini,const tdouble3 &mvpos,const tfloat3 &mvvel
  ,const unsigned *ridp,tdouble3 *pos,unsigned *dcell,tfloat4 *velrhop,word *code)const
{
  const unsigned fin=ini+np;
  for(unsigned id=ini;id<fin;id++){
    const unsigned pid=RidpMove[id];
    if(pid!=UINT_MAX){
      UpdatePos(pos[pid],mvpos.x,mvpos.y,mvpos.z,false,pid,pos,dcell,code);
      velrhop[pid].x=mvvel.x;  velrhop[pid].y=mvvel.y;  velrhop[pid].z=mvvel.z;
    }
  }
}

//==============================================================================
/// Aplica un movimiento lineal a un conjunto de particulas.
/// Apply a linear movement to a group of particles.
//==============================================================================
void JSphCpu::MoveMatBound(unsigned np,unsigned ini,tmatrix4d m,double dt
  ,const unsigned *ridpmv,tdouble3 *pos,unsigned *dcell,tfloat4 *velrhop,word *code)const
{
  const unsigned fin=ini+np;
  for(unsigned id=ini;id<fin;id++){
    const unsigned pid=RidpMove[id];
    if(pid!=UINT_MAX){
      tdouble3 ps=pos[pid];
      tdouble3 ps2=MatrixMulPoint(m,ps);
      if(Simulate2D)ps2.y=ps.y;
      const double dx=ps2.x-ps.x, dy=ps2.y-ps.y, dz=ps2.z-ps.z;
      UpdatePos(ps,dx,dy,dz,false,pid,pos,dcell,code);
      velrhop[pid].x=float(dx/dt);  velrhop[pid].y=float(dy/dt);  velrhop[pid].z=float(dz/dt);
    }
  }
}

//==============================================================================
/// Procesa movimiento de boundary particles
/// Process movement of boundary particles
//==============================================================================
void JSphCpu::RunMotion(double stepdt){
  const char met[]="RunMotion";
  TmcStart(Timers,TMC_SuMotion);

  unsigned nmove=0;
  if(Motion->ProcesTime(TimeStep+MotionTimeMod,stepdt)){
    nmove=Motion->GetMovCount();
    if(nmove){
      CalcRidp(PeriActive!=0,Npb,0,CaseNfixed,CaseNfixed+CaseNmoving,Codec,Idpc,RidpMove);
      //-Movement of  boundary particles / Movimiento de particulas boundary
      for(unsigned c=0;c<nmove;c++){
        unsigned ref;
        tdouble3 mvsimple;
        tmatrix4d mvmatrix;
        if(Motion->GetMov(c,ref,mvsimple,mvmatrix)){//-Single movement / Movimiento simple
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed,np=MotionObjBegin[ref+1]-MotionObjBegin[ref];
          mvsimple=OrderCode(mvsimple);
          if(Simulate2D)mvsimple.y=0;
          const tfloat3 mvvel=ToTFloat3(mvsimple/TDouble3(stepdt));
          MoveLinBound(np,pini,mvsimple,mvvel,RidpMove,Posc,Dcellc,Velrhopc,Codec);
        }
        else{//-Movement using a matrix / Movimiento con matriz
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed,np=MotionObjBegin[ref+1]-MotionObjBegin[ref];
          mvmatrix=OrderCode(mvmatrix);
          MoveMatBound(np,pini,mvmatrix,stepdt,RidpMove,Posc,Dcellc,Velrhopc,Codec);
        }
      }
      BoundChanged=true;
    }
  }
  //-Process other modes of motion / Procesa otros modos de motion.
  if(WaveGen){
    if(!nmove)CalcRidp(PeriActive!=0,Npb,0,CaseNfixed,CaseNfixed+CaseNmoving,Codec,Idpc,RidpMove);
    BoundChanged=true;
    //-Control of WaveGeneration (WaveGen) / Gestion de WaveGen.
    if(WaveGen)for(unsigned c=0;c<WaveGen->GetCount();c++){
      tdouble3 mvsimple;
      tmatrix4d mvmatrix;
      unsigned nparts;
      unsigned idbegin;
      if(WaveGen->GetMotion(c,TimeStep+MotionTimeMod,stepdt,mvsimple,mvmatrix,nparts,idbegin)){//-Movimiento simple
        mvsimple=OrderCode(mvsimple);
        if(Simulate2D)mvsimple.y=0;
        const tfloat3 mvvel=ToTFloat3(mvsimple/TDouble3(stepdt));
        MoveLinBound(nparts,idbegin-CaseNfixed,mvsimple,mvvel,RidpMove,Posc,Dcellc,Velrhopc,Codec);
      }
      else{
        mvmatrix=OrderCode(mvmatrix);
        MoveMatBound(nparts,idbegin-CaseNfixed,mvmatrix,stepdt,RidpMove,Posc,Dcellc,Velrhopc,Codec);
      }
    }
  }
  TmcStop(Timers,TMC_SuMotion);
}

//==============================================================================
/// Ajusta variables de particulas floating body
/// Adjust variables of floating body particles
//==============================================================================
void JSphCpu::InitFloating(){
  if(PartBegin){
    JPartFloatBi4Load ftdata;
    ftdata.LoadFile(PartBeginDir);
    //-Check cases of constant values / Comprueba coincidencia de datos constantes.
    for(unsigned cf=0;cf<FtCount;cf++)ftdata.CheckHeadData(cf,FtObjs[cf].mkbound,FtObjs[cf].begin,FtObjs[cf].count,FtObjs[cf].mass);
    //-Load PART data / Carga datos de PART.
    ftdata.LoadPart(PartBegin);
    for(unsigned cf=0;cf<FtCount;cf++){
      FtObjs[cf].center=OrderCodeValue(CellOrder,ftdata.GetPartCenter(cf));
      FtObjs[cf].fvel=OrderCodeValue(CellOrder,ftdata.GetPartFvel(cf));
      FtObjs[cf].fomega=OrderCodeValue(CellOrder,ftdata.GetPartFomega(cf));
      FtObjs[cf].radius=ftdata.GetHeadRadius(cf);
    }
    DemDtForce=ftdata.GetPartDemDtForce();
  }
}

//==============================================================================
/// Muestra los temporizadores activos.
/// Show active timers
//==============================================================================
void JSphCpu::ShowTimers(bool onlyfile){
  JLog2::TpMode_Out mode=(onlyfile? JLog2::Out_File: JLog2::Out_ScrFile);
  Log->Print("\n[CPU Timers]",mode);
  if(!SvTimers)Log->Print("none",mode);
  else for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c))Log->Print(TimerToText(c),mode);
}

//==============================================================================
/// Devuelve string con nombres y valores de los timers activos.
/// Return string with names and values of active timers.
//==============================================================================
void JSphCpu::GetTimersInfo(std::string &hinfo,std::string &dinfo)const{
  for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c)){
    hinfo=hinfo+";"+TimerGetName(c);
    dinfo=dinfo+";"+fun::FloatStr(TimerGetValue(c)/1000.f);
  }
}

//===============================================================================
///Find the closest fluid particle to each boundary particle
//===============================================================================
void JSphCpu::FindIrelation(unsigned n,unsigned pinit,const tdouble3 *pos,const unsigned *idpc,unsigned *irelation,const word *code)const{
  const int pfin=int(pinit+n);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(code[p1])==1){
    //-Load data of particle p1 / Carga datos de particula p1.
    const tdouble3 posp1=pos[p1];
    const unsigned idp1=idpc[p1];
    irelation[idp1]=n;
    float closestR=10*Fourh2;
    //-Interaction of boundary with type Fluid/Float / Interaccion de Bound con varias Fluid/Float.
    //----------------------------------------------
    for(int p2=int(pinit);p2<pfin;p2++){
      if(CODE_GetTypeValue(code[p2])==0){
        const float drx=float(posp1.x-pos[p2].x);
        const float dry=float(posp1.y-pos[p2].y);
        const float drz=float(posp1.z-pos[p2].z);
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<=closestR){
          closestR=rr2;
          irelation[idp1]=idpc[p2];
        }
      }
    }
  }
}

//===============================================================================
///Kernel Correction
//===============================================================================
void JSphCpu::InverseCorrection(unsigned n, unsigned pinit, tdouble3 *dwxcorr,tdouble3 *dwzcorr)const{

	const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
	for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(Codec[p1])!=1){
	  const double det=1.0/(dwxcorr[p1].x*dwzcorr[p1].z-dwxcorr[p1].z*dwzcorr[p1].x);
	
      if(det){
	    const double temp=dwxcorr[p1].x;
      dwxcorr[p1].x=dwzcorr[p1].z*det;
	    dwxcorr[p1].z=-dwxcorr[p1].z*det; 
	    dwzcorr[p1].x=-dwzcorr[p1].x*det;
	    dwzcorr[p1].z=temp*det;
	  }
	}
}

void JSphCpu::InverseCorrection3D(unsigned n, unsigned pinit,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr)const{

	const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
	for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(Codec[p1])!=1){
    tdouble3 dwx=dwxcorr[p1]; //  dwx.x   dwx.y   dwx.z
    tdouble3 dwy=dwycorr[p1]; //  dwy.x   dwy.y   dwy.z
    tdouble3 dwz=dwzcorr[p1]; //  dwz.x   dwz.y   dwz.z

    double det=dwx.x*(dwy.y*dwz.z-dwz.y*dwy.z) + dwx.y*(dwy.x*dwz.z-dwz.x*dwy.z)+dwx.z*(dwy.x*dwz.y-dwz.x*dwy.y);

    dwxcorr[p1].x=(dwy.y*dwz.z-dwz.y*dwy.z)/det;
    dwxcorr[p1].y=-(dwx.y*dwz.z-dwz.y*dwx.z)/det;
    dwxcorr[p1].z=(dwx.y*dwy.z-dwy.y*dwx.z)/det;
    dwycorr[p1].x=-(dwy.x*dwz.z-dwz.x*dwy.z)/det;
    dwycorr[p1].y=(dwx.x*dwz.z-dwz.x*dwx.z)/det;
    dwycorr[p1].z=-(dwx.x*dwy.z-dwy.x*dwx.z)/det;
    dwzcorr[p1].x=(dwy.x*dwz.y-dwz.x*dwy.y)/det;
    dwzcorr[p1].y=-(dwx.x*dwz.y-dwz.x*dwx.y)/det;
    dwzcorr[p1].z=(dwx.x*dwz.y-dwz.x*dwy.x)/det;
	}
}

//===============================================================================
///Matrix order for PPE
//===============================================================================
void JSphCpu::MatrixOrder(unsigned np,unsigned pinit,unsigned *porder,const unsigned *idpc,const unsigned *irelation,word *code, unsigned &ppedim){
	const int pfin=int(pinit+np);

  unsigned index=0;
	for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(code[p1])!=1){
    if(p1<int(Npb)&&p1>=int(NpbOk)) porder[p1]=np;
    else{
      porder[p1]=index; 
      index++;
    }
  }

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(code[p1])==1){
    porder[p1]=np;
    unsigned idp1=idpc[p1];//POSSIBLE BUG
    for(int p2=0;p2<int(Npb);p2++)if(irelation[idp1]==idpc[p2])porder[p1]=porder[p2];
  }

  ppedim=index;
}

//===============================================================================
///Find free surface
//===============================================================================
void JSphCpu::FreeSurfaceFind(bool psimple,unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,
	const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,const tdouble3 *pos,const tfloat3 *pspos,
	float *divr,const word *code)const{

  const int pfin=int(pinit+n);
  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++) if(CODE_GetTypeValue(code[p1])!=1){
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
	  const tfloat3 psposp1=(psimple? pspos[p1]: TFloat3(0));
    const tdouble3 posp1=(psimple? TDouble3(0): pos[p1]);

    //-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

    //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
    for(int z=zini;z<zfin;z++){
      const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        const unsigned pini=beginendcell[cxini+ymod];
        const unsigned pfin=beginendcell[cxfin+ymod];
        
        //-Interactions
        //------------------------------------------------
        for(unsigned p2=pini;p2<pfin;p2++){
          const float drx=(psimple? psposp1.x-pspos[p2].x: float(posp1.x-pos[p2].x));
          const float dry=(psimple? psposp1.y-pspos[p2].y: float(posp1.y-pos[p2].y));
          const float drz=(psimple? psposp1.z-pspos[p2].z: float(posp1.z-pos[p2].z));
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
			      //-Wendland kernel.
            float frx,fry,frz;
            GetKernel(rr2,drx,dry,drz,frx,fry,frz);
            
	          //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
            float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			      const float volume=massp2/RhopZero; //Volume of particle j
			
			      //=====Divergence of velocity==========
			      const float rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
			      divr[p1]-=volume*rDivW;
		      }
		    }
      }
	  }
  }
}

//===============================================================================
///Populate matrix b with values
//===============================================================================
void JSphCpu::RHSandLHSStorage(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,
	const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,const tdouble3 *pos,
	const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,std::vector<double> &matrixb,const unsigned *porder,const unsigned *idpc,const double dt, const unsigned ppedim,
	const float *divr,const float freesurface,std::vector<int> &row)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetTypeValue(Codec[p1])!=1){
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const tdouble3 posp1=pos[p1];
		unsigned numOfInteractions=0;
	  //-Particle order in Matrix
	  unsigned oi = porder[p1];

    if(divr[p1]>freesurface){
      //-Obtain interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
      for(int z=zini;z<zfin;z++){
        const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          const unsigned pini=beginendcell[cxini+ymod];
          const unsigned pfin=beginendcell[cxfin+ymod];

          //-Interactions
          //------------------------------------------------
          for(unsigned p2=pini;p2<pfin;p2++) if(CODE_GetTypeValue(Codec[p2])!=1){
          const double drx=posp1.x-pos[p2].x;
          const double dry=posp1.y-pos[p2].y;
          const double drz=posp1.z-pos[p2].z;
            const double rr2=drx*drx+dry*dry+drz*drz;

            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
			        //-Wendland kernel.
              double frx,fry,frz;
              GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);
			
			        //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
              float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			        const float volume=massp2/RhopZero; //Volume of particle j

			        //=====Divergence of velocity==========
							const float dvx=velp1.x-velrhop[p2].x, dvy=velp1.y-velrhop[p2].y, dvz=velp1.z-velrhop[p2].z;
							
			        const float temp_x=frx*dwxcorr[p1].x+fry*dwycorr[p1].x+frz*dwzcorr[p1].x;
              const float temp_y=frx*dwxcorr[p1].y+fry*dwycorr[p1].y+frz*dwzcorr[p1].y;
			        const float temp_z=frx*dwxcorr[p1].z+fry*dwycorr[p1].z+frz*dwzcorr[p1].z;
			        float temp=dvx*temp_x+dvy*temp_y+dvz*temp_z;

			        matrixb[oi]-=double(volume*temp);
							numOfInteractions++;
		        }
		      }
	      }
      }
	  }
		row[oi]+=numOfInteractions;
    if(boundp2) matrixb[oi]=matrixb[oi]/dt;
  } 
}

void JSphCpu::MatrixASetup(const unsigned ppedim,unsigned &nnz,std::vector<int> &row)const{
  for(unsigned i=0;i<ppedim;i++){
    unsigned nnzOld=nnz;
    nnz += row[i]+1;
    row[i] = nnzOld;
  }
  row[ppedim]=nnz;
}

//===============================================================================
///Populate matrix with values
//===============================================================================
void JSphCpu::PopulateMatrixAFluid(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,const tfloat4 *velrhop,float *divr,std::vector<double> &matrixInd,std::vector<int> &row,std::vector<int> &col,
  const unsigned *porder,const unsigned *irelation,std::vector<double> &matrixb,const unsigned *idpc,const word *code,const unsigned ppedim,const float freesurface,tfloat3 gravity,const double rhoZero)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if((CODE_GetTypeValue(code[p1])!=1)&&porder[p1]!=int(Np)){
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const tdouble3 posp1=pos[p1];
    
		//-Particle order in Matrix
		unsigned oi = porder[p1];
		const unsigned diag=row[oi];
		col[diag]=oi;
		unsigned index=diag+1;
	  if(divr[p1]>freesurface){  
      //FLUID INTERACTION
      //-Obtain interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
      for(int z=zini;z<zfin;z++){
        const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          const unsigned pini=beginendcell[cxini+ymod];
          const unsigned pfin=beginendcell[cxfin+ymod];

          //-Interactions
          //------------------------------------------------
          for(unsigned p2=pini;p2<pfin;p2++){
            const double drx=posp1.x-pos[p2].x;
            const double dry=posp1.y-pos[p2].y;
            const double drz=posp1.z-pos[p2].z;
            const double rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
  	          unsigned oj = porder[p2];

		          //-Wendland kernel.
              double frx,fry,frz;
              GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);

			        //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
              float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			        const float volume=massp2/RhopZero; //Volume of particle j
			
			        //===== Laplacian operator =====
			        const float rDivW=drx*frx+dry*fry+drz*frz;
			        float temp=2.0f*rDivW/(RhopZero*(rr2+Eta2));
              matrixInd[index]=double(-temp*volume);
              col[index]=oj;
			        matrixInd[diag]+=double(temp*volume);
              index++;
		        }  
          }
        }
	    }

      //BOUNDARY INTERACTION
      //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
      for(int z=zini;z<zfin;z++){
        const int zmod=(nc.w)*z; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          const unsigned pini=beginendcell[cxini+ymod];
          const unsigned pfin=beginendcell[cxfin+ymod];

          //-Interactions
          //------------------------------------------------
          for(unsigned p2=pini;p2<pfin;p2++){
            const double drx=posp1.x-pos[p2].x;
            const double dry=posp1.y-pos[p2].y;
            const double drz=posp1.z-pos[p2].z;
            const double rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
              const unsigned idp2=idpc[p2];
              const unsigned mkp2 = CODE_GetTypeValue(code[p2]);
  	          unsigned oj=porder[p2];

		          //-Wendland kernel.
              double frx,fry,frz;
              GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);

			        //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
              float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			        const float volume=massp2/RhopZero; //Volume of particle j
			
			        //===== Laplacian operator =====
			        const float rDivW=drx*frx+dry*fry+drz*frz;
			        float temp=2.0f*rDivW/(RhopZero*(rr2+Eta2));
              
              for(unsigned pk=diag;pk<unsigned(row[oi+1]);pk++){ 
                if(col[pk]==ppedim){
                  matrixInd[pk]=double(-temp*volume);
                  col[pk]=oj;
			            matrixInd[diag]+=(temp*volume);
                  break;
                }
                else if(col[pk]==oj){
                  matrixInd[pk]-=double(temp*volume);
                  matrixInd[diag]+=double(temp*volume);
                  break;
                }
              }
		        }  
          }
        }
	    }
	  }
    else matrixInd[diag]=1.0;
  }
}

void JSphCpu::PopulateMatrixABoundary(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,const tfloat4 *velrhop,float *divr,std::vector<double> &matrixInd,std::vector<int> &row,std::vector<int> &col,
  const unsigned *porder,const unsigned *irelation,std::vector<double> &matrixb,const unsigned *idpc,const word *code,const unsigned ppedim,const float freesurface,tfloat3 gravity,const double rhoZero)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if((CODE_GetTypeValue(code[p1])!=1)&&porder[p1]!=int(Np)){
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    const tdouble3 posp1=pos[p1];
    
		//-Particle order in Matrix
		unsigned oi = porder[p1];
		const unsigned diag=row[oi];
		col[diag]=oi;
		unsigned index=diag+1;
	  if(divr[p1]>freesurface){  
      //FLUID INTERACTION
      //-Obtain interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
      for(int z=zini;z<zfin;z++){
        const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          const unsigned pini=beginendcell[cxini+ymod];
          const unsigned pfin=beginendcell[cxfin+ymod];

          //-Interactions
          //------------------------------------------------
          for(unsigned p2=pini;p2<pfin;p2++){
            const double drx=posp1.x-pos[p2].x;
            const double dry=posp1.y-pos[p2].y;
            const double drz=posp1.z-pos[p2].z;
            const double rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
  	          unsigned oj = porder[p2];

		          //-Wendland kernel.
              double frx,fry,frz;
              GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);

			        //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
              float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			        const float volume=massp2/RhopZero; //Volume of particle j
			
			        //===== Laplacian operator =====
			        const float rDivW=drx*frx+dry*fry+drz*frz;
			        float temp=2.0f*rDivW/(RhopZero*(rr2+Eta2));
              matrixInd[index]=double(-temp*volume);
              col[index]=oj;
			        matrixInd[diag]+=double(temp*volume);
              index++;
		        }  
          }
        }
	    }

      //BOUNDARY INTERACTION
      //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
      for(int z=zini;z<zfin;z++){
        const int zmod=(nc.w)*z; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          const unsigned pini=beginendcell[cxini+ymod];
          const unsigned pfin=beginendcell[cxfin+ymod];

          //-Interactions
          //------------------------------------------------
          for(unsigned p2=pini;p2<pfin;p2++){
            const double drx=posp1.x-pos[p2].x;
            const double dry=posp1.y-pos[p2].y;
            const double drz=posp1.z-pos[p2].z;
            const double rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
              const unsigned idp2=idpc[p2];
              const unsigned mkp2 = CODE_GetTypeValue(code[p2]);
  	          unsigned oj=porder[p2];

		          //-Wendland kernel.
              double frx,fry,frz;
              GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);

			        //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
              float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			        const float volume=massp2/RhopZero; //Volume of particle j
			
			        //===== Laplacian operator =====
			        const float rDivW=drx*frx+dry*fry+drz*frz;
			        float temp=2.0f*rDivW/(RhopZero*(rr2+Eta2));
              
              if(oi!=oj){
                for(unsigned pk=diag;pk<unsigned(row[oi+1]);pk++){ 
                  if(col[pk]==ppedim){
                    matrixInd[pk]=double(-temp*volume);
                    col[pk]=oj;
			              matrixInd[diag]+=(temp*volume);
                    break;
                  }
                  else if(col[pk]==oj){
                    matrixInd[pk]-=double(temp*volume);
                    matrixInd[diag]+=double(temp*volume);
                    break;
                  }
                }
              }
              if(mkp2==1){
                unsigned p2k;
                for(unsigned k=0;k<Npb;k++) if(idpc[k]==irelation[idp2]){
                  p2k=k;
                  break;
                }

                double dist = pos[p2k].z-pos[p2].z;
			          temp = temp * RhopZero * fabs(Gravity.z) * dist;
			          matrixb[oi]+=double(volume*temp); 
              }
		        }  
          }
        }
	    }
	  }
    else matrixInd[diag]=1.0;
  }
}

void JSphCpu::FreeSurfaceMark(unsigned n,unsigned pinit,float *divr,std::vector<double> &matrixInd,std::vector<double> &matrixb,
  std::vector<int> &row,const unsigned *porder,const unsigned *idpc,const word *code,const unsigned ppedim)const{
  const int pfin=int(pinit+n);
  
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++) if(CODE_GetTypeValue(code[p1])!=1){   
	  //-Particle order in Matrix
	  unsigned oi = porder[p1];
    const int Mark=row[oi]+1;
    if(divr[p1]>=FreeSurface && divr[p1]<=FreeSurface+0.2f){
      double alpha=0.5*(1.0-cos(PI*double(divr[p1]-FreeSurface)/0.2));

      matrixb[oi]=matrixb[oi]*alpha;

      for(int index=Mark;index<row[oi+1];index++) matrixInd[index]=matrixInd[index]*alpha;
    }
  }
}

//===============================================================================
///Reorder pressure for particles
//===============================================================================
void JSphCpu::PressureAssign(unsigned np,unsigned pinit,const tdouble3 *pos,tfloat4 *velrhop,
  const unsigned *idpc,const unsigned *irelation,const unsigned *porder,std::vector<double> &x,const word *code,const unsigned npb,float *divr,tfloat3 gravity)const{

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<int(np);p1++) if((CODE_GetTypeValue(code[p1])!=1)&&porder[p1]!=np){
    velrhop[p1].w=float(x[porder[p1]]);

    if(!NegativePressureBound)if(p1<int(npb)&&velrhop[p1].w<0)velrhop[p1].w=0.0;
  }

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<int(np);p1++)if(CODE_GetTypeValue(code[p1])==1&&porder[p1]!=np){
    const unsigned j=irelation[idpc[p1]];
    if(j!=npb){
      unsigned p2k;
      for(int k=int(pinit);k<int(np);k++) if(idpc[k]==j){
        p2k=k;
        break;
      }

      const double drz=pos[p2k].z-pos[p1].z;
      if(divr[p2k]>0.0)velrhop[p1].w=float(x[porder[p1]]+double(RhopZero)*abs(Gravity.z)*drz);
      else velrhop[p1].w=float(double(RhopZero)*abs(Gravity.z)*drz);
    }
  }
}

#ifndef _WITHGPU
template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void JSphCpu::run_solver(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,std::vector<double> &matrixx,const unsigned ppedim){ 
  VectorType result(rhs);
  VectorType residual(rhs);
  viennacl::tools::timer timer;
  timer.start();
  result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  viennacl::backend::finish();    
  Log->Printf("  > Solver time: %f",timer.get());   
  residual -= viennacl::linalg::prod(matrix, result); 
	double normResidual=viennacl::linalg::norm_2(residual);
  if(normResidual){
		Log->Printf("  > Relative residual: %e",normResidual / viennacl::linalg::norm_2(rhs));  
		Log->Printf("  > Iterations: %u",solver.iters());
	}

  copy(result,matrixx);
}

void JSphCpu::solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,std::vector<double> &matrixa,std::vector<double> &matrixb,std::vector<double> &matrixx,std::vector<int> &row,std::vector<int> &col,const unsigned ppedim,const unsigned nnz){
    viennacl::context ctx;
   
    typedef double ScalarType;

		viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix;
    vcl_compressed_matrix.set(&row[0],&col[0],&matrixa[0],ppedim,ppedim,nnz);

    viennacl::vector<ScalarType> vcl_vec(matrixb.size(),ctx);
    viennacl::vector<ScalarType> vcl_result(vcl_compressed_matrix.size1(),ctx);
    copy(matrixb,vcl_vec);
		copy(matrixx,vcl_result);

    viennacl::linalg::bicgstab_tag bicgstab(tolerance,iterations);

    if(tprecond==PRECOND_Jacobi){
      Log->Printf("JACOBI PRECOND");
      viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_compressed_matrix,viennacl::linalg::jacobi_tag());
      run_solver(vcl_compressed_matrix,vcl_vec,bicgstab,vcl_jacobi,matrixx,ppedim);
    }
    else if(tprecond==PRECOND_AMG){
        Log->Printf("AMG PRECOND");
        viennacl::context host_ctx(viennacl::MAIN_MEMORY);
        viennacl::context target_ctx = viennacl::traits::context(vcl_compressed_matrix);

        viennacl::linalg::amg_tag amg_tag_agg_pmis;
        amg_tag_agg_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_AGGREGATION);
        if(tamginter==AMGINTER_AG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION); Log->Printf("INTERPOLATION: AGGREGATION ");}
        else if(tamginter==AMGINTER_SAG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION); Log->Printf("INTERPOLATION: SMOOTHED AGGREGATION");}
        amg_tag_agg_pmis.set_strong_connection_threshold(strongconnection);
        amg_tag_agg_pmis.set_jacobi_weight(jacobiweight);
        amg_tag_agg_pmis.set_presmooth_steps(presmooth);
        amg_tag_agg_pmis.set_postsmooth_steps(postsmooth); 
        amg_tag_agg_pmis.set_coarsening_cutoff(coarsecutoff); 
        amg_tag_agg_pmis.set_setup_context(host_ctx);
        amg_tag_agg_pmis.set_target_context(target_ctx); 
        viennacl::linalg::amg_precond<viennacl::compressed_matrix<double> > vcl_AMG(vcl_compressed_matrix,amg_tag_agg_pmis);
        Log->Printf(" * Setup phase (ViennaCL types)...");
        viennacl::tools::timer timer;
        timer.start();
        vcl_AMG.setup(); 
        std::cout << "levels = " << vcl_AMG.levels() << "\n";
        for(int i =0; i< vcl_AMG.levels();i++) std::cout << "level " << i << "\t" << "size = " << vcl_AMG.size(i) << "\n";
        viennacl::backend::finish(); 
        Log->Printf("  > Setup time: %f",timer.get());
        run_solver(vcl_compressed_matrix,vcl_vec,bicgstab,vcl_AMG,matrixx,ppedim);
    }
}    
#endif

//==============================================================================
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template <TpFtMode ftmode> void JSphCpu::InteractionForcesShifting
  (unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,float visco
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const tdouble3 *pos,tfloat4 *velrhop,const word *code,const unsigned *idp
  ,TpShifting tshifting,tfloat3 *shiftpos,float *divr,const float tensileN,const float tensileR)const
{
  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  //-Initialize viscth to calculate viscdt maximo con OpenMP / Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[MAXTHREADS_OMP*STRIDE_OMP];
  for(int th=0;th<OmpThreads;th++)viscth[th*STRIDE_OMP]=0;
  //-Initial execution with OpenMP / Inicia ejecucion con OpenMP.
  const int pfin=int(pinit+n);
  const float Wab1=GetKernelWab(Dp*Dp);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++){
   
    tfloat3 shiftposp1=TFloat3(0);
    float divrp1=0;

    //-Obtain data of particle p1 in case of floating objects / Obtiene datos de particula p1 en caso de existir floatings.
    //bool ftp1=false;     //-Indicate if it is floating / Indica si es floating.
    //float ftmassp1=1.f;  //-Contains floating particle mass or 1.0f if it is fluid / Contiene masa de particula floating o 1.0f si es fluid.
    /*if(USE_FLOATING){
      ftp1=(CODE_GetType(code[p1])==CODE_TYPE_FLOATING);
      if(ftp1)ftmassp1=FtObjs[CODE_GetTypeValue(code[p1])].massp;
      //if(ftp1 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
      if(ftp1)shiftposp1.x=FLT_MAX;  //-For floating objects do not calculate shifting / Para floatings no se calcula shifting.
    }*/

    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    //const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
    
    const tdouble3 posp1=pos[p1];
    
    //-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
    //-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
    for(int z=zini;z<zfin;z++){
      const int zmod=(nc.w)*z+cellinitial; //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        const unsigned pini=beginendcell[cxini+ymod];
        const unsigned pfin=beginendcell[cxfin+ymod];

        //-Interaction of Fluid with type Fluid or Bound / Interaccion de Fluid con varias Fluid o Bound.
        //------------------------------------------------
        for(unsigned p2=pini;p2<pfin;p2++){

          const double drx=posp1.x-pos[p2].x;
          const double dry=posp1.y-pos[p2].y;
          const double drz=posp1.z-pos[p2].z;
          const double rr2=drx*drx+dry*dry+drz*drz;

          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
            //-Wendland kernel.
            double frx,fry,frz;
            GetKernelDouble(rr2,drx,dry,drz,frx,fry,frz);
			
            //===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
            float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			      const float volumep2=massp2/RhopZero; //Volume of particle j
            //bool ftp2=false;    //-Indicate if it is floating / Indica si es floating.
            //bool compute=true;  //-Deactivate when using DEM and if it is of type float-float or float-bound /  Se desactiva cuando se usa DEM y es float-float o float-bound.
            /*if(USE_FLOATING){
              ftp2=(CODE_GetType(code[p2])==CODE_TYPE_FLOATING);
              if(ftp2)massp2=FtObjs[CODE_GetTypeValue(code[p2])].massp;
              //if(ftp2 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
              if(ftp2 && tshifting==SHIFT_NoBound)shiftposp1.x=FLT_MAX; //-With floating objects do not use shifting / Con floatings anula shifting.
              compute=!(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Deactivate when using DEM and if it is of type float-float or float-bound / Se desactiva cuando se usa DEM y es float-float o float-bound.
            }*/

            //-Shifting correction
            //if(shiftposp1.x!=FLT_MAX){
              const float massrhop=massp2/RhopZero;
              const float tensile=tensileN*powf(GetKernelWab(rr2)/Wab1,tensileR);
             
              //const bool noshift=(boundp2 && (tshifting==SHIFT_NoBound || (tshifting==SHIFT_NoFixed && CODE_GetType(code[p2])==CODE_TYPE_FIXED)));
              shiftposp1.x+=massrhop*(1.0f+tensile)*frx; //-For boundary do not use shifting / Con boundary anula shifting.
              shiftposp1.y+=massrhop*(1.0f+tensile)*fry;
              shiftposp1.z+=massrhop*(1.0f+tensile)*frz;
              divrp1-=massrhop*(drx*frx+dry*fry+drz*frz);
            //}
          }
        }
      }
    }
    divr[p1]+=divrp1;  
    shiftpos[p1]=shiftpos[p1]+shiftposp1; 
  }
}

void JSphCpu::Interaction_Shifting
  (unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell,const tdouble3 *pos
  ,tfloat4 *velrhop,const unsigned *idp,const word *code
  ,tfloat3 *shiftpos,float *divr,const float tensileN,const float tensileR)const
{
  const unsigned npf=np-npb;
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);

  if(npf){
    if(!WithFloating){                   const TpFtMode ftmode=FTMODE_None;
      //-Interaction Fluid-Fluid / Interaccion Fluid-Fluid
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
      //-Interaction Fluid-Fluid / Interaccion Fluid-Bound
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
    }else if(!UseDEM){                   const TpFtMode ftmode=FTMODE_Sph;
      //-Interaction Fluid-Fluid / Interaccion Fluid-Fluid
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
      //-Interaction Fluid-Fluid / Interaccion Fluid-Bound
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
    }else{                               const TpFtMode ftmode=FTMODE_Dem; 
      //-Interaction Fluid-Fluid / Interaccion Fluid-Fluid
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
      //-Interaction Fluid-Fluid / Interaccion Fluid-Bound
      InteractionForcesShifting<ftmode>(npf,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,divr,tensileN,tensileR);
    }
  }
}

void JSphCpu::Shift(double dt){
  //-Calculate rhop of boudary and set velocity=0 / Calcula rhop de contorno y vel igual a cero.
  const int npb=int(Npb);

  //-Calculate fluid values / Calcula datos de fluido.
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=npb;p<np;p++){
    if(!WithFloating || CODE_GetType(Codec[p])==CODE_TYPE_FLUID){//-Particulas: Fluid
      //-Calculate displacement and update position / Calcula desplazamiento y actualiza posicion.
      double dx=double(ShiftPosc[p].x);
      double dy=double(ShiftPosc[p].y);
      double dz=double(ShiftPosc[p].z); 
      bool outrhop=false;//(rhopnew<RhopOutMin||rhopnew>RhopOutMax);
      UpdatePos(PosPrec[p],dx,dy,dz,outrhop,p,Posc,Dcellc,Codec);
    }
  }
}