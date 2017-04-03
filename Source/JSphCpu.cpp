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

  #include "viennacl/linalg/amg.hpp" 
  #include "viennacl/tools/timer.hpp"
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
#include "JTimeOut.h"
#include "JSphAccInput.h"

#include <climits>

#ifdef _WITHOMP
  #include <omp.h>  //Activate tb in Properties config -> C/C++ -> Language -> OpenMp

#else
  #define omp_get_thread_num() 0
  #define omp_get_max_threads() 1
#endif

#ifndef WIN32
  #include <unistd.h>
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
  Acec=NULL; Divr=NULL;
  dWxCorrShiftPos=NULL; dWyCorr=NULL; dWzCorrTensile=NULL;
  RidpMove=NULL; 
  FtRidp=NULL;
  FtoForces=NULL;
	MirrorPosc=NULL;
	MirrorCell=NULL;
	MLS=NULL;
  a=NULL;
  b=NULL;
  x=NULL;
	rowInd=NULL;
  colInd=NULL;
  FreeCpuMemoryParticles();
  FreeCpuMemoryFixed();
}

//==============================================================================
/// Libera memoria fija en cpu para moving y floating.
/// Release fixed memory on CPU for moving and floating bodies.
//==============================================================================
void JSphCpu::FreeCpuMemoryFixed(){
  MemCpuFixed=0;
  delete[] RidpMove;				RidpMove=NULL;
  delete[] FtRidp;					FtRidp=NULL;
  delete[] FtoForces;				FtoForces=NULL;
	delete[] MirrorPosc;			MirrorPosc=NULL;
	delete[] MirrorCell;			MirrorCell=NULL;
	delete[] MLS;							MLS=NULL;
	delete[] rowInd;					rowInd=NULL;	
	delete[] a;								a=NULL;
	delete[] colInd;					colInd=NULL;
	delete[] Acec;						Acec=NULL;
	delete[] dWxCorrShiftPos;	dWxCorrShiftPos=NULL;
	delete[] dWzCorrTensile;	dWzCorrTensile=NULL;
}

//==============================================================================
/// Allocates memory for arrays with fixed size (motion and floating bodies).
//==============================================================================
void JSphCpu::AllocCpuMemoryFixed(){
  MemCpuFixed=0;
	const unsigned np=Np;
	const unsigned npb=Npb;
	const unsigned npf=np-npb;
  unsigned PPEMem=MatrixMemory*np; //Predicts max number of neighbours per particle dependant on kernel support size

  try{
		MirrorPosc=new tdouble3[npb];				MemCpuFixed+=(sizeof(tdouble3)*npb);
		MirrorCell=new unsigned[npb];				MemCpuFixed+=(sizeof(unsigned)*npb);
		MLS=new tfloat4[npb];								MemCpuFixed+=(sizeof(tfloat4)*npb);
		rowInd=new int[np+1];								MemCpuFixed+=(sizeof(int)*(np+1));
		a=new double[PPEMem];								MemCpuFixed+=(sizeof(double)*(PPEMem));
		colInd=new int[PPEMem];							MemCpuFixed+=(sizeof(int)*PPEMem);
		Acec=new tfloat3[npf];							MemCpuFixed+=(sizeof(tfloat3)*npf);
		dWxCorrShiftPos=new tfloat3[npf];		MemCpuFixed+=(sizeof(tfloat3)*npf);
		dWzCorrTensile=new tfloat3[npf];		MemCpuFixed+=(sizeof(tfloat3)*npf);
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
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_2B,2);  ///<-code*2
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_4B,5);  ///<-idp*2,dcell*2,divr+npfout
	ArraysCpu->AddArrayCount(JArraysCpu::SIZE_8B,2); ///<-b,x
	ArraysCpu->AddArrayCount(JArraysCpu::SIZE_12B,1); ///<-Saving/dWyCorr
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_16B,2); ///<-velrhop,velrhoppre
  ArraysCpu->AddArrayCount(JArraysCpu::SIZE_24B,3); ///<-pos*2, pospre+npfout
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
  Log->Printf("**JSphCpu: Requesting cpu memory for %u particles: %.1f MB.",npnew,mbparticle*npnew);
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
    Log->Print("\nWave paddles configuration:");
    WaveGen->Init(TimeMax,Gravity,Simulate2D,CellOrder,MassFluid,Dp,Dosh,Scell,Hdiv,DomPosMin,DomRealPosMin,DomRealPosMax);
    WaveGen->VisuConfig(""," ");
  }

  //-Prepares AccInput configuration.
  if(AccInput){
     Log->Print("\nAccInput configuration:");
     AccInput->Init(TimeMax);
     AccInput->VisuConfig(""," ");
  }

  //-Process Special configurations in XML.
  JXml xml; xml.LoadFile(FileXml);

  //-Configuration of SaveDt.
  if(xml.GetNode("case.execution.special.savedt",false)){
    SaveDt=new JSaveDt(Log);
    SaveDt->Config(&xml,"case.execution.special.savedt",TimeMax,TimePart);
    SaveDt->VisuConfig("\nSaveDt configuration:"," ");
  }

  //-Shows configuration of JTimeOut.
  if(TimeOut->UseSpecialConfig())TimeOut->VisuConfig(Log,"\nTimeOut configuration:"," ");
  Part=PartIni; Nstep=0; PartNstep=0; PartOut=0;
  TimeStep=TimeStepIni; TimeStepM1=TimeStep;
  if(DtFixed)DtIni=DtFixed->GetDt(TimeStep,DtIni);
  TimePartNext=TimeOut->GetNextTime(TimeStep);
}

//==============================================================================
/// Adds variable acceleration from input files.
//==============================================================================
void JSphCpu::AddAccInput(){
  for(unsigned c=0;c<AccInput->GetCount();c++){
    unsigned mkfluid;
    tdouble3 acclin,accang,centre,velang,vellin;
    bool setgravity;
    AccInput->GetAccValues(c,TimeStep,mkfluid,acclin,accang,centre,velang,vellin,setgravity);
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
          const tdouble3 vel=TDouble3(Velrhopc[p].x,Velrhopc[p].y,Velrhopc[p].z);//-Get the current particle's velocity

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
          acc.x+=((2.0*velang.y)*vel.z)-((2.0*velang.z)*(vel.y-vellin.y));
          acc.y+=((2.0*velang.z)*vel.x)-((2.0*velang.x)*(vel.z-vellin.z));
          acc.z+=((2.0*velang.x)*vel.y)-((2.0*velang.y)*(vel.x-vellin.x));
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
  //if(Deltac)memset(Deltac,0,sizeof(float)*np);                       //Deltac[]=0

  //-Apply the extra forces to the correct particle sets.
  if(AccInput)AddAccInput();

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
	}

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
/// Devuelve valores de kernel gradients: frx, fry y frz.
/// Return values of kernel gradients: frx, fry and frz.
//==============================================================================
void JSphCpu::GetKernelQuintic(float rr2,float drx,float dry,float drz,float &frx,float &fry,float &frz)const{
  const float rad=sqrt(rr2);
  const float qq=rad/H;

	//-Quintic Spline
  float fac;
  if(qq<1.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f)-75.0f*powf(1.0f-qq,4.0f));
  else if(qq<2.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f));
  else if(qq<3.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f));
  else fac=0;
  fac=fac/rad;

  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

void JSphCpu::GetKernelWendland(float rr2,float drx,float dry,float drz,float &frx,float &fry,float &frz)const{
  const float rad=sqrt(rr2);
  const float qq=rad/H;
  //-Wendland kernel
  const float wqq1=1.f-0.5f*qq;
  const float fac=Bwen*qq*wqq1*wqq1*wqq1/rad;

  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

//==============================================================================
/// Devuelve valores de kernel: Wab = W(q) con q=r/H.
/// Return values of kernel: Wab = W(q) where q=r/H.
//==============================================================================
float JSphCpu::GetKernelQuinticWab(float rr2)const{
  const float qq=sqrt(rr2)/H;
  //-Quintic Spline
  float wab;
 
  if(qq<1.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f)+15.0f*powf(1.0f-qq,5.0f));
  else if(qq<2.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f));
  else if(qq<3.0f)wab=Awen*(powf(3.0f-qq,5.0f));
  else wab=0;
  return(wab);
}

float JSphCpu::GetKernelWendlandWab(float rr2)const{
  const float qq=sqrt(rr2)/H;
  //-Wendland kernel.
  const float wqq=2.0f*qq+1.0f;
  const float wqq1=1.0f-0.5f*qq;

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

void JSphCpu::MirrorDCell(unsigned npb,const word *code,const tdouble3 *mirrorPos,unsigned *mirrorCell,unsigned *idpc){
	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npb);p1++){
		unsigned idp1=idpc[p1];
		const tdouble3 ps=mirrorPos[idp1];
		const double dx=ps.x-DomPosMin.x;
		const double dy=ps.y-DomPosMin.y;
		const double dz=ps.z-DomPosMin.z;
		unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
		mirrorCell[idp1]=PC__Cell(DomCellCode,cx,cy,cz);
	}
}

//=============================================================================
/// Slip Conditions and Boundary interactions
//=============================================================================
template<TpKernel tker> void JSphCpu::Boundary_Velocity(TpSlipCond TSlipCond,unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,tfloat4 *velrhop,const word *code,float *divr,tdouble3 *mirrorPos,const unsigned *idp,const unsigned *mirrorCell,tfloat4 *mls,int *row)const{

  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++){
		unsigned idp1=idp[p1];
		tfloat3 Sum=TFloat3(0);
		const tdouble3 posp1=mirrorPos[idp1];
		float divrp1=0;
		const tfloat4 mlsp1=mls[p1];
		unsigned rowCount=0;
		//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
		float massp2=MassFluid; //-Contiene masa de particula segun sea bound o fluid.
		const float volume=massp2/RhopZero; //Volume of particle j

		//-Obtain interaction limits / Obtiene limites de interaccion
		int cxini,cxfin,yini,yfin,zini,zfin;
		GetInteractionCells(mirrorCell[idp1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

		for(int fluid=0;fluid<=1;fluid++){
			for(int z=zini;z<zfin;z++){
				const int zmod=(nc.w)*z+(cellinitial*fluid); //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
				for(int y=yini;y<yfin;y++){
					int ymod=zmod+nc.x*y;
					const unsigned pini=beginendcell[cxini+ymod];
					const unsigned pfin=beginendcell[cxfin+ymod];
					//-Interactions
					//------------------------------------------------
					for(unsigned p2=pini;p2<pfin;p2++){
						const float drx=float(posp1.x-pos[p2].x);
						const float dry=float(posp1.y-pos[p2].y);
						const float drz=float(posp1.z-pos[p2].z);
						const float rr2=drx*drx+dry*dry+drz*drz;
						if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
							//-Wendland kernel.
							float frx,fry,frz;
							if(tker==KERNEL_Quintic) GetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
							else if(tker==KERNEL_Wendland) GetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
							const float rDivW=drx*frx+dry*fry+drz*frz;
							divrp1-=volume*rDivW;
							if(fluid){
								const tfloat4 velrhop2=velrhop[p2];
								float Wab;
								if(tker==KERNEL_Quintic) Wab=GetKernelQuinticWab(rr2);
								else if(tker==KERNEL_Wendland) Wab=GetKernelWendlandWab(rr2);
								const float temp=(mlsp1.w+mlsp1.x*drx+mlsp1.y*dry+mlsp1.z*drz)*Wab;
								Sum.x+=velrhop2.x*temp*volume;
								Sum.y+=velrhop2.y*temp*volume;
								Sum.z+=velrhop2.z*temp*volume;
								rowCount++;
							}
						}
					}
				}
			}
		}
		
		divr[p1]=divrp1;
		row[p1]=rowCount;
			
		if(TSlipCond==SLIPCOND_Slip){
			tfloat3 NormDir=TFloat3(0),NormVel=TFloat3(0),TangDir=TFloat3(0),TangVel=TFloat3(0),BitangDir=TFloat3(0),BitangVel=TFloat3(0); 
			NormDir.x=float(posp1.x-pos[p1].x);
			if(!Simulate2D)NormDir.y=float(posp1.y-pos[p1].y);
			NormDir.z=float(posp1.z-pos[p1].z);

			TangDir.x=NormDir.z+NormDir.y;
			if(!Simulate2D)TangDir.y=-(NormDir.x+NormDir.z);
			TangDir.z=-NormDir.x+NormDir.y;

			BitangDir.x=TangDir.y*NormDir.z-NormDir.y*TangDir.z;
			if(!Simulate2D)BitangDir.y=NormDir.x*TangDir.z-TangDir.x*NormDir.z;
			BitangDir.z=TangDir.x*NormDir.y-NormDir.x*TangDir.y;

			float MagNorm=NormDir.x*NormDir.x+NormDir.y*NormDir.y+NormDir.z*NormDir.z;
			if(MagNorm){MagNorm=sqrtf(MagNorm); NormDir.x=NormDir.x/MagNorm; NormDir.y=NormDir.y/MagNorm; NormDir.z=NormDir.z/MagNorm;}

			float MagTang=TangDir.x*TangDir.x+TangDir.y*TangDir.y+TangDir.z*TangDir.z;
			if(MagTang){MagTang=sqrtf(MagTang); TangDir.x=TangDir.x/MagTang; TangDir.y=TangDir.y/MagTang; TangDir.z=TangDir.z/MagTang;}

			float MagBitang=BitangDir.x*BitangDir.x+BitangDir.y*BitangDir.y+BitangDir.z*BitangDir.z;
			if(MagBitang){MagBitang=sqrtf(MagBitang); BitangDir.x=BitangDir.x/MagBitang; BitangDir.y=BitangDir.y/MagBitang; BitangDir.z=BitangDir.z/MagBitang;}

			float NormProdVel=Sum.x*NormDir.x+Sum.y*NormDir.y+Sum.z*NormDir.z;
			float TangProdVel=Sum.x*TangDir.x+Sum.y*TangDir.y+Sum.z*TangDir.z;
			float BitangProdVel=Sum.x*BitangDir.x+Sum.y*BitangDir.y+Sum.z*BitangDir.z;

			NormVel.x=NormDir.x*NormProdVel;
			NormVel.y=NormDir.y*NormProdVel;
			NormVel.z=NormDir.z*NormProdVel;
			TangVel.x=TangDir.x*TangProdVel;
			TangVel.y=TangDir.y*TangProdVel;
			TangVel.z=TangDir.z*TangProdVel;
			BitangVel.x=BitangDir.x*BitangProdVel;
			BitangVel.y=BitangDir.y*BitangProdVel;
			BitangVel.z=BitangDir.z*BitangProdVel;
				
			velrhop[p1].x=2.0f*velrhop[p1].x+TangVel.x+BitangVel.x-NormVel.x;
			velrhop[p1].y=2.0f*velrhop[p1].y+TangVel.y+BitangVel.y-NormVel.y;
			velrhop[p1].z=2.0f*velrhop[p1].z+TangVel.z+BitangVel.z-NormVel.z;
		}
		else if(TSlipCond==SLIPCOND_NoSlip){
			velrhop[p1].x=2.0f*velrhop[p1].x-Sum.x;
			velrhop[p1].y=2.0f*velrhop[p1].y-Sum.y;
			velrhop[p1].z=2.0f*velrhop[p1].z-Sum.z;
		}
	}
}

//==============================================================================
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template<TpKernel tker,TpFtMode ftmode> void JSphCpu::InteractionForcesFluid
  (TpInter tinter, unsigned npf,unsigned npb,tint4 nc,int hdiv,unsigned cellinitial,float visco
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const tdouble3 *pos,const tfloat4 *velrhop,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,const word *code,const unsigned *idp
  ,tfloat3 *ace,float *divr,int *row,const unsigned matOrder)const
{
  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  //-Initialize viscth to calculate viscdt maximo con OpenMP / Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[MAXTHREADS_OMP*STRIDE_OMP];
  for(int th=0;th<OmpThreads;th++)viscth[th*STRIDE_OMP]=0;
  //-Initial execution with OpenMP / Inicia ejecucion con OpenMP.
  const int pfin=int(npb+npf);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(npb);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
		const unsigned Correctp1=p1-npb;
		float divrp1=0;
    tfloat3 acep1=TFloat3(0);
		tdouble3 dwxp1=TDouble3(0); tdouble3 dwyp1=TDouble3(0); tdouble3 dwzp1=TDouble3(0);
		tfloat3 dwx=dwxcorr[Correctp1]; tfloat3 dwy=dwycorr[Correctp1]; tfloat3 dwz=dwzcorr[Correctp1]; //  dwz.x   dwz.y   dwz.z
		int rowCount=0;
		float nearestBound=float(Dp*Dp);
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
          const float drx=float(posp1.x-pos[p2].x);
					const float dry=float(posp1.y-pos[p2].y);
					const float drz=float(posp1.z-pos[p2].z);
					const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){

            //-Wendland kernel.
            float frx,fry,frz;
            if(tker==KERNEL_Quintic) GetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
						else if(tker==KERNEL_Wendland) GetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
			
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

							dwxp1.x-=volumep2*frx*drx; dwxp1.y-=volumep2*frx*dry; dwxp1.z-=volumep2*frx*drz;
							dwyp1.x-=volumep2*fry*drx; dwyp1.y-=volumep2*fry*dry; dwyp1.z-=volumep2*fry*drz;
							dwzp1.x-=volumep2*frz*drx; dwzp1.y-=volumep2*frz*dry; dwzp1.z-=volumep2*frz*drz;
							rowCount++;
            }

			      //===== Acceleration from pressure gradient ===== 
            if(compute && tinter==2){
			        const float temp_x=frx*dwx.x+fry*dwy.x+frz*dwz.x;
              const float temp_y=frx*dwx.y+fry*dwy.y+frz*dwz.y;
			        const float temp_z=frx*dwx.z+fry*dwy.z+frz*dwz.z;
			        const float temp=volumep2*(velrhop[p2].w-pressp1);
              acep1.x+=temp*temp_x; acep1.y+=temp*temp_y; acep1.z+=temp*temp_z;

							//See if fluid particles are close to boundary
							if(boundp2){
								if(rr2<=nearestBound){
									nearestBound=rr2;
									row[p1]=p2;
								}
							}
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
				ace[Correctp1]=ace[Correctp1]+acep1; 
				divr[p1]+=divrp1;
				dwxcorr[Correctp1].x+=float(dwxp1.x); dwxcorr[Correctp1].y+=float(dwxp1.y); dwxcorr[Correctp1].z+=float(dwxp1.z);
				dwycorr[Correctp1].x+=float(dwyp1.x); dwycorr[Correctp1].y+=float(dwyp1.y); dwycorr[Correctp1].z+=float(dwyp1.z);
				dwzcorr[Correctp1].x+=float(dwzp1.x); dwzcorr[Correctp1].y+=float(dwzp1.y); dwzcorr[Correctp1].z+=float(dwzp1.z);
				row[p1-matOrder]+=rowCount;
			}
	    if(tinter==2) ace[Correctp1]=ace[Correctp1]+acep1/RhopZero;
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

void JSphCpu::AssignPeriodic(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
  const tdouble3 *pos,const unsigned *idpc,const word *code,const unsigned *dCell)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])==CODE_PERIODIC){
    const unsigned idp1=idpc[p1];

		tdouble3 rpos=pos[p1];
		double dx=rpos.x-MapRealPosMin.x;
		double dy=rpos.y-MapRealPosMin.y;
		double dz=rpos.z-MapRealPosMin.z;
		
		//-Adjust position according to periodic conditions and compare domain limits / Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
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
		rpos=TDouble3(dx,dy,dz)+MapRealPosMin;

		dx=rpos.x-DomPosMin.x;
		dy=rpos.y-DomPosMin.y;
		dz=rpos.z-DomPosMin.z;
		unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
		const unsigned Cell=PC__Cell(DomCellCode,cx,cy,cz);

		//FLUID INTERACTION
    //-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(Cell,hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

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
          if(idp1==Idpc[p2]){
						Acec[p1]=Acec[p2];
						break;
          }	
        }
	    }
	  }
  }
}

//==============================================================================
/// Seleccion de parametros template para Interaction_ForcesFluidT.
/// Selection of template parameters for Interaction_ForcesFluidT.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode> void JSphCpu::Interaction_ForcesT
  (TpInter tinter, unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
  ,const tdouble3 *pos,tfloat4 *velrhop,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,const word *code,const unsigned *idp
  ,tfloat3 *ace,float *divr,tdouble3 *mirrorPos,const unsigned *mirrorCell,tfloat4 *mls,int *row)const
{
  const unsigned npf=np-npb;
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);
	const unsigned matOrder=npb-npbok;
  if(npf){
		if(tinter==1){
			if(Simulate2D) MLSBoundary2D<tker> (NpbOk,0,nc,hdiv,cellfluid,begincell,cellzero,pos,velrhop,idp,code,mirrorPos,mirrorCell,mls);
			else MLSBoundary3D<tker> (NpbOk,0,nc,hdiv,cellfluid,begincell,cellzero,pos,velrhop,idp,code,mirrorPos,mirrorCell,mls);
			Boundary_Velocity<tker> (TSlipCond,NpbOk,0,nc,hdiv,cellfluid,begincell,cellzero,dcell,pos,velrhop,code,divr,mirrorPos,idp,mirrorCell,mls,row);
		}

		//-Interaction Fluid-Fluid / Interaccion Fluid-Fluid
    InteractionForcesFluid<tker,ftmode> (tinter,npf,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,row,matOrder);

	  //-Interaction Fluid-Bound / Interaccion Fluid-Bound
    InteractionForcesFluid<tker,ftmode> (tinter,npf,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,row,matOrder);

    //-Interaction of DEM Floating-Bound & Floating-Floating / Interaccion DEM Floating-Bound & Floating-Floating //(DEM)
    //if(USE_DEM)InteractionForcesDEM<psimple> (CaseNfloat,nc,hdiv,cellfluid,begincell,cellzero,dcell,FtRidp,DemObjs,pos,pspos,velrhop,code,idp,viscdt,ace);
		if(tinter==1){
			if(Simulate2D) JSphCpu::InverseCorrection(np,npb,dWxCorrShiftPos,dWzCorrTensile,code);
			else JSphCpu::InverseCorrection3D(np,npb,dWxCorrShiftPos,dWyCorr,dWzCorrTensile,code);
		}

		/*if(PeriActive){
			AssignPeriodic(npf,npb,nc,hdiv,cellfluid,begincell,cellzero,Posc,Idpc,Codec,Dcellc);
			if(tinter==1)AssignPeriodic(npbok,0,nc,hdiv,0,begincell,cellzero,Posc,Idpc,Codec,Dcellc);
		}*/
	}
}

//==============================================================================
/// Seleccion de parametros template para Interaction_ForcesX.
/// Selection of template parameters for Interaction_ForcesX.
//==============================================================================
void JSphCpu::Interaction_Forces(TpInter tinter,TpKernel tkernel,unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
  ,const tdouble3 *pos,tfloat4 *velrhop,const unsigned *idp,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,const word *code
  ,tfloat3 *ace,float *divr,tdouble3 *mirrorPos,const unsigned *mirrorCell,tfloat4 *mls,int *row)const
{
	if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
		if(!WithFloating) Interaction_ForcesT<tker,FTMODE_None> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
		else if(!UseDEM)  Interaction_ForcesT<tker,FTMODE_Sph> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
		else              Interaction_ForcesT<tker,FTMODE_Dem> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
	}
	else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
		if(!WithFloating) Interaction_ForcesT<tker,FTMODE_None> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
		else if(!UseDEM)  Interaction_ForcesT<tker,FTMODE_Sph> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
		else              Interaction_ForcesT<tker,FTMODE_Dem> (tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,code,idp,ace,divr,mirrorPos,mirrorCell,mls,row);
	}
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
  const int npb=int(Npb);
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=npb;p<np;p++){
    //-Calcula densidad.
    //const float rhopnew=float(double(VelrhopPrec[p].w)+dt05*Arc[p]);
    if(!WithFloating || CODE_GetType(Codec[p])==CODE_TYPE_FLUID){//-Fluid Particles / Particulas: Fluid
      //-Update velocity
			const unsigned correctp1=p-npb;
      Velrhopc[p].x=float(double(VelrhopPrec[p].x)+double(Acec[correctp1].x)* dt);
      Velrhopc[p].y=float(double(VelrhopPrec[p].y)+double(Acec[correctp1].y)* dt);
      Velrhopc[p].z=float(double(VelrhopPrec[p].z)+double(Acec[correctp1].z)* dt);
    }
  }

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
  const double dt05=dt*0.5;
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(np>LIMIT_COMPUTESTEP_OMP)
  #endif
  for(int p=npb;p<np;p++){
    if(!WithFloating || CODE_GetType(Codec[p])==CODE_TYPE_FLUID){//-Particulas: Fluid
      //-Update velocity & density / Actualiza velocidad y densidad.
			const unsigned correctp1=p-npb;
      Velrhopc[p].x-=float((Acec[correctp1].x-Gravity.x)*dt); 
      Velrhopc[p].y-=float((Acec[correctp1].y-Gravity.y)*dt);  
      Velrhopc[p].z-=float((Acec[correctp1].z-Gravity.z)*dt);

			if(rowInd[p]!=npb) CorrectVelocity(p,rowInd[p],Posc,Velrhopc,Idpc,MirrorPosc);

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

void JSphCpu::CorrectVelocity(const unsigned p1,const unsigned nearestBound,const tdouble3 *pos,tfloat4 *velrhop,const unsigned *idpc,const tdouble3 *mirrorPos){
	tfloat3 NormDir=TFloat3(0), NormVelWall=TFloat3(0), NormVelp1=TFloat3(0);
	const unsigned nearestID=idpc[nearestBound];
	const tfloat3 velwall=TFloat3(velrhop[nearestBound].x,velrhop[nearestBound].y,velrhop[nearestBound].z);
	const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
	NormDir.x=float(mirrorPos[nearestID].x-pos[nearestBound].x);
	NormDir.y=float(mirrorPos[nearestID].y-pos[nearestBound].y);
	NormDir.z=float(mirrorPos[nearestID].z-pos[nearestBound].z);
	float MagNorm=NormDir.x*NormDir.x+NormDir.y*NormDir.y+NormDir.z*NormDir.z;
	if(MagNorm){MagNorm=sqrtf(MagNorm); NormDir.x=NormDir.x/MagNorm; NormDir.y=NormDir.y/MagNorm; NormDir.z=NormDir.z/MagNorm;}
	float NormProdVelWall=velwall.x*NormDir.x+velwall.y*NormDir.y+velwall.z*NormDir.z;
	float NormProdVelp1=velp1.x*NormDir.x+velp1.y*NormDir.y+velp1.z*NormDir.z;

	NormVelWall.x=NormDir.x*NormProdVelWall; NormVelp1.x=NormDir.x*NormProdVelp1;
	NormVelWall.y=NormDir.y*NormProdVelWall; NormVelp1.y=NormDir.y*NormProdVelp1;
	NormVelWall.z=NormDir.z*NormProdVelWall; NormVelp1.z=NormDir.z*NormProdVelp1;

	float dux=NormVelp1.x-NormVelWall.x;
	float duy=NormVelp1.y-NormVelWall.y;
	float duz=NormVelp1.z-NormVelWall.z;

	float VelNorm=dux*NormDir.x+duy*NormDir.y+duz*NormDir.z;
	if(VelNorm<0){
		velrhop[p1].x-=VelNorm*NormDir.x;
		velrhop[p1].y-=VelNorm*NormDir.y;
		velrhop[p1].z-=VelNorm*NormDir.z;
	}
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
  const int npb=int(Npb),np=int(Np),npf=np-npb;
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static) if(npf>LIMIT_COMPUTELIGHT_OMP)
  #endif
  for(int p=int(npb);p<int(np);p++){
		const unsigned Correctp1=p-npb;
    tfloat3 rshiftpos=dWxCorrShiftPos[Correctp1];
    float divrp1=Divr[p];

    double umagn=-double(ShiftCoef)*double(H)*double(H);

 	  tfloat3 norm=TFloat3(-rshiftpos.x,-rshiftpos.y,-rshiftpos.z);
	  tfloat3 tang=TFloat3(0);
	  tfloat3 bitang=TFloat3(0);
		rshiftpos=rshiftpos+dWzCorrTensile[Correctp1];
	  //-tangent and bitangent calculation
	  tang.x=norm.z+norm.y;		
	  if(!Simulate2D)tang.y=-(norm.x+norm.z);	
	  tang.z=-norm.x+norm.y;
	  bitang.x=tang.y*norm.z-norm.y*tang.z;
	  if(!Simulate2D)bitang.y=norm.x*tang.z-tang.x*norm.z;
	  bitang.z=tang.x*norm.y-norm.x*tang.y;

	  //-unit normal vector
	  float temp=norm.x*norm.x+norm.y*norm.y+norm.z*norm.z;
	  if(temp){
      temp=sqrt(temp);
	    norm.x=norm.x/temp; norm.y=norm.y/temp; norm.z=norm.z/temp;
    }
    else {norm.x=0.f; norm.y=0.f; norm.z=0.f;}

	  //-unit tangent vector
	  temp=tang.x*tang.x+tang.y*tang.y+tang.z*tang.z;
	  if(temp){
      temp=sqrt(temp);
	    tang.x=tang.x/temp; tang.y=tang.y/temp; tang.z=tang.z/temp;
    }
    else{tang.x=0.f; tang.y=0.f; tang.z=0.f;}

	 //-unit bitangent vector
	 temp=bitang.x*bitang.x+bitang.y*bitang.y+bitang.z*bitang.z;
	 if(temp){
     temp=sqrt(temp);
	   bitang.x=bitang.x/temp; bitang.y=bitang.y/temp; bitang.z=bitang.z/temp;
   }
   else{bitang.x=0.f; bitang.y=0.f; bitang.z=0.f;}

	  //-gradient calculation
	  float dcds=tang.x*rshiftpos.x+tang.z*rshiftpos.z+tang.y*rshiftpos.y;
	  float dcdn=norm.x*rshiftpos.x+norm.z*rshiftpos.z+norm.y*rshiftpos.y;
	  float dcdb=bitang.x*rshiftpos.x+bitang.z*rshiftpos.z+bitang.y*rshiftpos.y;

		if(divrp1<FreeSurface){
			rshiftpos.x=float(dcds*tang.x+dcdb*bitang.x);
			rshiftpos.y=float(dcds*tang.y+dcdb*bitang.y);
			rshiftpos.z=float(dcds*tang.z+dcdb*bitang.z);
    }
    else if(divrp1<=FreeSurface+ShiftOffset){ 
			rshiftpos.x=float(dcds*tang.x+dcdb*bitang.x+dcdn*norm.x*FactorNormShift);
			rshiftpos.y=float(dcds*tang.y+dcdb*bitang.y+dcdn*norm.y*FactorNormShift);
			rshiftpos.z=float(dcds*tang.z+dcdb*bitang.z+dcdn*norm.z*FactorNormShift);
    }

    rshiftpos.x=float(double(rshiftpos.x)*umagn);
    rshiftpos.y=float(double(rshiftpos.y)*umagn);
    rshiftpos.z=float(double(rshiftpos.z)*umagn);
  
    //Max Shifting
		if(TShifting==SHIFT_Max){
      float absShift=sqrt(rshiftpos.x*rshiftpos.x+rshiftpos.y*rshiftpos.y+rshiftpos.z*rshiftpos.z);
      if(abs(rshiftpos.x>0.1*Dp)) rshiftpos.x=float(0.1*Dp*rshiftpos.x/absShift);
      if(abs(rshiftpos.y>0.1*Dp)) rshiftpos.y=float(0.1*Dp*rshiftpos.y/absShift);
      if(abs(rshiftpos.z>0.1*Dp)) rshiftpos.z=float(0.1*Dp*rshiftpos.z/absShift);
    }

    dWxCorrShiftPos[Correctp1]=rshiftpos; //particles in fluid bulk, normal shifting
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
  ,const unsigned *ridp,tdouble3 *pos,unsigned *dcell,tfloat4 *velrhop,word *code,const unsigned *idpc,tdouble3 *mirrorPos)const
{
  const unsigned fin=ini+np;
  for(unsigned id=ini;id<fin;id++){
    const unsigned pid=RidpMove[id];
		unsigned idp1=idpc[pid];
		mirrorPos[idp1].x+=mvpos.x;
		mirrorPos[idp1].y+=mvpos.y;
		mirrorPos[idp1].z+=mvpos.z;
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
          MoveLinBound(np,pini,mvsimple,mvvel,RidpMove,Posc,Dcellc,Velrhopc,Codec,Idpc,MirrorPosc);
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
        //MoveLinBound(nparts,idbegin-CaseNfixed,mvsimple,mvvel,RidpMove,Posc,Dcellc,Velrhopc,Codec);
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
void JSphCpu::MirrorBoundary(unsigned npb,const tdouble3 *pos,const unsigned *idpc,tdouble3 *mirrorPos,const word *code,unsigned *Physrelation)const{

	//--Connect boundary-------
	//-------------------------
	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npb);p1++)if(CODE_GetTypeValue(code[p1])!=0&&CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
		const unsigned idp1=idpc[p1];
		const tdouble3 posp1=pos[p1];	
		double closestR=2.25*Fourh2;
		unsigned Physparticle=npb;
		Physrelation[p1]=npb;
		bool secondPoint=false;
		unsigned secondIrelation=npb;

		for(int p2=0;p2<int(npb);p2++) if(CODE_GetTypeValue(code[p2])==0){
			const double drx=posp1.x-pos[p2].x;
			const double dry=posp1.y-pos[p2].y;
			const double drz=posp1.z-pos[p2].z;
			const double rr2=drx*drx+dry*dry+drz*drz;
			if(rr2==closestR){
					secondPoint=true;
					secondIrelation=p2;
			}
			else if(rr2<closestR){
				closestR=rr2;
				Physparticle=p2;
				if(secondPoint)	secondPoint=false;
			}
		}

		if(Physparticle!=npb){
			if(secondPoint) mirrorTwoPoints(p1,Physparticle,secondIrelation,posp1,pos,npb);
			Physrelation[p1]=Physparticle;
		}
	}

	//--Find Mirror Points-----
	//-------------------------
	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npb);p1++)if(CODE_GetTypeValue(code[p1])==0&&CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
		const unsigned idp1=idpc[p1];
		const tdouble3 posp1=pos[p1];
		tdouble3 NormDir=TDouble3(0);

		for(int p2=0;p2<int(npb);p2++) if(CODE_GetTypeValue(code[p2])!=0){
			if(Physrelation[p2]==p1){
				const double drx=posp1.x-pos[p2].x;
				const double dry=posp1.y-pos[p2].y;
				const double drz=posp1.z-pos[p2].z;
				double rr2=drx*drx+dry*dry+drz*drz;
		
				rr2=sqrt(rr2);
				NormDir.x+=drx/rr2;
				if(!Simulate2D) NormDir.y+=dry/rr2;
				NormDir.z+=drz/rr2;
			}
		}

		double MagNorm=NormDir.x*NormDir.x+NormDir.y*NormDir.y+NormDir.z*NormDir.z;
		if(MagNorm){MagNorm=sqrt(MagNorm); NormDir.x=NormDir.x/MagNorm; NormDir.y=NormDir.y/MagNorm; NormDir.z=NormDir.z/MagNorm;}

		//Scale Norm to dp in each direction.
		double largestDir=abs(NormDir.x);
		if(abs(NormDir.y)>largestDir)largestDir=abs(NormDir.y);
		if(abs(NormDir.z)>largestDir)largestDir=abs(NormDir.z);

		if(largestDir){
			NormDir.x=NormDir.x/largestDir;
			NormDir.y=NormDir.y/largestDir;
			NormDir.z=NormDir.z/largestDir;
		}
		else{
			double closestR=2.25*Fourh2;
			unsigned closestp;
			for(int p2=0;p2<int(npb);p2++) if(CODE_GetTypeValue(code[p2])!=0){
				const double drx=posp1.x-pos[p2].x;
				const double dry=posp1.y-pos[p2].y;
				const double drz=posp1.z-pos[p2].z;
				const double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<closestR){
					closestR=rr2;
					closestp=p2;
				}
			}

			const double drx=posp1.x-pos[closestp].x;
			const double dry=posp1.y-pos[closestp].y;
			const double drz=posp1.z-pos[closestp].z;
			const double rr2=drx*drx+dry*dry+drz*drz;

			NormDir.x=drx/rr2; NormDir.y=dry/rr2; NormDir.z=drz/rr2;

			//Scale Norm to dp in each direction.
			double largestDir=abs(NormDir.x);
			if(abs(NormDir.y)>largestDir)largestDir=abs(NormDir.y);
			if(abs(NormDir.z)>largestDir)largestDir=abs(NormDir.z);
			
			NormDir.x=NormDir.x/largestDir;
			NormDir.y=NormDir.y/largestDir;
			NormDir.z=NormDir.z/largestDir;
		}

		mirrorPos[idp1].x=posp1.x+0.5*Dp*NormDir.x;
		if(!Simulate2D) mirrorPos[idp1].y=posp1.y+0.5*Dp*NormDir.y;
		mirrorPos[idp1].z=posp1.z+0.5*Dp*NormDir.z;
		Physrelation[p1]=p1;
	}

	//--Create Mirrors--------- 
	//-------------------------
	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npb);p1++)if(CODE_GetTypeValue(code[p1])!=0&&CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
		const unsigned idp1=idpc[p1];
		const tdouble3 posp1=pos[p1];	
		const unsigned Physparticle=Physrelation[p1];
		const unsigned mirIdp1=Idpc[Physparticle];
		const tdouble3 mirrorPoint=TDouble3(mirrorPos[mirIdp1].x,mirrorPos[mirIdp1].y,mirrorPos[mirIdp1].z);

		if(Physparticle!=npb){
			mirrorPos[idp1].x=2.0*mirrorPoint.x-posp1.x;
			mirrorPos[idp1].y=2.0*mirrorPoint.y-posp1.y;
			mirrorPos[idp1].z=2.0*mirrorPoint.z-posp1.z;
		}
		else{
			mirrorPos[idp1].x=mirrorPos[idp1].x;
			mirrorPos[idp1].y=mirrorPos[idp1].y;
			mirrorPos[idp1].z=mirrorPos[idp1].z;
		}
	}

	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npb);p1++)if(CODE_GetTypeValue(code[p1])==0&&CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
		const unsigned idp1=idpc[p1];
		const tdouble3 posp1=pos[p1];	
		const unsigned Physparticle=Physrelation[p1];
		const unsigned mirIdp1=Idpc[Physparticle];
		const tdouble3 mirrorPoint=TDouble3(mirrorPos[mirIdp1].x,mirrorPos[mirIdp1].y,mirrorPos[mirIdp1].z);

		mirrorPos[idp1].x=2.0*mirrorPoint.x-posp1.x;
		mirrorPos[idp1].y=2.0*mirrorPoint.y-posp1.y;
		mirrorPos[idp1].z=2.0*mirrorPoint.z-posp1.z;
	}
}

void JSphCpu::mirrorTwoPoints(const unsigned p1,unsigned &Physparticle,const unsigned secondIrelation,const tdouble3 posp1,const tdouble3 *pos,const unsigned npb)const{
	const double drx1=posp1.x-pos[Physparticle].x;
	const double dry1=posp1.y-pos[Physparticle].y;
	const double drz1=posp1.z-pos[Physparticle].z;
	const double drx2=posp1.x-pos[secondIrelation].x;
	const double dry2=posp1.y-pos[secondIrelation].y;
	const double drz2=posp1.z-pos[secondIrelation].z;

	tdouble3 searchPoint=TDouble3(0);
	searchPoint.x=posp1.x-(drx1+drx2);
	searchPoint.y=posp1.y-(dry1+dry2);
	searchPoint.z=posp1.z-(drz1+drz2);

	for(int i=0;i<int(npb);i++) {
		const double drx=searchPoint.x-pos[i].x;
		const double dry=searchPoint.y-pos[i].y;
		const double drz=searchPoint.z-pos[i].z;

		double rr2=drx*drx+dry*dry+drz*drz;
		if(rr2<=ALMOSTZERO) Physparticle=i;
	}

}

template<TpKernel tker> void JSphCpu::MLSBoundary2D(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
  const tdouble3 *pos,const tfloat4 *velrhop,const unsigned *idpc,const word *code,const tdouble3 *mirrorPos,const unsigned *mirrorCell,tfloat4 *mls)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
    const unsigned idp1=idpc[p1];
		
		//-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tdouble3 posp1=mirrorPos[idp1]; 
		double b11 = 0.0; double b12 = 0.0; double b13 = 0.0;
		double b21 = 0.0; double b22 = 0.0; double b23 = 0.0;
		double b31 = 0.0; double b32 = 0.0; double b33 = 0.0;

    //FLUID INTERACTION
		//-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(mirrorCell[idp1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

		//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
    float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
		const float volume=massp2/RhopZero; //Volume of particle j
			
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
					const float drx=float(posp1.x-pos[p2].x);
					const float dry=float(posp1.y-pos[p2].y);
					const float drz=float(posp1.z-pos[p2].z);
					const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
						float Wab;
						if(tker==KERNEL_Quintic) Wab=GetKernelQuinticWab(rr2);
						else if(tker==KERNEL_Wendland) Wab=GetKernelWendlandWab(rr2);
		        const double temp=Wab*volume;
						b11+= temp;		b12+=drx*temp;			b13+=drz*temp;
													b22+=drx*drx*temp;	b23+=drx*drz*temp;
																							b33+=drz*drz*temp;
					}  
				}
			}
		}
		
		b21=b12; b31=b13; b32=b23;

		double det = (b11*b22*b33+b12*b23*b31+b21*b32*b13)-(b31*b22*b13+b21*b12*b33+b23*b32*b11);
		
		if(det){
			mls[p1].w=float((b22*b33-b23*b32)/det);
			mls[p1].x=-float((b21*b33-b23*b31)/det);
			mls[p1].y=0.0;
			mls[p1].z=float((b21*b32-b22*b31)/det);
		}
  }
}

template<TpKernel tker> void JSphCpu::MLSBoundary3D(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
  const tdouble3 *pos,const tfloat4 *velrhop,const unsigned *idpc,const word *code,const tdouble3 *mirrorPos,const unsigned *mirrorCell,tfloat4 *mls)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
    const unsigned idp1=idpc[p1];
		
		//-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tdouble3 posp1=mirrorPos[idp1]; 
		double b11=0.0; double b12=0.0; double b13=0.0; double b14=0.0;
		double b21=0.0; double b22=0.0; double b23=0.0; double b24=0.0;
		double b31=0.0; double b32=0.0; double b33=0.0; double b34=0.0;
		double b41=0.0; double b42=0.0; double b43=0.0; double b44=0.0;

    //FLUID INTERACTION
		//-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(mirrorCell[idp1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

		//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
    float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
		const float volume=massp2/RhopZero; //Volume of particle j
			
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
					const float drx=float(posp1.x-pos[p2].x);
					const float dry=float(posp1.y-pos[p2].y);
					const float drz=float(posp1.z-pos[p2].z);
					const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
		        float Wab;
						if(tker==KERNEL_Quintic) Wab=GetKernelQuinticWab(rr2);
						else if(tker==KERNEL_Wendland) Wab=GetKernelWendlandWab(rr2);
		        const double temp=Wab*volume;
						b11+= temp;		b12+=drx*temp;			b13+=dry*temp;			b14+=drz*temp;			
													b22+=drx*drx*temp;	b23+=drx*dry*temp;	b24+=drx*drz*temp;
																							b33+=dry*dry*temp;  b34+=dry*drz*temp;
																																	b44+=drz*drz*temp;
					}  
				}
			}
		}
		
		b21=b12; b31=b13; b32=b23; b41=b14; b42=b24; b43=b34;

		double det=0;
		det+=(b11*b22*b33*b44 + b11*b23*b34*b42 + b11*b24*b32*b43);
		det+=(b12*b21*b34*b43 + b12*b23*b31*b44 + b12*b24*b33*b41);
		det+=(b13*b21*b32*b44 + b13*b22*b34*b41 + b13*b24*b31*b42);
		det+=(b14*b21*b33*b42 + b14*b22*b31*b43 + b14*b23*b32*b41);
		det-=(b11*b22*b34*b43 + b11*b23*b32*b44 + b11*b24*b33*b42);
		det-=(b12*b21*b33*b44 + b12*b23*b34*b41 + b12*b24*b31*b43);
		det-=(b13*b21*b34*b42 + b13*b22*b31*b44 + b13*b24*b32*b41);
		det-=(b14*b21*b32*b43 + b14*b22*b33*b41 + b14*b23*b31*b42);
		
		if(det){
			mls[p1].w=float((b22*b33*b44+b23*b34*b42+b24*b32*b43-b22*b34*b43-b23*b32*b44-b24*b33*b42)/det);
			mls[p1].x=float((b21*b34*b43+b23*b31*b44+b24*b33*b41-b21*b33*b44-b23*b34*b41-b24*b31*b43)/det);
			mls[p1].y=float((b21*b32*b44+b22*b34*b41+b24*b31*b42-b21*b34*b42-b22*b31*b44-b24*b32*b41)/det);
			mls[p1].z=float((b21*b33*b42+b22*b31*b43+b23*b32*b41-b21*b32*b43-b22*b33*b41-b23*b31*b42)/det);
		}
  }
}

//===============================================================================
///Kernel Correction
//===============================================================================
void JSphCpu::InverseCorrection(unsigned np,unsigned npb,tfloat3 *dwxcorr,tfloat3 *dwzcorr,const word *code)const{
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
	for(int p1=int(npb);p1<int(np);p1++){
		const unsigned Correctp1=p1-npb;
		tdouble3 dwx; dwx.x=dwxcorr[Correctp1].x; dwx.y=dwxcorr[Correctp1].y; dwx.z=dwxcorr[Correctp1].z;
		tdouble3 dwz; dwz.x=dwzcorr[Correctp1].x; dwz.y=dwzcorr[Correctp1].y; dwz.z=dwzcorr[Correctp1].z;

	  const double det=1.0/(dwx.x*dwz.z-dwz.x*dwx.z);
	
    if(det){
      dwxcorr[Correctp1].x=float(dwz.z*det);
	    dwxcorr[Correctp1].z=-float(dwx.z*det); 
	    dwzcorr[Correctp1].x=-float(dwz.x*det);
	    dwzcorr[Correctp1].z=float(dwx.x*det);
	  }
	}
}

void JSphCpu::InverseCorrection3D(unsigned np,unsigned npb,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,const word *code)const{
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
	for(int p1=int(npb);p1<int(np);p1++){
		const unsigned Correctp1=p1-npb;
    tdouble3 dwx; dwx.x=dwxcorr[Correctp1].x; dwx.y=dwxcorr[Correctp1].y; dwx.z=dwxcorr[Correctp1].z; //  dwx.x   dwx.y   dwx.z
    tdouble3 dwy; dwy.x=dwycorr[Correctp1].x; dwy.y=dwycorr[Correctp1].y; dwy.z=dwycorr[Correctp1].z; //  dwy.x   dwy.y   dwy.z
    tdouble3 dwz; dwz.x=dwzcorr[Correctp1].x; dwz.y=dwzcorr[Correctp1].y; dwz.z=dwzcorr[Correctp1].z; //  dwz.x   dwz.y   dwz.z

    const double det=(dwx.x*dwy.y*dwz.z+dwx.y*dwy.z*dwz.x+dwy.x*dwz.y*dwx.z)-(dwz.x*dwy.y*dwx.z+dwy.x*dwx.y*dwz.z+dwy.z*dwz.y*dwx.x);

    dwxcorr[Correctp1].x=float((dwy.y*dwz.z-dwy.z*dwz.y)/det);
    dwxcorr[Correctp1].y=-float((dwx.y*dwz.z-dwx.z*dwz.y)/det);
    dwxcorr[Correctp1].z=float((dwx.y*dwy.z-dwx.z*dwy.y)/det);
		dwycorr[Correctp1].x=-float((dwy.x*dwz.z-dwy.z*dwz.x)/det);
    dwycorr[Correctp1].y=float((dwx.x*dwz.z-dwx.z*dwz.x)/det);
    dwycorr[Correctp1].z=-float((dwx.x*dwy.z-dwx.z*dwy.x)/det);
    dwzcorr[Correctp1].x=float((dwy.x*dwz.y-dwy.y*dwz.x)/det);
    dwzcorr[Correctp1].y=-float((dwx.x*dwz.y-dwx.y*dwz.x)/det);
    dwzcorr[Correctp1].z=float((dwx.x*dwy.y-dwx.y*dwy.x)/det);
	}
}

void JSphCpu::MatrixASetup(const unsigned np,const unsigned npb,const unsigned npbok,const unsigned ppedim,unsigned &nnz,int *row,const float *divr,const float freeSurface,unsigned &numfreesurface)const{
  const unsigned matOrder=npb-npbok;
	
	for(unsigned p1=0;p1<npbok;p1++){
		if(divr[p1]<=freeSurface){
			row[p1]=0;
			numfreesurface++;
		}
    const unsigned nnzOld=nnz;
    nnz += row[p1]+1;
    row[p1] = nnzOld;
  }

	for(unsigned p1=npb;p1<np;p1++){
		const unsigned oi=p1-matOrder;
		if(divr[p1]<=freeSurface){
			row[oi]=0;
			numfreesurface++;
		}
    const unsigned nnzOld=nnz;
    nnz += row[oi]+1;
    row[oi] = nnzOld;
  }

  row[ppedim]=nnz;
}

//===============================================================================
///Populate matrix with values
//===============================================================================
template<TpKernel tker> void JSphCpu::PopulateMatrixAFluid(unsigned np,unsigned npb,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,const tfloat4 *velrhop,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,float *divr,double *matrixInd,int *row,int *col,
  double *matrixb,const unsigned *idpc,const word *code,const float freesurface,const double rhoZero,const unsigned matOrder,const double dt)const{

	#ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(npb);p1<int(np);p1++)if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
    //-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tdouble3 posp1=pos[p1];
    const tfloat3 velp1=TFloat3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
		//-Particle order in Matrix
		unsigned oi=p1-matOrder;
		const unsigned diag=row[oi];
		col[diag]=oi;
		unsigned index=diag+1;
		double divU=0;
		double Neumann=0;
		const unsigned Correctp1=p1-npb;
		tfloat3 dwx=dwxcorr[Correctp1]; //  dwx.x   dwx.y   dwx.z
    tfloat3 dwy=dwycorr[Correctp1]; //  dwy.x   dwy.y   dwy.z
    tfloat3 dwz=dwzcorr[Correctp1]; //  dwz.x   dwz.y   dwz.z
	  if(divr[p1]>freesurface){  
      //FLUID INTERACTION
      //-Obtain interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

			//-Search for neighbours in adjacent cells / Busqueda de vecinos en celdas adyacentes.
			for(int fluid=0;fluid<=1;fluid++){
				//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
				float massp2=(fluid? MassFluid:MassBound); //-Contiene masa de particula segun sea bound o fluid.
				const float volume=massp2/RhopZero; //Volume of particle j
				for(int z=zini;z<zfin;z++){
					const int zmod=(nc.w)*z+(cellinitial*fluid); //-Sum from start of fluid or boundary cells / Le suma donde empiezan las celdas de fluido o bound.
					for(int y=yini;y<yfin;y++){
						int ymod=zmod+nc.x*y;
						const unsigned pini=beginendcell[cxini+ymod];
						const unsigned pfin=beginendcell[cxfin+ymod];

						//-Interactions
						//------------------------------------------------
						for(unsigned p2=pini;p2<pfin;p2++){
							const float drx=float(posp1.x-pos[p2].x);
							const float dry=float(posp1.y-pos[p2].y);
							const float drz=float(posp1.z-pos[p2].z);
							const float rr2=drx*drx+dry*dry+drz*drz;
							if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
  							unsigned oj=p2;
								if(fluid) oj-=matOrder;

								//-Wendland kernel.
								float frx,fry,frz;
								if(tker==KERNEL_Quintic) GetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
								else if(tker==KERNEL_Wendland) GetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
			
								//===== Laplacian operator =====
								const float rDivW=drx*frx+dry*fry+drz*frz;
								float temp=2.0f*rDivW/(RhopZero*(rr2+Eta2));
								matrixInd[index]=double(-temp*volume);
								col[index]=oj;
								matrixInd[diag]+=double(temp*volume);
								index++;

								//=====Divergence of velocity==========
								float dvx=velp1.x-velrhop[p2].x, dvy=velp1.y-velrhop[p2].y, dvz=velp1.z-velrhop[p2].z;
							
								const float temp_x=frx*dwx.x+fry*dwy.x+frz*dwz.x;
								const float temp_y=frx*dwx.y+fry*dwy.y+frz*dwz.y;
								const float temp_z=frx*dwx.z+fry*dwy.z+frz*dwz.z;
								const double tempDivU=double(dvx*temp_x+dvy*temp_y+dvz*temp_z);
								divU-=double(volume*tempDivU);

								if(!fluid){
									double dist=pos[p2].z-MirrorPosc[idpc[p2]].z;
									double temp2=temp*RhopZero*Gravity.z*dist;
									Neumann+=double(temp2*volume);
								}
							}  
						}
					}
				}
			}
	  }
    else matrixInd[diag]=1.0;

		matrixb[oi]=Neumann+divU/dt;
  }
}

template<TpKernel tker> void JSphCpu::PopulateMatrixABound(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
  const tdouble3 *pos,double *matrixInd,int *row,int *col,double *matrixb,float *divr,const float freesurface,const unsigned *idpc,const word *code,const tdouble3 *mirrorPos,
	const unsigned *mirrorCell,tfloat4 *mls,tfloat3 gravity)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
    const unsigned idp1=idpc[p1];

		//-Obtain data of particle p1 / Obtiene datos de particula p1.
    const tdouble3 posp1=mirrorPos[idp1];
		const tfloat4 mlsp1=mls[p1];
		//-Particle order in Matrix
		unsigned oi=p1;
		const unsigned diag=row[oi];
		col[diag]=oi;
		unsigned index=diag+1;
	  if(divr[p1]>freesurface){  
			matrixInd[diag]=1.0;
      //FLUID INTERACTION
      //-Obtain interaction limits / Obtiene limites de interaccion
      int cxini,cxfin,yini,yfin,zini,zfin;
      GetInteractionCells(mirrorCell[idp1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

			//===== Get mass of particle p2  /  Obtiene masa de particula p2 ===== 
      float massp2=(boundp2? MassBound: MassFluid); //-Contiene masa de particula segun sea bound o fluid.
			const float volume=massp2/RhopZero; //Volume of particle j
			
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
						const float drx=float(posp1.x-pos[p2].x);
						const float dry=float(posp1.y-pos[p2].y);
						const float drz=float(posp1.z-pos[p2].z);
						const float rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
  	          unsigned oj=(p2-Npb)+NpbOk;
			
							float Wab;
							if(tker==KERNEL_Quintic) Wab=GetKernelQuinticWab(rr2);
							else if(tker==KERNEL_Wendland) Wab=GetKernelWendlandWab(rr2);
							const float temp=(mlsp1.w+mlsp1.x*drx+mlsp1.y*dry+mlsp1.z*drz)*Wab;
							matrixInd[index]=double(-temp*volume);
              col[index]=oj;
							index++;
		        }  
          }	
        }
	    }
	  }
    else matrixInd[diag]=1.0;
  }
}

void JSphCpu::PopulateMatrix(TpKernel tkernel,unsigned np,unsigned npb,unsigned npbok,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell,
  const tdouble3 *pos,const tfloat4 *velrhop,tfloat3 *dwxcorr,tfloat3 *dwycorr,tfloat3 *dwzcorr,float *divr,double *matrixInd,int *row,int *col,
  double *matrixb,const unsigned *idpc,const word *code,const float freesurface,const double rhoZero,const unsigned matOrder,const double dt,const tdouble3 *mirrorPos,
	const unsigned *mirrorCell,tfloat4 *mls,tfloat3 gravity)const{
 
	if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
		PopulateMatrixAFluid<tker> (np,npb,nc,hdiv,cellinitial,beginendcell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,divr,matrixInd,row,col,matrixb,idpc,code,freesurface,rhoZero,matOrder,dt);//-Fluid-Fluid
		PopulateMatrixABound<tker> (npbok,0,nc,hdiv,cellinitial,beginendcell,cellzero,pos,matrixInd,row,colInd,matrixb,divr,freesurface,idpc,code,mirrorPos,mirrorCell,mls,gravity);
	}
	else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
		PopulateMatrixAFluid<tker> (np,npb,nc,hdiv,cellinitial,beginendcell,cellzero,dcell,pos,velrhop,dwxcorr,dwycorr,dwzcorr,divr,matrixInd,row,col,matrixb,idpc,code,freesurface,rhoZero,matOrder,dt);//-Fluid-Fluid
		PopulateMatrixABound<tker> (npbok,0,nc,hdiv,cellinitial,beginendcell,cellzero,pos,matrixInd,row,col,matrixb,divr,freesurface,idpc,code,mirrorPos,mirrorCell,mls,gravity);
	}
}

void JSphCpu::PopulatePeriodic(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
  const tdouble3 *pos,double *matrixInd,int *row,int *col,
  const unsigned *idpc,const word *code,const unsigned *dCell)const{

  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  const int pfin=int(pinit+n);

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++)if(CODE_GetSpecialValue(code[p1])==CODE_PERIODIC){
    const unsigned idp1=idpc[p1];

		//-Particle order in Matrix
		unsigned oi=p1;
		if(p1>=int(Npb)) oi=(oi-Npb)+NpbOk;
		const unsigned diag=row[oi];
		col[diag]=oi;
		unsigned index=diag+1;
		tdouble3 rpos=pos[p1];
		double dx=rpos.x-MapRealPosMin.x;
		double dy=rpos.y-MapRealPosMin.y;
		double dz=rpos.z-MapRealPosMin.z;
		
		//-Adjust position according to periodic conditions and compare domain limits / Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
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
		rpos=TDouble3(dx,dy,dz)+MapRealPosMin;

		dx=rpos.x-DomPosMin.x;
		dy=rpos.y-DomPosMin.y;
		dz=rpos.z-DomPosMin.z;
		unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
		const unsigned Cell=PC__Cell(DomCellCode,cx,cy,cz);

		//FLUID INTERACTION
    //-Obtain interaction limits / Obtiene limites de interaccion
    int cxini,cxfin,yini,yfin,zini,zfin;
    GetInteractionCells(Cell,hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

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
          if(idp1==Idpc[p2]){
  	        unsigned oj=p2;
						if(p2>=int(Npb)) oj=(oj-Npb)+NpbOk;

            matrixInd[index]=1.0;
            col[index]=oj;
			      matrixInd[diag]=-1.0; 
						break;
          }	
        }
	    }
	  }
  }
}

void JSphCpu::FreeSurfaceMark(unsigned n,unsigned pinit,float *divr,double *matrixInd,double *matrixb,
  int *row,const unsigned *idpc,const word *code,const float shiftoffset,const unsigned matOrder,const float freeSurface)const{
  const int pfin=int(pinit+n);
  
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(pinit);p1<pfin;p1++) if(CODE_GetSpecialValue(Codec[p1])!=CODE_PERIODIC){   
	  //-Particle order in Matrix
	  unsigned oi=p1-matOrder;
    const int Mark=row[oi]+1;
		const float divrp1=divr[p1];
    if(divrp1>=freeSurface && divrp1<=freeSurface+shiftoffset){
      double alpha=0.5*(1.0-cos(PI*double(divrp1-freeSurface)/shiftoffset));

      matrixb[oi]=matrixb[oi]*alpha;

      for(int index=Mark;index<row[oi+1];index++) matrixInd[index]=matrixInd[index]*alpha;
    }
  }
}

//===============================================================================
///Reorder pressure for particles
//===============================================================================
void JSphCpu::PressureAssign(unsigned np,unsigned npbok,const tdouble3 *pos,tfloat4 *velrhop,
  const unsigned *idpc,double *x,const word *code,const unsigned npb,float *divr,tfloat3 gravity)const{

  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=0;p1<int(npbok);p1++){
		double dist=MirrorPosc[Idpc[p1]].z-Posc[p1].z;
		double Neumann=double(RhopZero)*abs(Gravity.z)*dist;
		velrhop[p1].w=float(x[p1]+Neumann);
		if(!NegativePressureBound)if(velrhop[p1].w<0)velrhop[p1].w=0.0;
  }

	 #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(npb);p1<int(np);p1++) velrhop[p1].w=float(x[(p1-npb)+npbok]);
}

#ifndef _WITHGPU
template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void JSphCpu::run_solver(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,double *matrixx,const unsigned ppedim){ 
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

	#ifdef _WITHOMP
			#pragma omp parallel for schedule (static)
	#endif
	for(int i=0;i<int(ppedim);i++){
		matrixx[i]=result[i];
	}
}

void JSphCpu::solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,double *matrixa,double *matrixb,double *matrixx,int *row,int *col,const unsigned ppedim,const unsigned nnz,const unsigned numfreesurface){
    viennacl::context ctx;
   
    typedef double ScalarType;

		viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix;
    vcl_compressed_matrix.set(&row[0],&col[0],&matrixa[0],ppedim,ppedim,nnz);

    viennacl::vector<ScalarType> vcl_vec(vcl_compressed_matrix.size1(),ctx);
    #ifdef _WITHOMP
				#pragma omp parallel for schedule (static)
		#endif
		for(int i=0;i<int(ppedim);i++){
			vcl_vec[i]=matrixb[i];
		}

    viennacl::linalg::bicgstab_tag bicgstab(tolerance,iterations);

		if(viennacl::linalg::norm_2(vcl_vec)){
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
					amg_tag_agg_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION);
					if(tamginter==AMGINTER_AG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION); Log->Printf("INTERPOLATION: AGGREGATION ");}
					else if(tamginter==AMGINTER_SAG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION); Log->Printf("INTERPOLATION: SMOOTHED AGGREGATION");}
					amg_tag_agg_pmis.set_strong_connection_threshold(strongconnection);
					amg_tag_agg_pmis.set_jacobi_weight(jacobiweight);
					amg_tag_agg_pmis.set_presmooth_steps(presmooth);
					amg_tag_agg_pmis.set_postsmooth_steps(postsmooth); 
					amg_tag_agg_pmis.set_coarsening_cutoff(numfreesurface); 
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
		else Log->Printf("norm(b)=0");
}    
#endif

//==============================================================================
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template <TpKernel tker,TpFtMode ftmode> void JSphCpu::InteractionForcesShifting
  (unsigned np,unsigned npb,tint4 nc,int hdiv,unsigned cellinitial,float visco
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const tdouble3 *pos,tfloat4 *velrhop,const word *code,const unsigned *idp
  ,TpShifting tshifting,tfloat3 *shiftpos,tfloat3 *tensile,float *divr,const float tensileN,const float tensileR)const
{
  const bool boundp2=(!cellinitial); //-Interaction with type boundary (Bound) /  Interaccion con Bound.
  //-Initialize viscth to calculate viscdt maximo con OpenMP / Inicializa viscth para calcular visdt maximo con OpenMP.
  float viscth[MAXTHREADS_OMP*STRIDE_OMP];
  for(int th=0;th<OmpThreads;th++)viscth[th*STRIDE_OMP]=0;
  //-Initial execution with OpenMP / Inicia ejecucion con OpenMP.
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p1=int(npb);p1<int(np);p1++){
		float Wab1;
		if(tker==KERNEL_Quintic) Wab1=GetKernelQuinticWab(float(Dp*Dp));
		else if(tker==KERNEL_Wendland) Wab1=GetKernelWendlandWab(float(Dp*Dp));
    tfloat3 shiftposp1=TFloat3(0);
    float divrp1=0;
		tfloat3 sumtensile=TFloat3(0);

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
        for(unsigned p2=pini;p2<pfin;p2++)if(divr[p2]!=-1.0||CODE_GetTypeValue(code[p2])==0){
          const float drx=float(posp1.x-pos[p2].x);
					const float dry=float(posp1.y-pos[p2].y);
					const float drz=float(posp1.z-pos[p2].z);
					const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2 && rr2>=ALMOSTZERO){
            //-Wendland kernel.
            float frx,fry,frz;
            if(tker==KERNEL_Quintic) GetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
						else if(tker==KERNEL_Wendland) GetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
			
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
						float Wab;
						if(tker==KERNEL_Quintic) Wab=GetKernelQuinticWab(rr2);
						else if(tker==KERNEL_Wendland) Wab=GetKernelWendlandWab(rr2);
						const float tensile=tensileN*powf((Wab/Wab1),tensileR);
            
						shiftposp1.x+=volumep2*frx; //-For boundary do not use shifting / Con boundary anula shifting.
            shiftposp1.y+=volumep2*fry;
            shiftposp1.z+=volumep2*frz;
						sumtensile.x+=volumep2*tensile*frx;
						sumtensile.y+=volumep2*tensile*fry;
						sumtensile.z+=volumep2*tensile*frz;
            divrp1-=volumep2*(drx*frx+dry*fry+drz*frz);
          }
        }
      }
    }
    divr[p1]+=divrp1;  
		const unsigned Correctp1=p1-npb;
    shiftpos[Correctp1]=shiftpos[Correctp1]+shiftposp1; 
		tensile[Correctp1]=tensile[Correctp1]+sumtensile;
  }
}

void JSphCpu::Interaction_Shifting
  (TpKernel tkernel,unsigned np,unsigned npb,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell,const tdouble3 *pos
  ,tfloat4 *velrhop,const unsigned *idp,const word *code
  ,tfloat3 *shiftpos,tfloat3 *tensile,float *divr,const float tensileN,const float tensileR)const
{
	const unsigned npf=np-npb;
  const tint4 nc=TInt4(int(ncells.x),int(ncells.y),int(ncells.z),int(ncells.x*ncells.y));
  const tint3 cellzero=TInt3(cellmin.x,cellmin.y,cellmin.z);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int hdiv=(CellMode==CELLMODE_H? 2: 1);

  if(npf){
		if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
			if(!WithFloating){                   const TpFtMode ftmode=FTMODE_None;
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}else if(!UseDEM){                   const TpFtMode ftmode=FTMODE_Sph;
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}else{                               const TpFtMode ftmode=FTMODE_Dem; 
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}
		}
		else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
			if(!WithFloating){                   const TpFtMode ftmode=FTMODE_None;
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}else if(!UseDEM){                   const TpFtMode ftmode=FTMODE_Sph;
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}else{                               const TpFtMode ftmode=FTMODE_Dem; 
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,cellfluid,Visco                 ,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
				InteractionForcesShifting<tker,ftmode>(np,npb,nc,hdiv,0        ,Visco*ViscoBoundFactor,begincell,cellzero,dcell,pos,velrhop,code,idp,TShifting,shiftpos,tensile,divr,tensileN,tensileR);
			}
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
			const tfloat3 shift=dWxCorrShiftPos[p-npb];
      //-Calculate displacement and update position / Calcula desplazamiento y actualiza posicion.
      double dx=double(shift.x);
      double dy=double(shift.y);
      double dz=double(shift.z); 
      bool outrhop=false;//(rhopnew<RhopOutMin||rhopnew>RhopOutMax);
      UpdatePos(PosPrec[p],dx,dy,dz,outrhop,p,Posc,Dcellc,Codec);
    }
  }
}