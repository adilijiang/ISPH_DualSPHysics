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

#ifndef _JSphGpu_
#define _JSphGpu_

#include "Types.h"
#include "JSphTimersGpu.h"
#include "JSph.h"
#include <string>

class JPartsOut;
class JArraysGpu;
class JCellDivGpu;
class JBlockSizeAuto;

class JSphGpu : public JSph
{
private:
  JCellDivGpu* CellDiv;

public:
  typedef struct {
    unsigned forcesfluid;
    unsigned forcesbound;
    unsigned forcesdem;
  }StBlockSizes;

protected:
  StBlockSizes BlockSizes;        ///<Almacena configuracion de BlockSizes. Stores configuration of BlockSizes
  std::string BlockSizesStr;      ///<Almacena configuracion de BlockSizes en texto. Stores configuration of BlockSizes in text form
  TpBlockSizeMode BlockSizeMode;  ///<Modes for BlockSize selection.
  JBlockSizeAuto *BsAuto;         ///<Object to calculate the optimum BlockSize for particle interactions.
  
  //-Vars. con informacion del hardware GPU.
  //-Variables with information for the GPU hardware
  int GpuSelect;          ///<ES: Gpu seleccionada (-1:sin seleccion). EN: GPU Selection (-1:no selection)
  std::string GpuName;    ///<ES: Nombre de la gpu seleccionada. EN: Name of the selected GPU
  size_t GpuGlobalMem;    ///<ES: Tamaño de memoria global en bytes. EN: Size of global memory in bytes
  unsigned GpuSharedMem;  ///<ES: Tamaño de memoria shared por bloque en bytes. EN: Size of shared memory for each block in bytes
  unsigned GpuCompute;    ///<Compute capability: 10,11,12,20 

  std::string RunMode;     ///<ES: Almacena modo de ejecucion (simetria,openmp,balanceo,...). EN: Stores execution mode (symmetry,OpenMP,balance)

  //-Numero de particulas del dominio.
  //-Number of particles in the domain
  unsigned Np;     ///<ES: Numero total de particulas (incluidas las duplicadas periodicas). EN: Total number of particles (including duplicate periodic particles)
  unsigned Npb;    ///<ES: Numero de particulas contorno (incluidas las contorno periodicas). EN: Number of boundary particles (including periodic boundaries)
  unsigned NpbOk;  ///<ES: Numero de particulas contorno cerca del fluido (incluidas las contorno periodicas). EN: Number of boundary particles interacting the fluid (including the periodic bounaries)

  unsigned NpfPer;   ///<ES: Numero de particulas fluidas-floating periodicas. EN: Number of periodic particles (fluid-floating)
  unsigned NpbPer;   ///<ES: Numero de particulas contorno periodicas. EN: Number of periodic boundary particles
  unsigned NpfPerM1; ///<ES: Numero de particulas fluidas-floating periodicas (valores anteriores). EN: Number of fluid-floating periodic particles (previous values)
  unsigned NpbPerM1; ///<ES: Numero de particulas contorno periodicas (valores anteriores). EN: Number of periodic boundary particles (previous values)

  bool WithFloating;
  bool BoundChanged; ///<ES: Indica si el contorno seleccionado a cambiado desde el ultimo divide. EN: Indicates if a selected boundary particle has changed since the last time step

  unsigned CpuParticlesSize; ///<ES: Numero de particulas para las cuales se reservo memoria en cpu. EN: Number of particles for which CPU memory was allocated
  llong MemCpuParticles;     ///<ES: Mermoria reservada para vectores de datos de particulas. EN: Allocated CPU memory for arrays with particle data

  //-Variables con datos de las particulas para ejecucion (size=ParticlesSize).
  //-Variables holding particle data for the execution (size=ParticlesSize)
  unsigned *Idp;   ///<ES: Identificador de particula. EN: Particle identifier
  word *Code;      ///<ES: Indica el grupo de las particulas y otras marcas especiales. EN: Indicates particle group and other special marks
  unsigned *Dcell; ///<ES: Celda dentro de DomCells codificada con DomCellCode. EN: Cell within DomCells encoded with DomCellCode.
  tdouble2 *Posxy;
  double *Posz;
  tfloat4 *Velrhop;

  //-Variables auxiliares para conversion (size=ParticlesSize).
  //-Auxiliary variables for the conversion (size=ParticlesSize)
  tdouble3 *AuxPos;
  tfloat3 *AuxVel; 
  float *AuxRhop;

  unsigned GpuParticlesSize;  ///<ES: Numero de particulas para las cuales se reservo memoria en gpu. EN: Number of particles for which GPU memory was allocated
  llong MemGpuParticles;      ///<ES: Mermoria reservada para vectores de datos de particulas. EN: Allocated GPU memory for arrays with particle data
	llong MemGpuMatrix;      ///<ES: Mermoria reservada para vectores de datos de particulas. EN: Allocated GPU Matrix memory for arrays with particle data
  llong MemGpuFixed;          ///<ES: Mermoria reservada en AllocGpuMemoryFixed. EN: Allocated memory in AllocGpuMemoryFixed

  //-Posicion de particula segun id para motion.
  //-Particle position according to the identifier for the motion
  unsigned *RidpMoveg;	///<ES: Solo para boundary moving particles [CaseNmoving] y cuando CaseNmoving!=0 EN: Only for moving boundary particles [CaseNmoving] and when CaseNmoving!=0

  //-Lista de arrays en Gpu para particulas.
  //-List of arrays in the GPU gor the particles
  JArraysGpu* ArraysGpu;
  //-Variables con datos de las particulas para ejecucion (size=ParticlesSize).
  //-Variables holding particle data for the execution (size=ParticlesSize)
  unsigned *Idpg;   ///<ES: Identificador de particula. EN: Particle identifier
  word *Codeg;      ///<ES: Indica el grupo de las particulas y otras marcas especiales. EN: Indicates paricle group and other special marks
  unsigned *Dcellg; ///<ES: Celda dentro de DomCells codificada con DomCellCode. EN: Cell within DomCells encoded within DomCellCode
  double2 *Posxyg;
  double *Poszg;
  float4 *Velrhopg;
    
  //-Vars. para compute step: SYMPLECTIC
  //-Variables for compute step: Symplectic
  double2 *PosxyPreg;  ///<ES: Sympletic: para guardar valores en predictor EN: Symplectic: for maintaining predictor values
  double *PoszPreg;
  float4 *VelrhopPreg;
  double DtPre;   

  //-Variables for floating bodies.
  unsigned *FtRidpg;         ///<Identifier to access the particles of the floating object [CaseNfloat] in GPU.
  float *FtoMasspg;          ///<Mass of the particle for each floating body [FtCount] in GPU (used in interaction forces).

  float4 *FtoDatag;    ///<ES: Datos constantes de floatings {pini_u,np_u,radius_f,mass_f} [FtCount] //__device__ int __float_as_int(float x) //__device__ float __int_as_float(int x) EN: Constant data of floating bodies {pini_u,np_u,radius_f,mass_f} [FtCount] //__device__ int __float_as_int (float x) //__device__ float __int_as_float(int x)
  float3 *FtoForcesg;  ///<ES: Almacena fuerzas de floatings {face_f3,fomegavel_f3} equivalente a JSphCpu::FtoForces [FtCount]. EN: Stores forces for the floating bodies {face_f3,fomegavel_f3} equivalent to JSphCpu::FtoForces [FtCount]

  double3 *FtoCenterg; ///<ES: Mantiene centro de floating. [FtCount]  EN: Maintains centre of floating bodies [Ftcount]
  float3 *FtoVelg;     ///<ES: Mantiene vel de floating. [FtCount] EN: Maintains velocity of floating bodies [FtCount]
  float3 *FtoOmegag;   ///<ES: Mantiene omega de floating. [FtCount] EN: Maintains omega of floating bodies [FtCount]

  StFtoForces *FtoForces; ///<ES: Almacena fuerzas de floatings en CPU [FtCount]. EN: Stores forces for floating bodies on the CPU
  tdouble3 *FtoCenter;    ///<ES: Almacena centro de floating en CPU. [FtCount]  EN: Stores centre of floating bodies on the CPU

  //-Variables for DEM. (DEM)
  float4 *DemDatag;          ///<Data of the object {mass, (1-poisson^2)/young, kfric, restitu} in GPU [DemObjsSize].

  //-Vars. para computo de fuerzas.
  //-Variables for computing forces
 
  //float *ViscDtg;
  float3 *Aceg;      ///<ES: Acumula fuerzas de interaccion EN: Accumulates acceleration of the particles

  double VelMax;      ///<Maximum value of Vel[] sqrt(vel.x^2 + vel.y^2 + vel.z^2) computed in PreInteraction_Forces().
  double AceMax;      ///<Maximum value of Ace[] (ace.x^2 + ace.y^2 + ace.z^2) computed in Interaction_Forces().
  float ViscDtMax;    ///<ES: Valor maximo de ViscDt calculado en Interaction_Forces(). EN: Maximum value of ViscDt computed in Interaction_Forces()

  TimersGpu Timers;
  
  void InitVars();
  void RunExceptionCuda(const std::string &method,const std::string &msg,cudaError_t error);
  void CheckCudaError(const std::string &method,const std::string &msg);

  void FreeGpuMemoryFixed();
  void AllocGpuMemoryFixed();
  void FreeCpuMemoryParticles();
  void AllocCpuMemoryParticles(unsigned np);
  void FreeGpuMemoryParticles();
  void AllocGpuMemoryParticles(unsigned np,float over);

  void ResizeGpuMemoryParticles(unsigned np);
  void ReserveBasicArraysGpu();

  template<class T> T* TSaveArrayGpu(unsigned np,const T *datasrc)const;
  word*        SaveArrayGpu(unsigned np,const word        *datasrc)const{ return(TSaveArrayGpu<word>       (np,datasrc)); }
  unsigned*    SaveArrayGpu(unsigned np,const unsigned    *datasrc)const{ return(TSaveArrayGpu<unsigned>   (np,datasrc)); }
  float*       SaveArrayGpu(unsigned np,const float       *datasrc)const{ return(TSaveArrayGpu<float>      (np,datasrc)); }
  float4*      SaveArrayGpu(unsigned np,const float4      *datasrc)const{ return(TSaveArrayGpu<float4>     (np,datasrc)); }
  double*      SaveArrayGpu(unsigned np,const double      *datasrc)const{ return(TSaveArrayGpu<double>     (np,datasrc)); }
  double2*     SaveArrayGpu(unsigned np,const double2     *datasrc)const{ return(TSaveArrayGpu<double2>    (np,datasrc)); }
  tsymatrix3f* SaveArrayGpu(unsigned np,const tsymatrix3f *datasrc)const{ return(TSaveArrayGpu<tsymatrix3f>(np,datasrc)); }
  unsigned*    SaveArrayGpu_Uint(unsigned np,const unsigned *datasrc)const;
  template<class T> void TRestoreArrayGpu(unsigned np,T *data,T *datanew)const;
  void RestoreArrayGpu(unsigned np,word        *data,word        *datanew)const{ TRestoreArrayGpu<word>       (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,unsigned    *data,unsigned    *datanew)const{ TRestoreArrayGpu<unsigned>   (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,float       *data,float       *datanew)const{ TRestoreArrayGpu<float>      (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,float4      *data,float4      *datanew)const{ TRestoreArrayGpu<float4>     (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,double      *data,double      *datanew)const{ TRestoreArrayGpu<double>     (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,double2     *data,double2     *datanew)const{ TRestoreArrayGpu<double2>    (np,data,datanew); }
  void RestoreArrayGpu(unsigned np,tsymatrix3f *data,tsymatrix3f *datanew)const{ TRestoreArrayGpu<tsymatrix3f>(np,data,datanew); }
  void RestoreArrayGpu_Uint(unsigned np,unsigned *data,unsigned *datanew)const;

  llong GetAllocMemoryCpu()const;
  llong GetAllocMemoryGpu()const;
  void PrintAllocMemory(llong mcpu,llong mgpu,llong mMatrixgpu)const;

  void ConstantDataUp();
  void ParticlesDataUp(unsigned n);
  unsigned ParticlesDataDown(unsigned n,unsigned pini,bool code,bool cellorderdecode,bool onlynormal);
  
  void SelecDevice(int gpuid);
  void ConfigBlockSizes(bool usezone,bool useperi);

  void ConfigRunMode(std::string preinfo);
  void ConfigCellDiv(JCellDivGpu* celldiv){ CellDiv=celldiv; }
  void InitFloating();
  void InitRun();

  void AddAccInput();

  void PreInteractionVars_Forces(TpInter tinter,unsigned np,unsigned npb);
  void Stage1PreInteraction_ForcesPre(double dt);
  void Stage3PreInteraction_ForcesCor(double dt);

  void ComputeSymplecticPre(double dt);
  void ComputeSymplecticCorr(double dt);
  double DtVariable(bool final);
  void RunShifting(double dt);

  void RunMotion(double stepdt);

  void ShowTimers(bool onlyfile=false);
  void GetTimersInfo(std::string &hinfo,std::string &dinfo)const;
  unsigned TimerGetCount()const{ return(TmgGetCount()); }
  bool TimerIsActive(unsigned ct)const{ return(TmgIsActive(Timers,(CsTypeTimerGPU)ct)); }
  float TimerGetValue(unsigned ct)const{ return(TmgGetValue(Timers,(CsTypeTimerGPU)ct)); }
  const double* TimerGetPtrValue(unsigned ct)const{ return(TmgGetPtrValue(Timers,(CsTypeTimerGPU)ct)); }
  std::string TimerGetName(unsigned ct)const{ return(TmgGetName((CsTypeTimerGPU)ct)); }
  std::string TimerToText(unsigned ct)const{ return(JSph::TimerToText(TimerGetName(ct),TimerGetValue(ct))); }

  ///////////////////////////////////////////////
  //PPE Functions, variables, Kernel Correction//
  ///////////////////////////////////////////////
  unsigned *MirrorCellg;
	double3 *MirrorPosg;
  float3 *dWxCorrg; //Kernel correction in the x direction
  float3 *dWyCorrg; //Kernel correction in the y direction
  float3 *dWzCorrg; //Kernel correction in the z direction
  float4 *MLSg;
	float3 *sumFrg;
	float *Divrg; //Divergence of position
	float3 *ShiftPosg;
	float *Taog;
	float3 *Normal;
	float3 *smoothNormal;
  //matrix variables 
  double *bg;
  double *ag;
  unsigned PPEDim;
  unsigned *colIndg;
  unsigned *rowIndg;
  double *Xg;
	float PaddleAccel;
  unsigned *counterNnzGPU;
  unsigned *counterNnzCPU;
	unsigned *NumFreeSurfaceGPU;
	unsigned *NumFreeSurfaceCPU;

	float3 *BoundaryNormal;
	double TFocus;
	double *Focussed_f;
	double *Focussed_Sp;
	double *Focussed_K;
	double *Focussed_Stroke;
	double *Focussed_Apm;
	double *Focussed_Phi;

	void RegularWavePiston(const double L,const double H,const double D,double &velocity);
	void FocussedWavePiston(const unsigned nspec,const double *f,const double *stroke,const double *phi,const double T,double &velocity);
	void FocussedWavePistonSpectrum(const double H,const double D,const double fp,const double focalpoint,const unsigned nspec,const double gamma,double *f,double *K,double *Sp,double *Stroke,double *Apm,double *Phi);

  void MatrixASetup(const unsigned np,const unsigned npb,const unsigned npbok,
		const unsigned ppedim,unsigned int *rowGpu,const float *divr,const float freesurface,unsigned &nnz,unsigned &numFreeSurface,const float boundaryfs);

	void Shift(double dt,const unsigned bsfluid);
	double ComputeVariable();

public:
  JSphGpu(bool withmpi);
  ~JSphGpu();
};

#endif


