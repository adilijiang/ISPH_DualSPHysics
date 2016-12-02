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

#ifndef _JSphCpu_
#define _JSphCpu_

#include "Types.h"
#include "JSphTimersCpu.h"
#include "JSph.h"
#include <string>

class JPartsOut;
class JArraysCpu;
class JCellDivCpu;

class JSphCpu : public JSph
{
private:
  JCellDivCpu* CellDiv;

protected:
  int OmpThreads;        //-Max number of OpenMP threads in execution on CPU host (minimum 1) / Numero maximo de hilos OpenMP en ejecucion por host en CPU (minimo 1).
  std::string RunMode;   //-Overall mode of execution (symmetry, openmp, load balancing) /  Almacena modo de ejecucion (simetria,openmp,balanceo,...).

  //-Number of particles in domain / Numero de particulas del dominio.
  unsigned Np;     //-Total number of particles (including periodic duplicates) / Numero total de particulas (incluidas las duplicadas periodicas).
  unsigned Npb;    //-Total number of boundary particles (including periodic boundaries) / Numero de particulas contorno (incluidas las contorno periodicas).
  unsigned NpbOk;  //-Total number of boundary particles near fluid (including periodic duplicates) / Numero de particulas contorno cerca del fluido (incluidas las contorno periodicas).

  unsigned NpfPer;   //-Number of periodic floating-fluid particles / Numero de particulas fluidas-floating periodicas.
  unsigned NpbPer;   //-Number of periodic boundary particles / Numero de particulas contorno periodicas.
  unsigned NpfPerM1; //-Number of periodic floating-fluid particles (previous values) / Numero de particulas fluidas-floating periodicas (valores anteriores).
  unsigned NpbPerM1; //-Number of periodic boundary particles (previous values) / Numero de particulas contorno periodicas (valores anteriores).

  bool WithFloating;
  bool BoundChanged; //-Indicates if selected boundary has changed since last call of divide / Indica si el contorno seleccionado a cambiado desde el ultimo divide.

  unsigned CpuParticlesSize;  //-Number of particles with reserved memory on the CPU / Numero de particulas para las cuales se reservo memoria en cpu.
  llong MemCpuParticles;      //-Memory reserved for particles' vectors / Mermoria reservada para vectores de datos de particulas.
  llong MemCpuFixed;          //-Memory reserved in AllocMemoryFixed / Mermoria reservada en AllocMemoryFixed.

  //-Particle Position according to id / Posicion de particula segun id.
  unsigned *RidpMove;//-Only for moving boundary particles [CaseNmoving] and when CaseNmoving!=0 / Solo para boundary moving particles [CaseNmoving] y cuando CaseNmoving!=0 

  //-List of particle arrays on CPU / Lista de arrays en Cpu para particulas.
  JArraysCpu* ArraysCpu;
  //-Execution Variables for particles (size=ParticlesSize). / Variables con datos de las particulas para ejecucion (size=ParticlesSize).
  unsigned *Idpc;    //-Identifier of particle / Identificador de particula.
  word *Codec;       //-Indicator of group of particles & other special markers / Indica el grupo de las particulas y otras marcas especiales.
  unsigned *Dcellc;  //-Cells inside DomCells coded with DomCellCode / Celda dentro de DomCells codificada con DomCellCode.
  tdouble3 *Posc;
  tfloat4 *Velrhopc;
    
  //-Variables for compute step: SYMPLECTIC / Vars. para compute step: SYMPLECTIC
  tdouble3 *PosPrec;  //-Sympletic: in order to keep previous values / Sympletic: para guardar valores en predictor
  tfloat4 *VelrhopPrec;
  double DtPre;   

  //-Variables for floating bodies.
  unsigned *FtRidp;   ///<Identifier to access to the particles of the floating object [CaseNfloat].
  StFtoForces *FtoForces; //-Almacena fuerzas de floatings [FtCount].

  tfloat3 *Acec;      //-Sum of interaction forces / Acumula fuerzas de interaccion
  float *Deltac;      //-Adjusted sum with Delta-SPH with DELTA_DynamicExt / Acumula ajuste de Delta-SPH con DELTA_DynamicExt

  tfloat3 *ShiftPosc;    //-Particle displacement using Shifting.

  double VelMax;      ///<Maximum value of Vel[] sqrt(vel.x^2 + vel.y^2 + vel.z^2) computed in PreInteraction_Forces().
  double AceMax;      ///<Maximum value of Ace[] sqrt(ace.x^2 + ace.y^2 + ace.z^2) computed in Interaction_Forces().
  float ViscDtMax;    ///<Max value of ViscDt calculated in Interaction_Forces() / Valor maximo de ViscDt calculado en Interaction_Forces().

  TimersCpu Timers;


  void InitVars();

  void FreeCpuMemoryFixed();
  void AllocCpuMemoryFixed();
  void FreeCpuMemoryParticles();
  void AllocCpuMemoryParticles(unsigned np,float over);

  void ResizeCpuMemoryParticles(unsigned np);
  void ReserveBasicArraysCpu();

  template<class T> T* TSaveArrayCpu(unsigned np,const T *datasrc)const;
  word*        SaveArrayCpu(unsigned np,const word        *datasrc)const{ return(TSaveArrayCpu<word>       (np,datasrc)); }
  unsigned*    SaveArrayCpu(unsigned np,const unsigned    *datasrc)const{ return(TSaveArrayCpu<unsigned>   (np,datasrc)); }
  float*       SaveArrayCpu(unsigned np,const float       *datasrc)const{ return(TSaveArrayCpu<float>      (np,datasrc)); }
  tfloat4*     SaveArrayCpu(unsigned np,const tfloat4     *datasrc)const{ return(TSaveArrayCpu<tfloat4>    (np,datasrc)); }
  double*      SaveArrayCpu(unsigned np,const double      *datasrc)const{ return(TSaveArrayCpu<double>     (np,datasrc)); }
  tdouble3*    SaveArrayCpu(unsigned np,const tdouble3    *datasrc)const{ return(TSaveArrayCpu<tdouble3>   (np,datasrc)); }
  tsymatrix3f* SaveArrayCpu(unsigned np,const tsymatrix3f *datasrc)const{ return(TSaveArrayCpu<tsymatrix3f>(np,datasrc)); }
  template<class T> void TRestoreArrayCpu(unsigned np,T *data,T *datanew)const;
  void RestoreArrayCpu(unsigned np,word        *data,word        *datanew)const{ TRestoreArrayCpu<word>       (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,unsigned    *data,unsigned    *datanew)const{ TRestoreArrayCpu<unsigned>   (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,float       *data,float       *datanew)const{ TRestoreArrayCpu<float>      (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,tfloat4     *data,tfloat4     *datanew)const{ TRestoreArrayCpu<tfloat4>    (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,double      *data,double      *datanew)const{ TRestoreArrayCpu<double>     (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,tdouble3    *data,tdouble3    *datanew)const{ TRestoreArrayCpu<tdouble3>   (np,data,datanew); }
  void RestoreArrayCpu(unsigned np,tsymatrix3f *data,tsymatrix3f *datanew)const{ TRestoreArrayCpu<tsymatrix3f>(np,data,datanew); }
  void RestoreArrayCpu_Uint(unsigned np,unsigned *data,unsigned *datanew)const;


  llong GetAllocMemoryCpu()const;
  void PrintAllocMemory(llong mcpu)const;

  unsigned GetParticlesData(unsigned n,unsigned pini,bool cellorderdecode,bool onlynormal
    ,unsigned *idp,tdouble3 *pos,tfloat3 *vel,float *rhop,word *code);
  void ConfigOmp(const JCfgRun *cfg);

  void ConfigRunMode(const JCfgRun *cfg,std::string preinfo="");
  void ConfigCellDiv(JCellDivCpu* celldiv){ CellDiv=celldiv; }
  void InitFloating();
  void InitRun();

  void AddVarAcc();

  void PreInteractionVars_Forces(TpInter tinter,unsigned np,unsigned npb);
  void PreInteraction_Forces(TpInter tinter);
  void PosInteraction_Forces(TpInter tinter);

  inline void GetKernel(float rr2,float drx,float dry,float drz,float &frx,float &fry,float &frz)const;
  inline void GetKernelDouble(double rr2,double drx,double dry,double drz,double &frx,double &fry,double &frz)const;
  inline float GetKernelWab(float rr2)const;
  inline void GetInteractionCells(unsigned rcell
    ,int hdiv,const tint4 &nc,const tint3 &cellzero
    ,int &cxini,int &cxfin,int &yini,int &yfin,int &zini,int &zfin)const;

  template<TpFtMode ftmode> void InteractionForcesFluid
    (TpInter tinter, unsigned n,unsigned pini,tint4 nc,int hdiv,unsigned cellfluid,float visco
    ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
    ,const tdouble3 *pos,const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code,const unsigned *idp
    ,tfloat3 *ace,float *divr)const;

  template<bool psimple> void InteractionForcesDEM
    (unsigned nfloat,tint4 nc,int hdiv,unsigned cellfluid
    ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
    ,const unsigned *ftridp,const StDemData* demobjs
    ,const tdouble3 *pos,const tfloat3 *pspos,const tfloat4 *velrhop,const word *code,const unsigned *idp
    ,float &viscdt,tfloat3 *ace)const;

	  void Boundary_Velocity(TpSlipCond TSlipCond,unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
	  const unsigned *dcell,const tdouble3 *pos,tfloat4 *velrhop,const word *code,float *divr,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr)const;

  template<TpFtMode ftmode> void Interaction_ForcesT
    (TpInter tinter,unsigned np,unsigned npb,unsigned npbok
    ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
    ,const tdouble3 *pos,const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code,const unsigned *idp
    ,tfloat3 *ace,float *divr)const;

  void Interaction_Forces(TpInter tinter,unsigned np,unsigned npb,unsigned npbok
    ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
    ,const tdouble3 *pos,const tfloat4 *velrhop,const unsigned *idp,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,const word *code
    ,tfloat3 *ace,float *divr)const;

  void UpdatePos(tdouble3 pos0,double dx,double dy,double dz,bool outrhop,unsigned p,tdouble3 *pos,unsigned *cell,word *code)const;

  template<bool shift> void ComputeSymplecticPreT(double dt);
  void ComputeSymplecticPre(double dt);
  template<bool shift> void ComputeSymplecticCorrT(double dt);
  void ComputeSymplecticCorr(double dt);
  double DtVariable(bool final);

  void RunShifting(double dt);

  void CalcRidp(bool periactive,unsigned np,unsigned pini,unsigned idini,unsigned idfin,const word *code,const unsigned *idp,unsigned *ridp)const;
  void MoveLinBound(unsigned np,unsigned ini,const tdouble3 &mvpos,const tfloat3 &mvvel,const unsigned *ridp,tdouble3 *pos,unsigned *dcell,tfloat4 *velrhop,word *code)const;
  void MoveMatBound(unsigned np,unsigned ini,tmatrix4d m,double dt,const unsigned *ridpmv,tdouble3 *pos,unsigned *dcell,tfloat4 *velrhop,word *code)const;
  void RunMotion(double stepdt);

  void ShowTimers(bool onlyfile=false);
  void GetTimersInfo(std::string &hinfo,std::string &dinfo)const;
  unsigned TimerGetCount()const{ return(TmcGetCount()); }
  bool TimerIsActive(unsigned ct)const{ return(TmcIsActive(Timers,(CsTypeTimerCPU)ct)); }
  float TimerGetValue(unsigned ct)const{ return(TmcGetValue(Timers,(CsTypeTimerCPU)ct)); }
  const double* TimerGetPtrValue(unsigned ct)const{ return(TmcGetPtrValue(Timers,(CsTypeTimerCPU)ct)); }
  std::string TimerGetName(unsigned ct)const{ return(TmcGetName((CsTypeTimerCPU)ct)); }
  std::string TimerToText(unsigned ct)const{ return(JSph::TimerToText(TimerGetName(ct),TimerGetValue(ct))); }

  ///////////////////////////////////////////////
  //PPE Functions, variables, Kernel Correction//
  ///////////////////////////////////////////////
  unsigned *Irelationc; //The closest fluid particle, j, for a boundary particle, i
  tdouble3 *dWxCorr; //Kernel correction in the x direction
  tdouble3 *dWyCorr; //Kernel correction in the y direction
  tdouble3 *dWzCorr; //Kernel correction in the z direction
  float *Divr; //Divergence of position
  unsigned *POrder; //Position in Matrix
  unsigned PPEDim;
  //matrix variables for CULA
  std::vector<double> b;
  std::vector<double> a;
  std::vector<int> colInd;
  std::vector<int> rowInd;
  std::vector<double> x;
  
#ifndef _WITHGPU
  template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
  void run_solver(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,std::vector<double> &matrixx,const unsigned ppedim); 
  void solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,std::vector<double> &matrixa,
    std::vector<double> &matrixb,std::vector<double> &matrixx,std::vector<int> &row,std::vector<int> &col,const unsigned ppedim,const unsigned nnz);
#endif
 
  void InverseCorrection(unsigned n, unsigned pinit,tdouble3 *dwxcorr,tdouble3 *dwzcorr)const;
  void InverseCorrection3D(unsigned n, unsigned pinit,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr)const;
  void FindIrelation(unsigned n,unsigned pinit,const tdouble3 *pos,const unsigned *idpc,unsigned *irelation,const word *code)const;

  void solveMatrix();
  void MatrixOrder(unsigned n,unsigned pinit,unsigned *porder,const unsigned *idpc,const unsigned *irelation,word *code,unsigned &ppeDim);
  void MatrixASetup(const unsigned ppedim,unsigned &nnz,std::vector<int> &row)const;
  void RHSandLHSStorage(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
	  const unsigned *dcell,const tdouble3 *pos,const tfloat4 *velrhop,tdouble3 *dwxcorr,tdouble3 *dwycorr,tdouble3 *dwzcorr,std::vector<double> &matrixb,const unsigned *porder,
    const unsigned *idpc,const double dt,const unsigned ppedim,const float *divr,const float freesurface,std::vector<int> &row)const;
  void PopulateMatrixAFluid(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
	  const unsigned *dcell,const tdouble3 *pos,const tfloat4 *velrhop,float *divr,std::vector<double> &matrixInd,std::vector<int> &row,std::vector<int> &col,
    const unsigned *porder,const unsigned *irelation,std::vector<double> &matrixb,const unsigned *idpc,const word *code,const unsigned ppedim,const float freesurface,tfloat3 gravity,const double rhoZero)const;
   void PopulateMatrixABoundary(unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,const unsigned *beginendcell,tint3 cellzero,
	  const unsigned *dcell,const tdouble3 *pos,const tfloat4 *velrhop,float *divr,std::vector<double> &matrixInd,std::vector<int> &row,std::vector<int> &col,
    const unsigned *porder,const unsigned *irelation,std::vector<double> &matrixb,const unsigned *idpc,const word *code,const unsigned ppedim,const float freesurface,tfloat3 gravity,const double rhoZero)const;
	void PressureAssign(unsigned n,unsigned pinit,const tdouble3 *pos,tfloat4 *velrhop,const unsigned *idpc,const unsigned *irelation,const unsigned *porder,
    std::vector<double> &matrixx,const word *code,const unsigned npb,float *divr,tfloat3 gravity)const;
  void FreeSurfaceMark(unsigned n,unsigned pinit,float *divr,std::vector<double> &matrixInd,std::vector<double> &matrixb,std::vector<int> &row,const unsigned *porder,const unsigned *idpc,const word *code,const unsigned ppedim)const;

  
  void Interaction_Shifting(unsigned np,unsigned npb,unsigned npbok
    ,tuint3 ncells,const unsigned *begincell,tuint3 cellmin,const unsigned *dcell
    ,const tdouble3 *pos,tfloat4 *velrhop,const unsigned *idp,const word *code
    ,tfloat3 *shiftpos,float *divr,const float tensileN,const float tensileR)const;

  template<TpFtMode ftmode> void InteractionForcesShifting
  (unsigned n,unsigned pinit,tint4 nc,int hdiv,unsigned cellinitial,float visco
  ,const unsigned *beginendcell,tint3 cellzero,const unsigned *dcell
  ,const tdouble3 *pos,tfloat4 *velrhop,const word *code,const unsigned *idp
  ,TpShifting tshifting,tfloat3 *shiftpos,float *divr,const float tensileN,const float tensileR)const;

  void Shift(double dt);
  int count;
  unsigned PPEDimOld;
public:
  JSphCpu(bool withmpi);
  ~JSphCpu();
};

#endif


