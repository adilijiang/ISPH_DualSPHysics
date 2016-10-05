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

#ifndef _JSphGpu_ker_
#define _JSphGpu_ker_

#include "Types.h"
#include "JSphTimersGpu.h"
#include <cuda_runtime_api.h>
class JLog2;

#define SPHBSIZE 256

typedef struct{
  unsigned nbound;
  float massb,massf;
  float fourh2,h;
  float awen,bwen;
  float cs0,eta2;
  float delta2h;     //delta2h=DeltaSph*H*2
  float scell,dosh,dp;
  float cteb,gamma;
  float rhopzero;    //rhopzero=RhopZero
  float ovrhopzero;  //ovrhopzero=1/RhopZero
  float movlimit;
  unsigned periactive;
  double xperincx,xperincy,xperincz;
  double yperincx,yperincy,yperincz;
  double zperincx,zperincy,zperincz;
  double maprealposminx,maprealposminy,maprealposminz;
  double maprealsizex,maprealsizey,maprealsizez;
  //-Valores que dependen del dominio asignado (puden cambiar).
  //-Values depending on the assigned domain (can change).
  unsigned cellcode;
  double domposminx,domposminy,domposminz;
}StCteInteraction; 

namespace cusph{

inline float3 Float3(const tfloat3& v){ float3 p={v.x,v.y,v.z}; return(p); }
inline float3 Float3(float x,float y,float z){ float3 p={x,y,z}; return(p); }
inline tfloat3 ToTFloat3(const float3& v){ return(TFloat3(v.x,v.y,v.z)); }
inline double3 Double3(const tdouble3& v){ double3 p={v.x,v.y,v.z}; return(p); }

dim3 GetGridSize(unsigned n,unsigned blocksize);
inline unsigned ReduMaxFloatSize(unsigned ndata){ return((ndata/SPHBSIZE+1)+(ndata/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)); }
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data,float* resu);
float ReduMaxFloat_w(unsigned ndata,unsigned inidata,float4* data,float* resu);

void CteInteractionUp(const StCteInteraction *cte);
void InitArray(unsigned n,float3 *v,tfloat3 value);
void Resety(unsigned n,unsigned ini,float3 *v);
void ComputeAceMod(unsigned n,const float3 *ace,float *acemod);
void ComputeAceMod(unsigned n,const word *code,const float3 *ace,float *acemod);

void ComputeVelMod(unsigned n,const float4 *vel,float *velmod);

//# Kernels para preparar calculo de fuerzas con Pos-Simple.
//# Kernels for preparing the force calculation for Pos-Simple.
void PreInteractionSimple(unsigned np,const double2 *posxy,const double *posz
  ,const float4 *velrhop,float4 *pospress,float cteb,float ctegamma);

//# Kernels para calculo de fuerzas.
//# Kernels for the force calculation.
void Interaction_Forces(bool psimple,bool floating,bool usedem,TpSlipCond tslipcond,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwzcorrg
  ,const float *ftomassp,float3 *ace,bool simulate2d);

//# Kernels para calculo de fuerzas DEM
//# for the calculation of the DEM forces
void Interaction_ForcesDem(bool psimple,TpCellMode cellmode,unsigned bsize
  ,unsigned nfloat,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const unsigned *ftridp,const float4 *demdata,float dtforce
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,const word *code,const unsigned *idp,float *viscdt,float3 *ace);

//# Kernels para Delta-SPH
//# Kernels for Delta-SPH
void AddDelta(unsigned n,const float *delta,float *ar);

//# Kernels para Shifting
//# Kernels for Shifting
void RunShifting(unsigned np,unsigned npb,double dt
  ,double shiftcoef,float freesurface,double coeftfs
  ,float4 *velrhop,const float *divr,float3 *shiftpos);

void ComputeStepSymplecticPre(bool floating,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float3 *ace,double dtm,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop);

void ComputeStepSymplecticCor(bool floating,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float3 *ace,double dtm,double dt,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop,tfloat3 gravity);

//# Kernels para ComputeStep (position)
//# Kernels for ComputeStep (position)
void ComputeStepPos(byte periactive,bool floating,unsigned np,unsigned npb,const double2 *movxy,const double *movz,double2 *posxy,double *posz,unsigned *dcell,word *code);
void ComputeStepPos2(byte periactive,bool floating,unsigned np,unsigned npb,const double2 *posxypre,const double *poszpre,const double2 *movxy,const double *movz,double2 *posxy,double *posz,unsigned *dcell,word *code);

//# Kernels para Motion
//# Kernels for Motion
void CalcRidp(bool periactive,unsigned np,unsigned pini,unsigned idini,unsigned idfin,const word *code,const unsigned *idp,unsigned *ridp);
void MoveLinBound(byte periactive,unsigned np,unsigned ini,tdouble3 mvpos,tfloat3 mvvel,const unsigned *ridp,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code);
void MoveMatBound(byte periactive,bool simulate2d,unsigned np,unsigned ini,tmatrix4d m,double dt,const unsigned *ridpmv,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code);

//# Kernels para Floating bodies
//# Kernels for Floating bodies
void FtCalcForces(bool periactive,unsigned ftcount
  ,tfloat3 gravity,const float4 *ftodata,const float *ftomassp,const double3 *ftocenter,const unsigned *ftridp
  ,const double2 *posxy,const double *posz,const float3 *ace
  ,float3 *ftoforces);
void FtUpdate(bool periactive,bool predictor,bool simulate2d,unsigned ftcount
  ,double dt,tfloat3 gravity,const float4 *ftodata,const unsigned *ftridp
  ,const float3 *ftoforces,double3 *ftocenter,float3 *ftovel,float3 *ftoomega
  ,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code);

//# Kernels para Periodic conditions
//# Kernels for periodic conditions
void PeriodicIgnore(unsigned n,word *code);
unsigned PeriodicMakeList(unsigned n,unsigned pini,bool stable,unsigned nmax,tdouble3 mapposmin,tdouble3 mapposmax,tdouble3 perinc,const double2 *posxy,const double *posz,const word *code,unsigned *listp);
void PeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,tuint3 domcells,tdouble3 perinc,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,tsymatrix3f *spstau,double2 *posxypre,double *poszpre,float4 *velrhoppre);

//# Kernels para external forces (JSphVarAcc)
//# Kernels for external forces (JSphVarAcc)
void AddVarAcc(unsigned n,unsigned pini,word codesel
  ,tdouble3 acclin,tdouble3 accang,tdouble3 centre,tdouble3 velang,tdouble3 vellin,bool setgravity
  ,tfloat3 gravity,const word *code,const double2 *posxy,const double *posz,const float4 *velrhop,float3 *ace);

//# Kernels for initial advection
void ComputeRStar(bool floating,unsigned np,unsigned npb,const float4 *velrhoppre,double dtm,word *code,double2 *movxy,double *movz);

//# Kernels for finding a dummy particles corresponding wall particle
void FindIrelation(const unsigned bsbound,unsigned npbok
  ,const double2 *posxy,const double *posz
  ,const word *code,const unsigned *idp,unsigned *irelationg);

//# Kernels for kernel Correction
void KernelCorrection(bool psimple,TpCellMode cellmode
  ,const unsigned bsfluid,const unsigned bsbound,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,double3 *dwxcorrg,double3 *dwzcorrg,const word *codeg);

//# Kernels for particle matrix order
void MatrixOrderFluid(const unsigned bsfluid,unsigned np,unsigned npb, unsigned *porder,const unsigned matrixBound);

void MatrixOrderDummy(TpCellMode cellmode
  ,const unsigned bsbound,unsigned np,unsigned npb,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const word *code,const unsigned *idp,const unsigned *irelationg,unsigned *porder);

//# Kernels for finding the freesurface
void FreeSurfaceFind(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop
  ,const word *code,const unsigned *idp,float *divr);

//# Kernels for marking the freesurface
void FreeSurfaceMark(bool psimple,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,float *divr
  ,double *matrixInd,double *matrixb,unsigned int *row,const unsigned *porder,const word *code,const double pi,const float freesurface);

//# Kernels for Populating matrix B
void PopulateMatrixB(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,double3 *dwxcorrg,double3 *dwzcorrg,double *matrixb
  ,const unsigned *porder,const unsigned *idp,const double dt,const unsigned ppedim,const float *divr,const word *code,const float freesurface);

//# Kernels for matrix storage
void MatrixStorage(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,const word *code,const unsigned *idp,float *divr,const unsigned *porder,unsigned int *row,const float freesurface);

//# Kernels for Populating matrix A
void PopulateMatrixA(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell,tfloat3 gravity,const double2 *posxy
  ,const double *posz,const float4 *pospress,const float4 *velrhop,double *matrixInd,double *matrixb
  ,unsigned int *row,unsigned int *col,const unsigned *porder,const unsigned *idp,const unsigned ppedim
  ,const float *divr,const word *code,const unsigned *irelationg,const float freesurface);

//# Kernels for Assigning Pressure
void PressureAssign(bool psimple,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok
  ,const tfloat3 gravity,const double2 *posxy,const double *posz,const float4 *pospress
 ,float4 *velrhop,double *press,const unsigned *porder,const unsigned *idp,const word *code,const unsigned *irelationg,const float *divr,const float freesurface);

//# Kernels for ArrayInitialisation
void InitArrayPOrder(unsigned n,unsigned *v,unsigned value);
void InitArrayCol(unsigned n,unsigned int *v,int value);

//# Kernels for solving with ViennaCL
void solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,double *matrixa,double *matrixx,double *matrixb,unsigned int *row,unsigned int *col,const unsigned nnz,const unsigned ppedim);
void solveViennaCPU(double *matrixa,double *matrixx,double *matrixb,unsigned int *row,unsigned int *col,const unsigned ppedim,const unsigned nnz);

//Kernels for shifting
void Interaction_Shifting(bool psimple,bool floating,bool usedem,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const float *ftomassp
  ,TpShifting tshifting,float3 *shiftpos,float *divr,const float tensilen,const float tensiler);

void ComputeShift(bool floating,const unsigned bsfluid,unsigned np,unsigned npb
  ,const float3 *shiftpos,word *code,double2 *movxy,double *movz);

}
#endif



