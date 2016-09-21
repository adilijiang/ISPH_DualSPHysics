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

#include "JSphGpu_ker.h"
#include <float.h>
#include "JLog2.h"
#include <math_constants.h>
//#include "JDgKerPrint.h"
//#include "JDgKerPrint_ker.h"

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda.h>

#define VIENNACL_WITH_CUDA
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/forwards.h"
#include "viennacl/linalg/amg.hpp"
#include "viennacl/linalg/prod.hpp"

__constant__ StCteInteraction CTE;

namespace cusph{

//==============================================================================
/// Comprueba error y finaliza ejecucion.
/// Checks mistake and ends execution.
//==============================================================================
#define CheckErrorCuda(text)  __CheckErrorCuda(text,__FILE__,__LINE__)
void __CheckErrorCuda(const char *text,const char *file,const int line){
  cudaError_t err=cudaGetLastError();
  if(cudaSuccess!=err){
    char cad[2048]; 
    sprintf(cad,"%s (CUDA error: %s -> %s:%i).\n",text,cudaGetErrorString(err),file,line); 
    throw std::string(cad);
  }
}

//==============================================================================
/// Devuelve tamaño de gridsize segun parametros.
/// Returns size of gridsize according to parameters.
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize;//-Numero total de bloques a lanzar. //-Total block number to throw.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//==============================================================================
/// Reduccion mediante maximo de valores float en memoria shared para un warp.
/// Reduction using maximum of float values in shared memory for a warp.
//==============================================================================
template <unsigned blockSize> __device__ void KerReduMaxFloatWarp(volatile float* sdat,unsigned tid) {
  if(blockSize>=64)sdat[tid]=max(sdat[tid],sdat[tid+32]);
  if(blockSize>=32)sdat[tid]=max(sdat[tid],sdat[tid+16]);
  if(blockSize>=16)sdat[tid]=max(sdat[tid],sdat[tid+8]);
  if(blockSize>=8)sdat[tid]=max(sdat[tid],sdat[tid+4]);
  if(blockSize>=4)sdat[tid]=max(sdat[tid],sdat[tid+2]);
  if(blockSize>=2)sdat[tid]=max(sdat[tid],sdat[tid+1]);
}

//==============================================================================
/// ES:
/// Acumula la suma de n valores del vector dat[], guardando el resultado al 
/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
/// quedando el resultado final en res[0]).
/// - EN:
/// Accumulates the sum of n values of array dat[], storing the result in 
/// the beginning of res[].(Many positions of res[] are used as blocks, 
/// storing the final result in res[0]).
//==============================================================================
template <unsigned blockSize> __global__ void KerReduMaxFloat(unsigned n,unsigned ini,const float *dat,float *res){
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini]: -FLT_MAX);
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sdat[tid]=max(sdat[tid],sdat[tid+256]);  __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sdat[tid]=max(sdat[tid],sdat[tid+128]);  __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sdat[tid]=max(sdat[tid],sdat[tid+64]);   __syncthreads(); }
  if(tid<32)KerReduMaxFloatWarp<blockSize>(sdat,tid);
  if(tid==0)res[blockIdx.y*gridDim.x + blockIdx.x]=sdat[0];
}

//==============================================================================
/// ES:
/// Devuelve el maximo de un vector, usando resu[] como vector auxiliar. El tamaño
/// de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
/// - EN:
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
//==============================================================================
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data,float* resu){
  unsigned n=ndata,ini=inidata;
  unsigned smemSize=SPHBSIZE*sizeof(float);
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  float *dat=data;
  float *resu1=resu,*resu2=resu+n_blocks;
  float *res=resu1;
  while(n>1){
    KerReduMaxFloat<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
    n=n_blocks; ini=0;
    sgrid=GetGridSize(n,SPHBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    if(n>1){
      dat=res; res=(dat==resu1? resu2: resu1); 
    }
  }
  float resf;
  if(ndata>1)cudaMemcpy(&resf,res,sizeof(float),cudaMemcpyDeviceToHost);
  else cudaMemcpy(&resf,data,sizeof(float),cudaMemcpyDeviceToHost);
  return(resf);
}

//==============================================================================
/// ES:
/// Acumula la suma de n valores del vector dat[].w, guardando el resultado al 
/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
/// quedando el resultado final en res[0]).
/// - EN:
/// Accumulates the sum of n values of array dat[], storing the result in 
/// the beginning of res[].(Many positions of res[] are used as blocks, 
/// storing the final result in res[0]).
//==============================================================================
template <unsigned blockSize> __global__ void KerReduMaxFloat_w(unsigned n,unsigned ini,const float4 *dat,float *res){
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini].w: -FLT_MAX);
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sdat[tid]=max(sdat[tid],sdat[tid+256]);  __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sdat[tid]=max(sdat[tid],sdat[tid+128]);  __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sdat[tid]=max(sdat[tid],sdat[tid+64]);   __syncthreads(); }
  if(tid<32)KerReduMaxFloatWarp<blockSize>(sdat,tid);
  if(tid==0)res[blockIdx.y*gridDim.x + blockIdx.x]=sdat[0];
}

//==============================================================================
/// ES:
/// Devuelve el maximo de la componente w de un vector float4, usando resu[] como 
/// vector auxiliar. El tamaño de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
/// - EN:
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
//==============================================================================
float ReduMaxFloat_w(unsigned ndata,unsigned inidata,float4* data,float* resu){
  unsigned n=ndata,ini=inidata;
  unsigned smemSize=SPHBSIZE*sizeof(float);
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  float *dat=NULL;
  float *resu1=resu,*resu2=resu+n_blocks;
  float *res=resu1;
  while(n>1){
    if(!dat)KerReduMaxFloat_w<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,data,res);
    else KerReduMaxFloat<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
    n=n_blocks; ini=0;
    sgrid=GetGridSize(n,SPHBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    if(n>1){
      dat=res; res=(dat==resu1? resu2: resu1); 
    }
  }
  float resf;
  if(ndata>1)cudaMemcpy(&resf,res,sizeof(float),cudaMemcpyDeviceToHost);
  else{
    float4 resf4;
    cudaMemcpy(&resf4,data,sizeof(float4),cudaMemcpyDeviceToHost);
    resf=resf4.w;
  }
  return(resf);
}

//==============================================================================
/// Graba constantes para la interaccion a la GPU.
/// Stores constants for the GPU interaction.
//==============================================================================
void CteInteractionUp(const StCteInteraction *cte){
  cudaMemcpyToSymbol(CTE,cte,sizeof(StCteInteraction));
}

//------------------------------------------------------------------------------
/// Inicializa array con el valor indicado.
/// Initialises array with the indicated value.
//------------------------------------------------------------------------------
__global__ void KerInitArray(unsigned n,float3 *v,float3 value)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la particula //-NI of the particle
  if(p<n)v[p]=value;
}

//==============================================================================
/// Inicializa array con el valor indicado.
/// Initialises array with the indicated value.
//==============================================================================
void InitArray(unsigned n,float3 *v,tfloat3 value){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerInitArray <<<sgrid,SPHBSIZE>>> (n,v,Float3(value));
  }
}

//------------------------------------------------------------------------------
/// Pone v[].y a cero.
/// Sets v[].y to zero.
//------------------------------------------------------------------------------
__global__ void KerResety(unsigned n,unsigned ini,double3 *v)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n)v[p+ini].y=0;
}

//==============================================================================
// Pone v[].y a cero.
/// Sets v[].y to zero.
//==============================================================================
void Resety(unsigned n,unsigned ini,double3 *v){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE>>> (n,ini,v);
  }
}

//------------------------------------------------------------------------------
/// Calculates module^2 of ace.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const double3 *ace,float *acemod)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const double3 r=ace[p];
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace.
//==============================================================================
void ComputeAceMod(unsigned n,const double3 *ace,float *acemod){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerComputeAceMod <<<sgrid,SPHBSIZE>>> (n,ace,acemod);
  }
}

//------------------------------------------------------------------------------
/// Calculates module^2 of ace, comprobando que la particula sea normal.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const word *code,const double3 *ace,float *acemod)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const double3 r=(CODE_GetSpecialValue(code[p])==CODE_NORMAL? ace[p]: make_double3(0,0,0));
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace, comprobando que la particula sea normal.
//==============================================================================
void ComputeAceMod(unsigned n,const word *code,const double3 *ace,float *acemod){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerComputeAceMod <<<sgrid,SPHBSIZE>>> (n,code,ace,acemod);
  }
}


//##############################################################################
//# Otros kernels...
//# Other kernels...
//##############################################################################
//------------------------------------------------------------------------------
/// Calculates module^2 of vel.
//------------------------------------------------------------------------------
__global__ void KerComputeVelMod(unsigned n,const float4 *vel,float *velmod)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const float4 r=vel[p];
    velmod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of vel.
//==============================================================================
void ComputeVelMod(unsigned n,const float4 *vel,float *velmod){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerComputeVelMod <<<sgrid,SPHBSIZE>>> (n,vel,velmod);
  }
}


//##############################################################################
//# Kernels para preparar calculo de fuerzas con Pos-Simple.
//# Kernels for preparing force computation with Pos-Simple.
//##############################################################################
//------------------------------------------------------------------------------
/// Prepara variables para interaccion pos-simple.
/// Prepare variables for pos-simple interaction.
//------------------------------------------------------------------------------
__global__ void KerPreInteractionSimple(unsigned n,const double2 *posxy,const double *posz
  ,const float4 *velrhop,float4 *pospress,float cteb,float gamma)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula
  if(p<n){
    //Calcular press en simple o doble precision no parece tener ningun efecto positivo significativo,
    //y como para PosDouble si se calcula antes y se lee en la interaccion supondria una perdida de 
    //rendimiento del 6% o 15% (gtx480 o k20c) mejor se calcula en simple siempre.
	//Computes press in single or double precision,although the latter does not have any significant positive effect,
    //and like PosDouble if it is previously calculated and read the interaction can incur losses of
    //performance of 6% or 15% (GTX480 or k20c) so it is best calculated as always simple.
    const float rrhop=velrhop[p].w;
    float press=cteb*(powf(rrhop*CTE.ovrhopzero,gamma)-1.0f);
    double2 rpos=posxy[p];
    pospress[p]=make_float4(float(rpos.x),float(rpos.y),float(posz[p]),press);
  }
}

//==============================================================================
/// Prepara variables para interaccion pos-simple.
/// Prepare variables for pos-simple interaction.
//==============================================================================
void PreInteractionSimple(unsigned np,const double2 *posxy,const double *posz
  ,const float4 *velrhop,float4 *pospress,float cteb,float ctegamma)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    KerPreInteractionSimple <<<sgrid,SPHBSIZE>>> (np,posxy,posz,velrhop,pospress,cteb,ctegamma);
  }
}


//##############################################################################
//# Kernels auxiliares para interaccion.
//# Auxiliary kernels for the interaction.
//##############################################################################
//------------------------------------------------------------------------------
/// Devuelve posicion, vel, rhop y press de particula. USADA
/// Returns position, vel, rhop and press of a particle. USED
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerGetParticleData(unsigned p1
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,float3 &velp1,float &rhopp1,double3 &posdp1,float3 &posp1,float &pressp1)
{
  float4 r=velrhop[p1];
  velp1=make_float3(r.x,r.y,r.z);
  rhopp1=r.w;
  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
  pressp1=(CTE.cteb*(powf(rhopp1*CTE.ovrhopzero,CTE.gamma)-1.0f));
}

//------------------------------------------------------------------------------
/// Devuelve posicion y vel de particula.
/// Returns postion and vel of a particle.
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerGetParticleData(unsigned p1
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,float3 &velp1,double3 &posdp1,float3 &posp1)
{
  float4 r=velrhop[p1];
  velp1=make_float3(r.x,r.y,r.z);
  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
}

//------------------------------------------------------------------------------
/// Devuelve posicion de particula. USADA
/// Returns particle postion. USED
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerGetParticleData(unsigned p1
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,double3 &posdp1,float3 &posp1)
{
  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
}

//------------------------------------------------------------------------------
/// Devuelve drx, dry y drz entre dos particulas. USADA
/// Returns drx, dry and drz between the particles. USED
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerGetParticlesDr(int p2
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const double3 &posdp1,const float3 &posp1
  ,double &drx,double &dry,double &drz,float &pressp2)
{
  double2 posp2=posxy[p2];
  drx=posdp1.x-posp2.x;
  dry=posdp1.y-posp2.y;
  drz=posdp1.z-posz[p2];
  pressp2=0;
}

//------------------------------------------------------------------------------
/// Devuelve drx, dry y drz entre dos particulas.
/// Returns drx, dry and drz between the particles.
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerGetParticlesDr(int p2
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const double3 &posdp1,const float3 &posp1
  ,double &drx,double &dry,double &drz)
{
    double2 posp2=posxy[p2];
    drx=posdp1.x-posp2.x;
    dry=posdp1.y-posp2.y;
    drz=posdp1.z-posz[p2];
}

//------------------------------------------------------------------------------
/// Devuelve limites de celdas para interaccion.
/// Returns cell limits for the interaction.
//------------------------------------------------------------------------------
__device__ void KerGetInteractionCells(unsigned rcell
  ,int hdiv,const uint4 &nc,const int3 &cellzero
  ,int &cxini,int &cxfin,int &yini,int &yfin,int &zini,int &zfin)
{
  //-Obtiene limites de interaccion
  //-Obtains interaction limits.
  const int cx=PC__Cellx(CTE.cellcode,rcell)-cellzero.x;
  const int cy=PC__Celly(CTE.cellcode,rcell)-cellzero.y;
  const int cz=PC__Cellz(CTE.cellcode,rcell)-cellzero.z;
  //-Codigo para hdiv 1 o 2 pero no cero.
  //-Code for hdiv 1 or 2 but not zero.
  cxini=cx-min(cx,hdiv);
  cxfin=cx+min(nc.x-cx-1,hdiv)+1;
  yini=cy-min(cy,hdiv);
  yfin=cy+min(nc.y-cy-1,hdiv)+1;
  zini=cz-min(cz,hdiv);
  zfin=cz+min(nc.z-cz-1,hdiv)+1;
}

//------------------------------------------------------------------------------
/// Devuelve limites de celdas para interaccion.
/// Returns cell limits for the interaction.
//------------------------------------------------------------------------------
__device__ void KerGetInteractionCells(double px,double py,double pz
  ,int hdiv,const uint4 &nc,const int3 &cellzero
  ,int &cxini,int &cxfin,int &yini,int &yfin,int &zini,int &zfin)
{
  //-Obtiene limites de interaccion
  //-Obtains interaction limits.
  const int cx=int((px-CTE.domposminx)/CTE.scell)-cellzero.x;
  const int cy=int((py-CTE.domposminy)/CTE.scell)-cellzero.y;
  const int cz=int((pz-CTE.domposminz)/CTE.scell)-cellzero.z;
  //-Codigo para hdiv 1 o 2 pero no cero.
  //-Code for hdiv 1 or 2 but not zero.
  cxini=cx-min(cx,hdiv);
  cxfin=cx+min(nc.x-cx-1,hdiv)+1;
  yini=cy-min(cy,hdiv);
  yfin=cy+min(nc.y-cy-1,hdiv)+1;
  zini=cz-min(cz,hdiv);
  zfin=cz+min(nc.z-cz-1,hdiv)+1;
}

//------------------------------------------------------------------------------
/// Devuelve valores de kernel: frx, fry y frz. USADA
/// Returns kernel values: frx, fry and frz. USED
//------------------------------------------------------------------------------
__device__ void KerGetKernel(double rr2,double drx,double dry,double drz
  ,double &frx,double &fry,double &frz)
{
  const double rad=sqrt(rr2);
  const double qq=rad/CTE.h;
  //-Wendland kernel.
  const double wqq1=1.f-0.5f*qq;
  const double fac=CTE.bwen*qq*wqq1*wqq1*wqq1/rad;
  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

//------------------------------------------------------------------------------
/// Devuelve valores de kernel: wab.
/// returns kernel values: wab.
//------------------------------------------------------------------------------
__device__ float KerGetKernelWab(float rr2)
{
  const float rad=sqrt(rr2);
  const float qq=rad/CTE.h;
  //-Wendland kernel.
  const float wqq=2.f*qq+1.f;
  const float wqq1=1.f-0.5f*qq;
  const float wqq2=wqq1*wqq1;
  return(CTE.awen*wqq*wqq2*wqq2);
}


//##############################################################################
//# Kernels para calculo de fuerzas (Pos-Double)
//# Kernels for calculating forces (Pos-Double)
//##############################################################################
//------------------------------------------------------------------------------
/// Realiza la interaccion de una particula con un conjunto de ellas. Bound-Fluid/Float
/// Interaction of a particle with a set of particles (Bound-Fluid).
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __device__ void KerInteractionForcesBoundBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const float *ftomassp
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop,const word *code,const unsigned* idp
  ,float massf,double3 posdp1,float3 posp1,float3 velp1,float &arp1,float &visc)
{
    float Sum1x=0.0;
    float Sum1z=0.0;
    float Sum2=0.0;
    for(int p2=pini;p2<pfin;p2++){

    double drx,dry,drz;
    KerGetParticlesDr<psimple>(p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
      const float4 velrhop2=velrhop[p2];
      const float W=KerGetKernelWab(rr2);
      Sum1x+=W*velrhop2.x;
      Sum1z+=W*velrhop2.z;
      Sum2+=W;

      
      //-Obtiene masa de particula p2 en caso de existir floatings.
	  //-Obtains particle mass p2 if there are floating bodies.
     // float ftmassp2;    //-Contiene masa de particula floating o massf si es fluid. //-Contains mass of floating body or massf if fluid.
     // bool compute=true; //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
     /* if(USE_FLOATING){
        const word cod=code[p2];
        bool ftp2=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massf);
        compute=!(USE_DEM && ftp2); //-Se desactiva cuando se usa DEM y es bound-float. //-Deactivated when DEM is used and is bound-float.
      }*/
    }
  }
  if(Sum2){
    velrhop[p1].x=(Sum1x/Sum2)/2.0f;
    velrhop[p1].z=(Sum1z/Sum2)/2.0f;
  }
}

//------------------------------------------------------------------------------
/// Realiza interaccion entre particulas. Bound-Fluid/Float
/// Particle interaction. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __global__ void KerInteractionForcesBound
  (unsigned n,int hdiv,uint4 nc,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const float *ftomassp
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop,const word *code,const unsigned *idp
  ,float *viscdt,float *ar)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
  if(p1<n){
    if(CODE_GetTypeValue(code[p1])==0){
      float visc=0,arp1=0;
      //-Carga datos de particula p1.
	  //-Loads particle p1 data.
      double3 posdp1;
      float3 posp1,velp1;
      KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,posdp1,posp1);

      //-Obtiene limites de interaccion
	  //-Obtains interaction limits
      int cxini,cxfin,yini,yfin,zini,zfin;
      KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Interaccion de Contorno con Fluidas.
	  //-Boundary-Fluid interaction.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z+(nc.w*nc.z+1);//-Le suma Nct+1 que es la primera celda de fluido. //-Adds Nct + 1 which is the first cell fluid.
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin)KerInteractionForcesBoundBox<psimple,ftmode> (p1,pini,pfin,ftomassp,posxy,posz,pospress,velrhop,code,idp,CTE.massf,posdp1,posp1,velp1,arp1,visc);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __device__ void KerInteractionForcesFluidVisc
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco
  ,const float *ftomassp,const float2 *tauff
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const word *code,const unsigned *idp
  ,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,float3 posp1,float3 velp1,float pressp1,float rhopp1
  ,const float2 &taup1_xx_xy,const float2 &taup1_xz_yy,const float2 &taup1_yz_zz
  ,float2 &grap1_xx_xy,float2 &grap1_xz_yy,float2 &grap1_yz_zz
  ,double3 &acep1,float &arp1,float &visc)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<psimple> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
	  
      //-Obtiene masa de particula p2 en caso de existir floatings.
	  //-Obtains mass of particle p2 if any floating bodies exist.
      //bool ftp2;         //-Indica si es floating. //-indicates if it is floating.
      //float ftmassp2;    //-Contiene masa de particula floating o massp2 si es bound o fluid. //-Contains mass of floating body or massf if fluid.
      bool compute=true; //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
      /*if(USE_FLOATING){
        const word cod=code[p2];
        ftp2=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        if(ftp2 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
        if(ftp2 && shift && tshifting==SHIFT_NoBound)shiftposp1.x=FLT_MAX; //-Con floatings anula shifting. //-Cancels shifting with floating bodies
        compute=!(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
      }*/

      const float4 velrhop2=velrhop[p2];
      
      //===== Aceleration ===== 
      if(compute){
		    const double volumep2=massp2/CTE.rhopzero; //Volume of particle j 
		    const double rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
		    const double temp=volumep2*2.0f*visco*rDivW/(rr2+CTE.eta2);
	      const double dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
        acep1.x+=temp*dvx; acep1.y+=temp*dvy; acep1.z+=temp*dvz;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __device__ void KerInteractionForcesFluidPresGrad
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco
  ,const float *ftomassp,const float2 *tauff
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const word *code,const unsigned *idp
  ,double3 *dwxcorrg,double3 *dwzcorrg,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,float3 posp1,float3 velp1,float pressp1,float rhopp1
  ,const float2 &taup1_xx_xy,const float2 &taup1_xz_yy,const float2 &taup1_yz_zz
  ,float2 &grap1_xx_xy,float2 &grap1_xz_yy,float2 &grap1_yz_zz
  ,double3 &acep1,float &arp1,float &visc)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<psimple> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    pressp2=velrhop[p2].w;
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);

      //-Obtiene masa de particula p2 en caso de existir floatings.
	  //-Obtains mass of particle p2 if any floating bodies exist.
      //bool ftp2;         //-Indica si es floating. //-indicates if it is floating.
      //float ftmassp2;    //-Contiene masa de particula floating o massp2 si es bound o fluid. //-Contains mass of floating body or massf if fluid.
      bool compute=true; //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
     /* if(USE_FLOATING){
        const word cod=code[p2];
        ftp2=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        if(ftp2 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
        if(ftp2 && shift && tshifting==SHIFT_NoBound)shiftposp1.x=FLT_MAX; //-Con floatings anula shifting. //-Cancels shifting with floating bodies
        compute=!(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
      }*/

      //const float4 velrhop2=velrhop[p2];

	  
      //===== Aceleration ===== 
      if(compute){
		    const double volumep2=massp2/CTE.rhopzero; //Volume of particle j
        const double temp_x=frx*dwxcorrg[p1].x+frz*dwzcorrg[p1].x; 
			  const double temp_z=frx*dwxcorrg[p1].z+frz*dwzcorrg[p1].z; 
        const double temp=volumep2*(pressp2-pressp1);
        acep1.x+=temp*temp_x; acep1.y+=temp*fry; acep1.z+=temp*temp_z;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
/// EN:
/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __global__ void KerInteractionForcesFluid
  (TpInter tinter,unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,float viscob,float viscof
  ,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const float *ftomassp,const float2 *tauff,float2 *gradvelff
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const word *code,const unsigned *idp
  ,double3 *dwxcorrg,double3 *dwzcorrg,float *viscdt,float *ar,double3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    float visc=0,arp1=0; //deltap1=0;
    double3 acep1=make_double3(0,0,0);

    //-Vars para Shifting.
	//-Variables for Shifting.
    //float3 shiftposp1;
   /* if(shift){
      shiftposp1=make_float3(0,0,0);
      shiftdetectp1=0;
    }*/

    //-Obtiene datos de particula p1 en caso de existir floatings.
	//-Obtains data of particle p1 in case there are floating bodies.
    bool ftp1;       //-Indica si es floating. //-Indicates if it is floating.
    float ftmassp1;  //-Contiene masa de particula floating o 1.0f si es fluid. //-Contains floating particle mass or 1.0f if it is fluid.
    if(USE_FLOATING){
      const word cod=code[p1];
      ftp1=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
      ftmassp1=(ftp1? ftomassp[CODE_GetTypeValue(cod)]: 1.f);
      //if(ftp1 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
      //if(ftp1 && shift)shiftposp1.x=FLT_MAX;  //-Para floatings no se calcula shifting. //-Shifting is not calculated for floating bodies.
    }

    //-Obtiene datos basicos de particula p1.
	//-Obtains basic data of particle p1.
    double3 posdp1;
    float3 posp1,velp1;
    float rhopp1,pressp1;
    KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    pressp1=velrhop[p1].w;
    //-Vars para Laminar+SPS
	//-Variables for Laminar+SPS
    float2 taup1_xx_xy,taup1_xz_yy,taup1_yz_zz;
    /*if(lamsps){
      taup1_xx_xy=tauff[p1*3];
      taup1_xz_yy=tauff[p1*3+1];
      taup1_yz_zz=tauff[p1*3+2];
    }*/
    //-Vars para Laminar+SPS (calculo).
	//-Variables for Laminar+SPS (computation).
    float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
    /*if(lamsps){
      grap1_xx_xy=make_float2(0,0);
      grap1_xz_yy=make_float2(0,0);
      grap1_yz_zz=make_float2(0,0);
    }*/

    //-Obtiene limites de interaccion
	//-Obtains interaction limits
    int cxini,cxfin,yini,yfin,zini,zfin;
    KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

    //-Interaccion con Fluidas.
	//-Interaction with fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin){
		  if(tinter==1)KerInteractionForcesFluidVisc<psimple,ftmode> (false,p1,pini,pfin,viscof,ftomassp,tauff,posxy,posz,pospress,velrhop,code,idp,CTE.massf,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc);
		  else if(tinter==2) KerInteractionForcesFluidPresGrad<psimple,ftmode> (false,p1,pini,pfin,viscof,ftomassp,tauff,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,CTE.massf,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc);
		}
	  }
    }

    if(acep1.x||acep1.y||acep1.z){
      double3 r=ace[p1]; 
      if(tinter==1){ r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z;} 
	    if(tinter==2){ const float rho0=CTE.rhopzero; r.x+=(acep1.x/rho0); r.y+=(acep1.y/rho0); r.z+=(acep1.z/rho0);}
      ace[p1]=r;
      acep1=make_double3(0,0,0);
    }

    //-Interaccion con contorno.
	//-Interaction with boundaries.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin){
		  if(tinter==1)KerInteractionForcesFluidVisc<psimple,ftmode> (true ,p1,pini,pfin,viscob,ftomassp,tauff,posxy,posz,pospress,velrhop,code,idp,CTE.massb,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc);
		  else if(tinter==2) KerInteractionForcesFluidPresGrad<psimple,ftmode> (true ,p1,pini,pfin,viscob,ftomassp,tauff,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,CTE.massb,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc);
		}
      }
    }
    //-Almacena resultados.
	//-Stores results.
    if(acep1.x||acep1.y||acep1.z){
      /*if(tdelta==DELTA_Dynamic&&deltap1!=FLT_MAX)arp1+=deltap1;
      if(tdelta==DELTA_DynamicExt){
        float rdelta=delta[p1];
        delta[p1]=(rdelta==FLT_MAX||deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
      }*/
      //ar[p1]+=arp1;
      double3 r=ace[p1]; 
      if(tinter==1){ r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z;} 
	    if(tinter==2){ const float rho0=CTE.rhopzero; r.x+=(acep1.x/rho0); r.y+=(acep1.y/rho0); r.z+=(acep1.z/rho0);}
      ace[p1]=r;
      /*if(visc>viscdt[p1])viscdt[p1]=visc;
      if(lamsps){
        gradvelff[p1*3]=grap1_xx_xy;
        gradvelff[p1*3+1]=grap1_xz_yy;
        gradvelff[p1*3+2]=grap1_yz_zz;
      }
      if(shift){
        shiftpos[p1]=shiftposp1;
        if(shiftdetect)shiftdetect[p1]=shiftdetectp1;
      }*/
    }
  }
}

//==============================================================================
/// Interaccion para el calculo de fuerzas.
/// Interaction for the force computation.
//==============================================================================
template<bool psimple,TpFtMode ftmode> void Interaction_ForcesT
  (TpCellMode cellmode,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwzcorrg
  ,const float *ftomassp,const tsymatrix3f *tau,tsymatrix3f *gradvel
  ,float *viscdt,float* ar,double3 *ace,bool simulate2d)
{
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);

 if(tinter==1&&npbok){
    dim3 sgridb=GetGridSize(npbok,bsbound);
    KerInteractionForcesBound<psimple,ftmode> <<<sgridb,bsbound>>> (npbok,hdiv,nc,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,idp,viscdt,ar);
  }
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    KerInteractionForcesFluid<psimple,ftmode> <<<sgridf,bsfluid>>> (tinter,npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,(const float2*)tau,(float2*)gradvel,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,viscdt,ar,ace);
  }
}
//==============================================================================
/*template<bool psimple,TpFtMode ftmode,bool lamsps> void Interaction_Forces_t2(TpDeltaSph tdelta,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const unsigned *idp,float3 *dwxcorrg,float3 *dwzcorrg
  ,const float *ftomassp,const tsymatrix3f *tau,tsymatrix3f *gradvel
  ,float *viscdt,float* ar,float3 *ace,float *delta
  ,TpShifting tshifting,float3 *shiftpos,bool simulate2d)
{
  if(tshifting){                const bool shift=true;
    if(tdelta==DELTA_None)      Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_None,shift>       (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
    if(tdelta==DELTA_Dynamic)   Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_Dynamic,shift>    (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
    if(tdelta==DELTA_DynamicExt)Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_DynamicExt,shift> (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
  }
  else{                         const bool shift=false;
    if(tdelta==DELTA_None)      Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_None,shift>       (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
    if(tdelta==DELTA_Dynamic)   Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_Dynamic,shift>    (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
    if(tdelta==DELTA_DynamicExt)Interaction_ForcesT<psimple,ftmode,lamsps,DELTA_DynamicExt,shift> (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
  }
}
//==============================================================================
template<bool psimple,TpFtMode ftmode> void Interaction_Forces_t1(bool lamsps,TpDeltaSph tdelta,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const unsigned *idp,float3 *dwxcorrg,float3 *dwzcorrg
  ,const float *ftomassp,const tsymatrix3f *tau,tsymatrix3f *gradvel
  ,float *viscdt,float* ar,float3 *ace,float *delta
  ,TpShifting tshifting,float3 *shiftpos,bool simulate2d)
{
  if(lamsps)Interaction_Forces_t2<psimple,ftmode,true>  (tdelta,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
  else      Interaction_Forces_t2<psimple,ftmode,false> (tdelta,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,simulate2d);
}*/
//==============================================================================
void Interaction_Forces(bool psimple,bool floating,bool usedem,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwzcorrg
  ,const float *ftomassp,const tsymatrix3f *tau,tsymatrix3f *gradvel
  ,float *viscdt,float *ar,double3 *ace,bool simulate2d)
{
  if(psimple){      const bool psimple=true;
    if(!floating)   Interaction_ForcesT<psimple,FTMODE_None> (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
    else if(!usedem)Interaction_ForcesT<psimple,FTMODE_Sph>  (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
    else            Interaction_ForcesT<psimple,FTMODE_Dem>  (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
  }else{            const bool psimple=false;
    if(!floating)   Interaction_ForcesT<psimple,FTMODE_None> (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
    else if(!usedem)Interaction_ForcesT<psimple,FTMODE_Sph>  (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
    else            Interaction_ForcesT<psimple,FTMODE_Dem>  (cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,dwxcorrg,dwzcorrg,ftomassp,tau,gradvel,viscdt,ar,ace,simulate2d);
  }
}

//##############################################################################
//# Kernels para Ren correction.
//# Kernels for the Ren correction.
//##############################################################################
//------------------------------------------------------------------------------
/// Realiza la interaccion de una particula con un conjunto de ellas. Bound-Fluid/Float
/// Interaction of a particle with a set of particles (Bound-Fluid/Float).
//------------------------------------------------------------------------------
/*template<bool psimple,TpFtMode ftmode> __device__ void KerInteractionRenBoundBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,const word *code,const unsigned* idp
  ,const float *ftomassp,float3 gravity
  ,float massf,double3 posdp1,float3 posp1,float &pkfap1,float &pkfbp1)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<psimple>(p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);

    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel
      const float wab=KerGetKernelWab(rr2);

      //===== Obtiene datos de particula p2 ===== 
	  //===== Obtains particle p2 data =====
      const float rhopp2=velrhop[p2].w;
      if(!psimple)pressp2=(CTE.cteb*(powf(rhopp2*CTE.ovrhopzero,CTE.gamma)-1.0f));
      float ftmassp2;    //-Contiene masa de particula floating o massf si es fluid. //-Contains mass of floating particle or massf if it is fluid.
      if(USE_FLOATING){
        const word cod=code[p2];
        bool ftp2=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massf);
      }

      //-Acumula numerador y denominador para calculo de Pkf.
	  //-Sums numerator and denominator for the Pkf computation.
      const float pkf=((USE_FLOATING? ftmassp2: massf)/rhopp2)*wab;
      pkfap1+=pkf*(pressp2+rhopp2*gravity.z*drz); //<--La gravedad debe aplicarse de forma general... //<--Gravity must apply in general form... !!ASKJOSE!!!
      pkfbp1+=pkf;
    }
  }
}

//------------------------------------------------------------------------------
/// Realiza interaccion entre particulas para Ren correction. Bound-Fluid/Float
/// Interaction between particles for the Ren correction. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __global__ void KerInteractionRenBound
  (unsigned n,int hdiv,uint4 nc,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,const word *code,const unsigned *idp
  ,const float *ftomassp,float3 gravity,float *presskf)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p1<n){
    float pkfap1=0,pkfbp1=0;

    //-Carga datos de particula p1.
	//-Loads particle p1 data.
    double3 posdp1;
    float3 posp1;
    KerGetParticleData<psimple>(p1,posxy,posz,pospress,posdp1,posp1);

    //-Obtiene limites de interaccion
	//-Obtains interaction limits.
    int cxini,cxfin,yini,yfin,zini,zfin;
    KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
    //-Interaccion de Contorno con Fluidas.
	//-Interaction of boundaries with fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+(nc.w*nc.z+1);//-Le suma Nct+1 que es la primera celda de fluido. //-Adds Nct+1 which is the first cell fluid.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionRenBoundBox<psimple,ftmode> (p1,pini,pfin,posxy,posz,pospress,velrhop,code,idp,ftomassp,gravity,CTE.massf,posdp1,posp1,pkfap1,pkfbp1);
      }
    }
    //-Almacena resultados.
	//-Stores results.
    presskf[p1]=(pkfbp1!=0? pkfap1/pkfbp1: 0); //<--Se deberia controlar cuando se aplica en función de las particulas de fluido que vea... //<--It should be controlled when applied according to the fluid particles you see...
  }
}

//==============================================================================
/// Realiza interaccion entre particulas para Ren correction. Bound-Fluid/Float
/// Interaction between particles for the Ren correction. Bound-Fluid/Float
//==============================================================================
void Interaction_Ren(bool psimple,bool floating,TpCellMode cellmode
  ,unsigned npbok,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,const word *code,const unsigned *idp
  ,const float *ftomassp,tfloat3 gravity,float *presskf)
{
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  if(npbok){
    const unsigned bsbound=SPHBSIZE;
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(psimple){      const bool psimple=true;
      if(!floating)   KerInteractionRenBound<psimple,FTMODE_None> <<<sgridb,bsbound>>> (npbok,hdiv,nc,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,code,idp,ftomassp,Float3(gravity),presskf);
      else            KerInteractionRenBound<psimple,FTMODE_Sph>  <<<sgridb,bsbound>>> (npbok,hdiv,nc,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,code,idp,ftomassp,Float3(gravity),presskf);
    }else{            const bool psimple=false;
      if(!floating)   KerInteractionRenBound<psimple,FTMODE_None> <<<sgridb,bsbound>>> (npbok,hdiv,nc,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,code,idp,ftomassp,Float3(gravity),presskf);
      else            KerInteractionRenBound<psimple,FTMODE_Sph>  <<<sgridb,bsbound>>> (npbok,hdiv,nc,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,code,idp,ftomassp,Float3(gravity),presskf);
    }
  }
}

//------------------------------------------------------------------------------
/// Calcula nuevo valor de presion y desidad aplicando Ren correction.
/// Computes new value by applying pressure and Ren density correction.
//------------------------------------------------------------------------------
template<bool psimple> __global__ void KerComputeRenPress(unsigned n
  ,float beta,const float *presskf,float4 *velrhop,float4 *pospress)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    const float pressp=(psimple? pospress[p].w: (CTE.cteb*(powf(velrhop[p].w*CTE.ovrhopzero,CTE.gamma)-1.0f)));
    const float pressc=beta*presskf[p]+(1.f-beta)*pressp;
    if(psimple)pospress[p].w=pressc;
    velrhop[p].w=CTE.rhopzero*pow(pressc/CTE.cteb+1.f,1.f/CTE.gamma);
  }
}

//==============================================================================
/// Calcula nuevo valor de presion y desidad aplicando Ren correction.
/// Computes new value by applying pressure and Ren density correction.
//==============================================================================
void ComputeRenPress(bool psimple,unsigned npbok
  ,float beta,const float *presskf,float4 *velrhop,float4 *pospress)
{
  if(npbok){
    dim3 sgrid=GetGridSize(npbok,SPHBSIZE);
    if(psimple)KerComputeRenPress<true>  <<<sgrid,SPHBSIZE>>> (npbok,beta,presskf,velrhop,pospress);
    else       KerComputeRenPress<false> <<<sgrid,SPHBSIZE>>> (npbok,beta,presskf,velrhop,pospress);
  }
}*/


//##############################################################################
//# Kernels para interaccion DEM.
//# Kernels for DEM interaction
//##########################################.####################################
//------------------------------------------------------------------------------
/// Realiza la interaccion DEM de una particula con un conjunto de ellas. (Float-Float/Bound)
/// DEM interaction of a particle with a set of particles (Float-Float/Bound).
//------------------------------------------------------------------------------
/*template<bool psimple> __device__ void KerInteractionForcesDemBox 
  (bool boundp2,const unsigned &pini,const unsigned &pfin
  ,const float4 *demdata,float dtforce
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const word *code,const unsigned *idp
  ,double3 posdp1,float3 posp1,float3 velp1,word tavp1,float masstotp1,float taup1,float kfricp1,float restitup1
  ,float3 &acep1,float &demdtp1)
{
  for(int p2=pini;p2<pfin;p2++){
    const word codep2=code[p2];
    if(CODE_GetType(codep2)!=CODE_TYPE_FLUID && tavp1!=CODE_GetTypeAndValue(codep2)){
      float drx,dry,drz;
      KerGetParticlesDr<psimple> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz);
      const float rr2=drx*drx+dry*dry+drz*drz;
      const float rad=sqrt(rr2);

      //-Calcula valor maximo de demdt.
	  //-Computes maximum value of demdt.
      float4 demdatap2=demdata[CODE_GetTypeAndValue(codep2)];
      const float nu_mass=(boundp2? masstotp1/2: masstotp1*demdatap2.x/(masstotp1+demdatap2.x)); //-Con boundary toma la propia masa del floating 1. //-With boundary takes the actual mass of floating 1.
      const float kn=4/(3*(taup1+demdatap2.y))*sqrt(CTE.dp/4);  //generalized rigidity - Lemieux 2008
      const float demvisc=float(PI)/(sqrt( kn/nu_mass ))*40.f;
      if(demdtp1<demvisc)demdtp1=demvisc;

      const float over_lap=1.0f*CTE.dp-rad; //-(ri+rj)-|dij|
      if(over_lap>0.0f){ //-Contact
        const float dvx=velp1.x-velrhop[p2].x, dvy=velp1.y-velrhop[p2].y, dvz=velp1.z-velrhop[p2].z; //vji
        const float nx=drx/rad, ny=dry/rad, nz=drz/rad; //normal_ji             
        const float vn=dvx*nx+dvy*ny+dvz*nz; //vji.nji    
        //normal                    
        const float eij=(restitup1+demdatap2.w)/2;
        const float gn=-(2.0f*log(eij)*sqrt(nu_mass*kn))/(sqrt(float(PI)+log(eij)*log(eij))); //generalized damping - Cummins 2010
        //const float gn=0.08f*sqrt(nu_mass*sqrt(CTE.dp/2)/((taup1+demdatap2.y)/2)); //generalized damping - Lemieux 2008
        float rep=kn*pow(over_lap,1.5f);
        float fn=rep-gn*pow(over_lap,0.25f)*vn;                   
        acep1.x+=(fn*nx); acep1.y+=(fn*ny); acep1.z+=(fn*nz); //-Force is applied in the normal between the particles
        //tangencial
        float dvxt=dvx-vn*nx, dvyt=dvy-vn*ny, dvzt=dvz-vn*nz; //Vji_t
        float vt=sqrt(dvxt*dvxt + dvyt*dvyt + dvzt*dvzt);
        float tx=(vt!=0? dvxt/vt: 0), ty=(vt!=0? dvyt/vt: 0), tz=(vt!=0? dvzt/vt: 0); //Tang vel unit vector
        float ft_elast=2*(kn*dtforce-gn)*vt/7;   //Elastic frictional string -->  ft_elast=2*(kn*fdispl-gn*vt)/7; fdispl=dtforce*vt;
        const float kfric_ij=(kfricp1+demdatap2.z)/2;
        float ft=kfric_ij*fn*tanh(8*vt);  //Coulomb
        ft=(ft<ft_elast? ft: ft_elast);   //not above yield criteria, visco-elastic model
        acep1.x+=(ft*tx); acep1.y+=(ft*ty); acep1.z+=(ft*tz);
      }
    }
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
/// - EN:
/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
//------------------------------------------------------------------------------
template<bool psimple> __global__ void KerInteractionForcesDem
  (unsigned nfloat,int hdiv,uint4 nc,unsigned cellfluid
  ,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const unsigned *ftridp,const float4 *demdata,float dtforce
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const word *code,const unsigned *idp
  ,float *viscdt,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<nfloat){
    const unsigned p1=ftridp[p];      //-Nº de particula. //-NI of the particle
    if(p1!=UINT_MAX){
      float demdtp1=0;
      float3 acep1=make_float3(0,0,0);

      //-Obtiene datos basicos de particula p1.
	  //-Obtains basic data of particle p1.
      double3 posdp1;
      float3 posp1,velp1;
      KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,posdp1,posp1);
      const word tavp1=CODE_GetTypeAndValue(code[p1]);
      float4 rdata=demdata[tavp1];
      const float masstotp1=rdata.x;
      const float taup1=rdata.y;
      const float kfricp1=rdata.z;
      const float restitup1=rdata.w;

      //-Obtiene limites de interaccion
	  //-Obtains interaction limits
      int cxini,cxfin,yini,yfin,zini,zfin;
      KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Interaccion con contorno.
	  //-Interaction with boundaries.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z;
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin)KerInteractionForcesDemBox<psimple> (true ,pini,pfin,demdata,dtforce,posxy,posz,pospress,velrhop,code,idp,posdp1,posp1,velp1,tavp1,masstotp1,taup1,kfricp1,restitup1,acep1,demdtp1);
        }
      }

      //-Interaccion con Fluidas.
	  //-Interaction with fluids.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin)KerInteractionForcesDemBox<psimple> (false,pini,pfin,demdata,dtforce,posxy,posz,pospress,velrhop,code,idp,posdp1,posp1,velp1,tavp1,masstotp1,taup1,kfricp1,restitup1,acep1,demdtp1);
        }
      }
      //-Almacena resultados.
	  //-Stores results.
      if(acep1.x || acep1.y || acep1.z || demdtp1){
        float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
        if(viscdt[p1]<demdtp1)viscdt[p1]=demdtp1;
      }
    }
  }
}

//==============================================================================
/// Interaccion para el calculo de fuerzas.
/// Interaction for the force computation.
//==============================================================================
template<bool psimple> void Interaction_ForcesDemT
  (TpCellMode cellmode,unsigned bsize
  ,unsigned nfloat,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const unsigned *ftridp,const float4 *demdata,float dtforce
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,const word *code,const unsigned *idp,float *viscdt,float3 *ace)
{
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(nfloat){
    dim3 sgrid=GetGridSize(nfloat,bsize);
    KerInteractionForcesDem<psimple> <<<sgrid,bsize>>> (nfloat,hdiv,nc,cellfluid,begincell,cellzero,dcell,ftridp,demdata,dtforce,posxy,posz,pospress,velrhop,code,idp,viscdt,ace);
  }
}
//==============================================================================
void Interaction_ForcesDem(bool psimple,TpCellMode cellmode,unsigned bsize
  ,unsigned nfloat,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const unsigned *ftridp,const float4 *demdata,float dtforce
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,const word *code,const unsigned *idp,float *viscdt,float3 *ace)
{
  if(psimple)Interaction_ForcesDemT<true>  (cellmode,bsize,nfloat,ncells,begincell,cellmin,dcell,ftridp,demdata,dtforce,posxy,posz,pospress,velrhop,code,idp,viscdt,ace);
  else       Interaction_ForcesDemT<false> (cellmode,bsize,nfloat,ncells,begincell,cellmin,dcell,ftridp,demdata,dtforce,posxy,posz,pospress,velrhop,code,idp,viscdt,ace);
}*/


/*//##############################################################################
//# Kernels para Laminar+SPS.
//##############################################################################
//------------------------------------------------------------------------------
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//------------------------------------------------------------------------------
__global__ void KerComputeSpsTau(unsigned n,unsigned pini,float smag,float blin
  ,const float4 *velrhop,const float2 *gradvelff,float2 *tauff)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p+pini;
    float2 rr=gradvelff[p1*3];   const float grad_xx=rr.x,grad_xy=rr.y;
           rr=gradvelff[p1*3+1]; const float grad_xz=rr.x,grad_yy=rr.y;
           rr=gradvelff[p1*3+2]; const float grad_yz=rr.x,grad_zz=rr.y;
    const float pow1=grad_xx*grad_xx + grad_yy*grad_yy + grad_zz*grad_zz;
    const float prr= grad_xy*grad_xy + grad_xz*grad_xz + grad_yz*grad_yz + pow1+pow1;
    const float visc_sps=smag*sqrt(prr);
    const float div_u=grad_xx+grad_yy+grad_zz;
    const float sps_k=(2.0f/3.0f)*visc_sps*div_u;
    const float sps_blin=blin*prr;
    const float sumsps=-(sps_k+sps_blin);
    const float twovisc_sps=(visc_sps+visc_sps);
    float one_rho2=1.0f/velrhop[p1].w;
    //-Calcula nuevos valores de tau[].
	//-Computes new values of tau[].
    const float tau_xx=one_rho2*(twovisc_sps*grad_xx +sumsps);
    const float tau_xy=one_rho2*(visc_sps   *grad_xy);
    tauff[p1*3]=make_float2(tau_xx,tau_xy);
    const float tau_xz=one_rho2*(visc_sps   *grad_xz);
    const float tau_yy=one_rho2*(twovisc_sps*grad_yy +sumsps);
    tauff[p1*3+1]=make_float2(tau_xz,tau_yy);
    const float tau_yz=one_rho2*(visc_sps   *grad_yz);
    const float tau_zz=one_rho2*(twovisc_sps*grad_zz +sumsps);
    tauff[p1*3+2]=make_float2(tau_yz,tau_zz);
  }
}

//==============================================================================
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//==============================================================================
void ComputeSpsTau(unsigned np,unsigned npb,float smag,float blin
  ,const float4 *velrhop,const tsymatrix3f *gradvelg,tsymatrix3f *tau)
{
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,SPHBSIZE);
    KerComputeSpsTau <<<sgridf,SPHBSIZE>>> (npf,npb,smag,blin,velrhop,(const float2*)gradvelg,(float2*)tau);
  }
}*/


//##############################################################################
//# Kernels para Delta-SPH.
//# Kernels for Delta-SPH.
//##############################################################################
//------------------------------------------------------------------------------
/// Añade valor de delta[] a ar[] siempre que no sea FLT_MAX.
/// Adds value of delta[] to ar[] provided it is not FLT_MAX.
//------------------------------------------------------------------------------
__global__ void KerAddDelta(unsigned n,const float *delta,float *ar)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    float rdelta=delta[p];
    if(rdelta!=FLT_MAX)ar[p]+=rdelta;
  }
}

//==============================================================================
/// Añade valor de delta[] a ar[] siempre que no sea FLT_MAX.
/// Adds value of delta[] to ar[] provided it is not FLT_MAX.
//==============================================================================
void AddDelta(unsigned n,const float *delta,float *ar){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerAddDelta <<<sgrid,SPHBSIZE>>> (n,delta,ar);
  }
}


//##############################################################################
//# Kernels para Shifting.
//# Shifting kernels.
//##############################################################################
//------------------------------------------------------------------------------
/// Calcula Shifting final para posicion de particulas.
/// Computes final shifting for the particle position.
//------------------------------------------------------------------------------
__global__ void KerRunShifting(unsigned n,unsigned pini,double dt
  ,float shiftcoef,double freesurface,double coeftfs
  ,float4 *velrhop,const double *divr,float3 *shiftpos,const double ShiftOffset)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    const unsigned p1=p+pini;
    //const float4 rvel=velrhop[p1];
    float3 rshiftpos=shiftpos[p1];
    //const double vx=double(rvel.x);
    //const double vy=double(rvel.y);
    //const double vz=double(rvel.z);
    double umagn=-shiftcoef*double(CTE.h)*double(CTE.h);//double(shiftcoef)*double(CTE.h)*sqrt(vx*vx+vy*vy+vz*vz)*dt;

    if(divr[p1]<freesurface){
      double NormX=-rshiftpos.x;
      double NormZ=-rshiftpos.z;
      double temp=NormX*NormX+NormZ*NormZ;
      temp=sqrt(temp);
      NormX=NormX/temp;
      NormZ=NormZ/temp;
      double TangX=-NormZ;
      double TangZ=NormX;
      temp=TangX*rshiftpos.x+TangZ*rshiftpos.z;
      rshiftpos.x=temp*TangX;
      rshiftpos.z=temp*TangZ;
      /*const float rdetect=shiftdetect[p1];
      if(rdetect<freesurface)umagn=0;
      else umagn*=(double(rdetect)-freesurface)/coeftfs;*/
    }

    if(divr[p1]>=freesurface && divr[p1]<=freesurface+ShiftOffset){
      double NormX=-rshiftpos.x;
      double NormZ=-rshiftpos.z;
      double temp=NormX*NormX+NormZ*NormZ;
      temp=sqrt(temp);
      NormX=NormX/temp;
      NormZ=NormZ/temp;
      double TangX=-NormZ;
      double TangZ=NormX;
      double temp_s=TangX*rshiftpos.x+TangZ*rshiftpos.z;
      double temp_n=NormX*rshiftpos.x+NormZ*rshiftpos.z;
      double FactorShift=0.5*(1-cos(PI*(divr[p1]-freesurface)/0.2));
      rshiftpos.x=temp_s*TangX+temp_n*NormX*FactorShift;
      rshiftpos.z=temp_s*TangZ+temp_n*NormZ*FactorShift;
      /*const float rdetect=shiftdetect[p1];
      if(rdetect<freesurface)umagn=0;
      else umagn*=(double(rdetect)-freesurface)/coeftfs;*/
    }
        
    rshiftpos.x=float(double(rshiftpos.x)*umagn);
    rshiftpos.y=float(double(rshiftpos.y)*umagn);
    rshiftpos.z=float(double(rshiftpos.z)*umagn);
    shiftpos[p1]=rshiftpos;
  }
}

//==============================================================================
/// Calcula Shifting final para posicion de particulas.
/// Computes final shifting for the particle position.
//==============================================================================
void RunShifting(unsigned np,unsigned npb,double dt
  ,double shiftcoef,double freesurface,double coeftfs
  ,float4 *velrhop,const double *divr,float3 *shiftpos)
{
  const unsigned npf=np-npb;
  const double ShiftOffset=0.2;
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerRunShifting <<<sgrid,SPHBSIZE>>> (npf,npb,dt,shiftcoef,freesurface,coeftfs,velrhop,divr,shiftpos,ShiftOffset);
  }
}


//##############################################################################
//# Kernels para ComputeStep (vel & rhop)
//# Kernels for ComputeStep (vel & rhop)
//##############################################################################
//------------------------------------------------------------------------------
/// ES:
/// Calcula nuevos valores de  Pos, Check, Vel y Rhop (usando Verlet).
/// El valor de Vel para bound siempre se pone a cero.
/// - EN:
/// Computes new values for Pos, Check, Vel and Ros (using Verlet).
/// The value of Vel always set to be reset.
//------------------------------------------------------------------------------
template<bool floating,bool shift> __global__ void KerComputeStepVerlet
  (unsigned n,unsigned npb,float rhopoutmin,float rhopoutmax
  ,const float4 *velrhop1,const float4 *velrhop2
  ,const float *ar,const float3 *ace,const float3 *shiftpos
  ,double dt,double dt205,double dt2
  ,double2 *movxy,double *movz,word *code,float4 *velrhopnew)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    if(p<npb){//-Particulas: Fixed & Moving //-Particles: Fixed & Moving
      float rrhop=float(double(velrhop2[p].w)+dt2*ar[p]);
      rrhop=(rrhop<CTE.rhopzero? CTE.rhopzero: rrhop); //-Evita q las boundary absorvan a las fluidas. //-To prevent absorption of fluid particles by boundaries.
      velrhopnew[p]=make_float4(0,0,0,rrhop);
    }
    else{ //-Particulas: Floating & Fluid //-Particles: Floating & Fluid
      //-Actualiza densidad.
	  //-Updates density.
      float4 rvelrhop2=velrhop2[p];
      rvelrhop2.w=float(double(rvelrhop2.w)+dt2*ar[p]);
      float4 rvel1=velrhop1[p];
      if(!floating || CODE_GetType(code[p])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
        //-Comprueba limites de rhop.
		//-Checks rhop limits.
        if(rvelrhop2.w<rhopoutmin||rvelrhop2.w>rhopoutmax){//-Solo marca como excluidas las normales (no periodicas). //-Only brands as excluded normal particles (not periodic)
          const word rcode=code[p];
          if(CODE_GetSpecialValue(rcode)==CODE_NORMAL)code[p]=CODE_SetOutRhop(rcode);
        }
        //-Calcula y graba desplazamiento de posicion.
		//-Comutes and stores position displacement.
        const float3 race=ace[p];
        double dx=double(rvel1.x)*dt + double(race.x)*dt205;
        double dy=double(rvel1.y)*dt + double(race.y)*dt205;
        double dz=double(rvel1.z)*dt + double(race.z)*dt205;
        if(shift){
          const float3 rshiftpos=shiftpos[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
        //-Actualiza velocidad.
		//-Updates velocity.
        rvelrhop2.x=float(double(rvelrhop2.x)+double(race.x)*dt2);
        rvelrhop2.y=float(double(rvelrhop2.y)+double(race.y)*dt2);
        rvelrhop2.z=float(double(rvelrhop2.z)+double(race.z)*dt2);
        velrhopnew[p]=rvelrhop2;
      }
      else{//-Particulas: Floating //-Particles: Floating.
        rvel1.w=(rvelrhop2.w<CTE.rhopzero? CTE.rhopzero: rvelrhop2.w); //-Evita q las floating absorvan a las fluidas. //-To prevent absorption of fluid particles by boundaries.
        velrhopnew[p]=rvel1;
      }
    }
  }
}

//==============================================================================
/// Actualizacion de particulas segun fuerzas y dt usando Verlet.
/// Updates particles according to forces and dt using Verlet. 
//==============================================================================
void ComputeStepVerlet(bool floating,bool shift,unsigned np,unsigned npb
  ,const float4 *velrhop1,const float4 *velrhop2
  ,const float *ar,const float3 *ace,const float3 *shiftpos
  ,double dt,double dt2,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhopnew)
{
  double dt205=(0.5*dt*dt);
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    if(shift){    const bool shift=true;
      if(floating)KerComputeStepVerlet<true,shift>  <<<sgrid,SPHBSIZE>>> (np,npb,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftpos,dt,dt205,dt2,movxy,movz,code,velrhopnew);
      else        KerComputeStepVerlet<false,shift> <<<sgrid,SPHBSIZE>>> (np,npb,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftpos,dt,dt205,dt2,movxy,movz,code,velrhopnew);
    }else{        const bool shift=false;
      if(floating)KerComputeStepVerlet<true,shift>  <<<sgrid,SPHBSIZE>>> (np,npb,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftpos,dt,dt205,dt2,movxy,movz,code,velrhopnew);
      else        KerComputeStepVerlet<false,shift> <<<sgrid,SPHBSIZE>>> (np,npb,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftpos,dt,dt205,dt2,movxy,movz,code,velrhopnew);
    }
  }
}

//------------------------------------------------------------------------------
/// Calcula los nuevos valores de Pos, Vel y Rhop (usando para Symplectic-Predictor)
/// Computes new values for Pos, Check, Vel and Ros (used with Symplectic-Predictor).
//------------------------------------------------------------------------------
template<bool floating> __global__ void KerComputeStepSymplecticPre
  (unsigned n,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const double3 *ace,double dtm,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    if(p<npb){//-Particulas: Fixed & Moving //-Particles: Fixed & Moving
      float4 rvelrhop=velrhoppre[p];
       rvelrhop.w=velrhop[p].w;
      /*rvelrhop.w=float(double(rvelrhop.w)+dtm*ar[p]);
      rvelrhop.w=(rvelrhop.w<CTE.rhopzero? CTE.rhopzero: rvelrhop.w);*/ //-Evita q las boundary absorvan a las fluidas.  //-To prevent absorption of fluid particles by boundaries.
      velrhop[p]=rvelrhop;
    }
    else{ //-Particulas: Floating & Fluid //-Particles: Floating & Fluid
      //-Actualiza densidad.
	  //-Updates density.
      float4 rvelrhop=velrhoppre[p];
      if(!floating || CODE_GetType(code[p])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
        //-Comprueba limites de rhop.
		//-Checks rhop limits.
        /*if(rvelrhop.w<rhopoutmin||rvelrhop.w>rhopoutmax){//-Solo marca como excluidas las normales (no periodicas).  //-Only brands as excluded normal particles (not periodic)
          const word rcode=code[p];
          if(CODE_GetSpecialValue(rcode)==CODE_NORMAL)code[p]=CODE_SetOutRhop(rcode);
        }*/
        //-Calcula y graba desplazamiento de posicion.
		//-Computes and stores position displacement.
        double dx=double(rvelrhop.x)*dtm;
        double dy=double(rvelrhop.y)*dtm;
        double dz=double(rvelrhop.z)*dtm;
        /*if(shift){
          const float3 rshiftpos=shiftpos[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }*/
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
        //-Actualiza velocidad.
		//-Updates velocity.
        const double3 race=ace[p];
        rvelrhop.x=float(double(rvelrhop.x)+double(race.x)*dtm);
        rvelrhop.y=float(double(rvelrhop.y)+double(race.y)*dtm);
        rvelrhop.z=float(double(rvelrhop.z)+double(race.z)*dtm);
      }
      /*else{//-Particulas: Floating /-Particles: Floating
        rvelrhop.w=(rvelrhop.w<CTE.rhopzero? CTE.rhopzero: rvelrhop.w); //-Evita q las floating absorvan a las fluidas.  //-To prevent absorption of fluid particles by boundaries.
      }*/
      //-Graba nueva velocidad y densidad.
	  //-Stores new velocity and density.
      velrhop[p]=rvelrhop;
    }
  }
}

//==============================================================================
/// Actualizacion de particulas usando Symplectic-Predictor.
/// Updates particles using Symplectic-Predictor.
//==============================================================================   
void ComputeStepSymplecticPre(bool floating,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const double3 *ace,double dtm,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);

    if(floating)KerComputeStepSymplecticPre<true>  <<<sgrid,SPHBSIZE>>> (np,npb,velrhoppre,ar,ace,dtm,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop);
    else        KerComputeStepSymplecticPre<false> <<<sgrid,SPHBSIZE>>> (np,npb,velrhoppre,ar,ace,dtm,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop);
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Calcula los nuevos valores de Pos, Vel y Rhop (usandopara Symplectic-Corrector)
/// Pone vel de contorno a cero.
/// - EN:
/// Computes new values for Pos, Check, Vel and Ros (using Verlet).
/// The value of Vel always set to be reset.
//------------------------------------------------------------------------------
template<bool floating> __global__ void KerComputeStepSymplecticCor
  (unsigned n,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const double3 *ace,double dtm,double dt,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop,tfloat3 gravity)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    if(p<npb){//-Particulas: Fixed & Moving //-Particles: Fixed & Moving
      /*double epsilon_rdot=(-double(ar[p])/double(velrhop[p].w))*dt;
      float rrhop=float(double(velrhoppre[p].w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
      rrhop=(rrhop<CTE.rhopzero? CTE.rhopzero: rrhop); //-Evita q las boundary absorvan a las fluidas. //-To prevent absorption of fluid particles by boundaries.
      velrhop[p]=make_float4(0,0,0,rrhop);*/
      //velrhop[p].w=float(b[porder[p]]);
    }
    else{ //-Particulas: Floating & Fluid //-Particles: Floating & Fluid
      //-Actualiza densidad.
		//-Updates density.
      //double epsilon_rdot=(-double(ar[p])/double(velrhop[p].w))*dt;
      float4 rvelrhop=velrhop[p];
      //rvelrhop.w=float(double(rvelrhop.w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
      if(!floating || CODE_GetType(code[p])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
        float4 rvelp=velrhoppre[p];
        //-Actualiza velocidad.
		//-Updates velocity.
        double3 race=ace[p];
        rvelrhop.x-=float(double(race.x-gravity.x)*dt);
        rvelrhop.y-=float(double(race.y-gravity.y)*dt);
        rvelrhop.z-=float(double(race.z-gravity.z)*dt);
        //-Comprueba limites de rhop.
		/*//-Checks rhop limits.
        if(rvelrhop.w<rhopoutmin||rvelrhop.w>rhopoutmax){//-Solo marca como excluidas las normales (no periodicas). //-Only brands as excluded normal particles (not periodic)
          const word rcode=code[p];
          if(CODE_GetSpecialValue(rcode)==CODE_NORMAL)code[p]=CODE_SetOutRhop(rcode);
        }*/
        //-Calcula y graba desplazamiento de posicion.
		//-Computes and stores position displacement.
        double dx=(double(rvelp.x)+double(rvelrhop.x))*dtm;
        double dy=(double(rvelp.y)+double(rvelrhop.y))*dtm;
        double dz=(double(rvelp.z)+double(rvelrhop.z))*dtm;
        /*if(shift){
          const float3 rshiftpos=shiftpos[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }*/
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
      }
      /*else{//-Particulas: Floating //-Particles: Floating
        rvelrhop.w=(rvelrhop.w<CTE.rhopzero? CTE.rhopzero: rvelrhop.w); //-Evita q las floating absorvan a las fluidas. //-To prevent absorption of fluid particles by boundaries.
      }*/
      //-Graba nueva velocidad y densidad.
	  //-Stores new velocity and density.
      //rvelrhop.w=float(b[porder[p]]);
      velrhop[p]=rvelrhop;
    }
  }
}

//==============================================================================
// Actualizacion de particulas usando Symplectic-Corrector.
/// Updates particles using Symplectic-Corrector.
//==============================================================================   
void ComputeStepSymplecticCor(bool floating,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const double3 *ace,double dtm,double dt,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,float4 *velrhop,tfloat3 gravity)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);

    if(floating)KerComputeStepSymplecticCor<true>  <<<sgrid,SPHBSIZE>>> (np,npb,velrhoppre,ar,ace,dtm,dt,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop,gravity);
    else        KerComputeStepSymplecticCor<false> <<<sgrid,SPHBSIZE>>> (np,npb,velrhoppre,ar,ace,dtm,dt,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop,gravity);
  }
}


//##############################################################################
//# Kernels para ComputeStep (position)
//# Kernels for ComputeStep (position)
//##############################################################################
//------------------------------------------------------------------------------
/// ES:
/// Actualiza pos, dcell y code a partir del desplazamiento indicado.
/// Code puede ser CODE_OUTRHOP pq en ComputeStepVerlet/Symplectic se evalua esto 
/// y se ejecuta antes que ComputeStepPos.
/// Comprueba los limites en funcion de maprealposmin y maprealsize esto es valido
/// para single-gpu pq domrealpos y maprealpos son iguales. Para multi-gpu seria 
/// necesario marcar las particulas q salgan del dominio sin salir del mapa.
/// - EN:
/// Updates pos, dcell and code from the indicated displacement.
/// The code may be CODE_OUTRHOP because in ComputeStepVerlet / Symplectic this is evaluated
/// and is executed before ComputeStepPos.
/// Checks limits depending on maprealposmin and maprealsize, this is valid 
/// for single-GPU because maprealpos and domrealpos are equal. For multi-gpu it is
/// important to mark particles that leave the domain without leaving the map.
//------------------------------------------------------------------------------
template<bool periactive> __device__ void KerUpdatePos
  (double2 rxy,double rz,double movx,double movy,double movz
  ,bool outrhop,unsigned p,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  //-Comprueba validez del desplazamiento.
  //-Checks validity of displacement.
  bool outmove=(fmaxf(fabsf(float(movx)),fmaxf(fabsf(float(movy)),fabsf(float(movz))))>CTE.movlimit);
  //-Aplica desplazamiento.
  //-Applies diplacement.
  double3 rpos=make_double3(rxy.x,rxy.y,rz);
  rpos.x+=movx; rpos.y+=movy; rpos.z+=movz;
  //-Comprueba limites del dominio reales.
  //-Checks limits of real domain.
  double dx=rpos.x-CTE.maprealposminx;
  double dy=rpos.y-CTE.maprealposminy;
  double dz=rpos.z-CTE.maprealposminz;
  bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=CTE.maprealsizex || dy>=CTE.maprealsizey || dz>=CTE.maprealsizez);
  if(periactive && out){
    bool xperi=(CTE.periactive&1),yperi=(CTE.periactive&2),zperi=(CTE.periactive&4);
    if(xperi){
      if(dx<0)                { dx-=CTE.xperincx; dy-=CTE.xperincy; dz-=CTE.xperincz; }
      if(dx>=CTE.maprealsizex){ dx+=CTE.xperincx; dy+=CTE.xperincy; dz+=CTE.xperincz; }
    }
    if(yperi){
      if(dy<0)                { dx-=CTE.yperincx; dy-=CTE.yperincy; dz-=CTE.yperincz; }
      if(dy>=CTE.maprealsizey){ dx+=CTE.yperincx; dy+=CTE.yperincy; dz+=CTE.yperincz; }
    }
    if(zperi){
      if(dz<0)                { dx-=CTE.zperincx; dy-=CTE.zperincy; dz-=CTE.zperincz; }
      if(dz>=CTE.maprealsizez){ dx+=CTE.zperincx; dy+=CTE.zperincy; dz+=CTE.zperincz; }
    }
    bool outx=!xperi && (dx<0 || dx>=CTE.maprealsizex);
    bool outy=!yperi && (dy<0 || dy>=CTE.maprealsizey);
    bool outz=!zperi && (dz<0 || dz>=CTE.maprealsizez);
    out=(outx||outy||outz);
    rpos=make_double3(dx+CTE.maprealposminx,dy+CTE.maprealposminy,dz+CTE.maprealposminz);
  }
  //-Guarda posicion actualizada.
  //-Stoes updated position.
  posxy[p]=make_double2(rpos.x,rpos.y);
  posz[p]=rpos.z;
  //-Guarda celda y check.
  //-Stores cell and checks.
  if(outrhop || outmove || out){//-Particle out. Solo las particulas normales (no periodicas) se pueden marcar como excluidas. //-Particle out. Only brands as excluded normal particles (not periodic).
    word rcode=code[p];
    if(outrhop)rcode=CODE_SetOutRhop(rcode);
    else if(out)rcode=CODE_SetOutPos(rcode);
    else rcode=CODE_SetOutMove(rcode);
    code[p]=rcode;
    dcell[p]=0xFFFFFFFF;
  }
  else{//-Particle in
    if(periactive){
      dx=rpos.x-CTE.domposminx;
      dy=rpos.y-CTE.domposminy;
      dz=rpos.z-CTE.domposminz;
    }
    unsigned cx=unsigned(dx/CTE.scell),cy=unsigned(dy/CTE.scell),cz=unsigned(dz/CTE.scell);
    dcell[p]=PC__Cell(CTE.cellcode,cx,cy,cz);
  }
}

//------------------------------------------------------------------------------
/// Devuelve la posicion corregida tras aplicar condiciones periodicas.
/// Returns the corrected position after applying periodic conditions.
//------------------------------------------------------------------------------
__device__ double3 KerUpdatePeriodicPos(double3 ps)
{
  double dx=ps.x-CTE.maprealposminx;
  double dy=ps.y-CTE.maprealposminy;
  double dz=ps.z-CTE.maprealposminz;
  const bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=CTE.maprealsizex || dy>=CTE.maprealsizey || dz>=CTE.maprealsizez);
  //-Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
  //-Adjusts position according to periodic conditions and rechecks domain limits.
  if(out){
    bool xperi=(CTE.periactive&1),yperi=(CTE.periactive&2),zperi=(CTE.periactive&4);
    if(xperi){
      if(dx<0)                { dx-=CTE.xperincx; dy-=CTE.xperincy; dz-=CTE.xperincz; }
      if(dx>=CTE.maprealsizex){ dx+=CTE.xperincx; dy+=CTE.xperincy; dz+=CTE.xperincz; }
    }
    if(yperi){
      if(dy<0)                { dx-=CTE.yperincx; dy-=CTE.yperincy; dz-=CTE.yperincz; }
      if(dy>=CTE.maprealsizey){ dx+=CTE.yperincx; dy+=CTE.yperincy; dz+=CTE.yperincz; }
    }
    if(zperi){
      if(dz<0)                { dx-=CTE.zperincx; dy-=CTE.zperincy; dz-=CTE.zperincz; }
      if(dz>=CTE.maprealsizez){ dx+=CTE.zperincx; dy+=CTE.zperincy; dz+=CTE.zperincz; }
    }
    ps=make_double3(dx+CTE.maprealposminx,dy+CTE.maprealposminy,dz+CTE.maprealposminz);
  }
  return(ps);
}

//------------------------------------------------------------------------------
/// Actualizacion de posicion de particulas segun desplazamiento.
/// Updates particle position according to displacement.
//------------------------------------------------------------------------------
template<bool periactive,bool floating> __global__ void KerComputeStepPos(unsigned n,unsigned pini
  ,const double2 *movxy,const double *movz
  ,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  unsigned pt=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula
  if(pt<n){
    unsigned p=pt+pini;
    const word rcode=code[p];
    const bool outrhop=(CODE_GetSpecialValue(rcode)==CODE_OUTRHOP);
    const bool fluid=(!floating || CODE_GetType(rcode)==CODE_TYPE_FLUID);
    const bool normal=(!periactive || outrhop || CODE_GetSpecialValue(rcode)==CODE_NORMAL);
    if(normal && fluid){//-No se aplica a particulas periodicas o floating. //-Does not apply to periodic or floating particles.
      const double2 rmovxy=movxy[p];
      KerUpdatePos<periactive>(posxy[p],posz[p],rmovxy.x,rmovxy.y,movz[p],outrhop,p,posxy,posz,dcell,code);
    }
    //-En caso de floating mantiene la posicion original.
	//-In case of floating maintains the original position.
  }
}

//==============================================================================
/// Actualizacion de posicion de particulas segun desplazamiento.
/// Updates particle position according to displacement.
//==============================================================================
void ComputeStepPos(byte periactive,bool floating,unsigned np,unsigned npb
  ,const double2 *movxy,const double *movz
  ,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  const unsigned pini=npb;
  const unsigned npf=np-pini;
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    if(periactive){ const bool peri=true;
      if(floating)KerComputeStepPos<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
    }
    else{ const bool peri=false;
      if(floating)KerComputeStepPos<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
    }
  }
}

//------------------------------------------------------------------------------
/// Actualizacion de posicion de particulas segun desplazamiento.
/// Updates particle position according to displacement.
//------------------------------------------------------------------------------
template<bool periactive,bool floating> __global__ void KerComputeStepPos2(unsigned n,unsigned pini
  ,const double2 *posxypre,const double *poszpre,const double2 *movxy,const double *movz
  ,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  unsigned pt=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(pt<n){
    unsigned p=pt+pini;
    const word rcode=code[p];
    const bool outrhop=(CODE_GetSpecialValue(rcode)==CODE_OUTRHOP);
    const bool fluid=(!floating || CODE_GetType(rcode)==CODE_TYPE_FLUID);
    const bool normal=(!periactive || outrhop || CODE_GetSpecialValue(rcode)==CODE_NORMAL);
    if(normal){//-No se aplica a particulas periodicas //-Does not apply to periodic particles.
      if(fluid){//-Solo se aplica desplazamiento al fluido. //-Only applied for fluid displacement.
        const double2 rmovxy=movxy[p];
        KerUpdatePos<periactive>(posxypre[p],poszpre[p],rmovxy.x,rmovxy.y,movz[p],outrhop,p,posxy,posz,dcell,code);
      }
      else{
        posxy[p]=posxypre[p];
        posz[p]=poszpre[p];
      }
    }
  }
}

//==============================================================================
/// Actualizacion de posicion de particulas segun desplazamiento.
/// Updates particle position according to displacement.
//==============================================================================
void ComputeStepPos2(byte periactive,bool floating,unsigned np,unsigned npb
  ,const double2 *posxypre,const double *poszpre,const double2 *movxy,const double *movz
  ,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  const unsigned pini=npb;
  const unsigned npf=np-pini;
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    if(periactive){ const bool peri=true;
      if(floating)KerComputeStepPos2<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos2<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
    }
    else{ const bool peri=false;
      if(floating)KerComputeStepPos2<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos2<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
    }
  }
}


//##############################################################################
//# Kernels para Motion
//# Kernels for motion.
//##############################################################################
//------------------------------------------------------------------------------
/// Calcula para un rango de particulas calcula su posicion segun idp[].
/// Computes for a range of particles, their position according to idp[].
//------------------------------------------------------------------------------
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const word *code,const unsigned *idp,unsigned *ridp)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    p+=ini;
    unsigned id=idp[p];
    if(idini<=id && id<idfin){
      if(CODE_GetSpecialValue(code[p])==CODE_NORMAL)ridp[id-idini]=p;
    }
  }
}
//------------------------------------------------------------------------------
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    p+=ini;
    const unsigned id=idp[p];
    if(idini<=id && id<idfin)ridp[id-idini]=p;
  }
}

//==============================================================================
/// ES:
/// Calcula posicion de particulas segun idp[]. Cuando no la encuentra es UINT_MAX.
/// Cuando periactive es False sumpone que no hay particulas duplicadas (periodicas)
/// y todas son CODE_NORMAL.
/// - EN:
/// Calculate particle position according to idp[]. When it does not find UINT_MAX.
/// When periactive is false it means there are no duplicate particles (periodic)
/// and all are CODE_NORMAL.
//==============================================================================
void CalcRidp(bool periactive,unsigned np,unsigned pini,unsigned idini,unsigned idfin,const word *code,const unsigned *idp,unsigned *ridp){
  //-Asigna valores UINT_MAX
  //-Assigns values UINT_MAX
  const unsigned nsel=idfin-idini;
  cudaMemset(ridp,255,sizeof(unsigned)*nsel); 
  //-Calcula posicion segun id.
  //-Computes position according to id.
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    if(periactive)KerCalcRidp <<<sgrid,SPHBSIZE>>> (np,pini,idini,idfin,code,idp,ridp);
    else          KerCalcRidp <<<sgrid,SPHBSIZE>>> (np,pini,idini,idfin,idp,ridp);
  }
}

//------------------------------------------------------------------------------
/// Aplica un movimiento lineal a un conjunto de particulas.
/// Applies a linear movement to a set of particles.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerMoveLinBound(unsigned n,unsigned ini,double3 mvpos,float3 mvvel
  ,const unsigned *ridpmv,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    int pid=ridpmv[p+ini];
    if(pid>=0){
      //-Calcula desplazamiento y actualiza posicion.
	  //-Computes displacement and updates position.
      KerUpdatePos<periactive>(posxy[pid],posz[pid],mvpos.x,mvpos.y,mvpos.z,false,pid,posxy,posz,dcell,code);
      //-Calcula velocidad.
	  //-Computes velocity.
      velrhop[pid]=make_float4(mvvel.x,mvvel.y,mvvel.z,velrhop[pid].w);
    }
  }
}

//==============================================================================
/// Aplica un movimiento lineal a un conjunto de particulas.
/// Applies a linear movement to a set of particles.
//==============================================================================
void MoveLinBound(byte periactive,unsigned np,unsigned ini,tdouble3 mvpos,tfloat3 mvvel
  ,const unsigned *ridp,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  if(periactive)KerMoveLinBound<true>  <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridp,posxy,posz,dcell,velrhop,code);
  else          KerMoveLinBound<false> <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridp,posxy,posz,dcell,velrhop,code);
}



//------------------------------------------------------------------------------
/// Aplica un movimiento matricial a un conjunto de particulas.
/// Applies a linear movement to a set of particles.
//------------------------------------------------------------------------------
template<bool periactive,bool simulate2d> __global__ void KerMoveMatBound(unsigned n,unsigned ini,tmatrix4d m,double dt
  ,const unsigned *ridpmv,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    int pid=ridpmv[p+ini];
    if(pid>=0){
      double2 rxy=posxy[pid];
      double3 rpos=make_double3(rxy.x,rxy.y,posz[pid]);
      //-Calcula nueva posicion.
	  //-Computes new position.
      double3 rpos2;
      rpos2.x= rpos.x*m.a11 + rpos.y*m.a12 + rpos.z*m.a13 + m.a14;
      rpos2.y= rpos.x*m.a21 + rpos.y*m.a22 + rpos.z*m.a23 + m.a24;
      rpos2.z= rpos.x*m.a31 + rpos.y*m.a32 + rpos.z*m.a33 + m.a34;
      if(simulate2d)rpos2.y=rpos.y;
      //-Calcula desplazamiento y actualiza posicion.
	  //-Computes displacement and updates position.
      const double dx=rpos2.x-rpos.x;
      const double dy=rpos2.y-rpos.y;
      const double dz=rpos2.z-rpos.z;
      KerUpdatePos<periactive>(make_double2(rpos.x,rpos.y),rpos.z,dx,dy,dz,false,pid,posxy,posz,dcell,code);
      //-Calcula velocidad.
	  //-Computes velocity.
      velrhop[pid]=make_float4(float(dx/dt),float(dy/dt),float(dz/dt),velrhop[pid].w);
    }
  }
}

//==============================================================================
/// Aplica un movimiento matricial a un conjunto de particulas.
/// Applies a linear movement to a set of particles.
//==============================================================================
void MoveMatBound(byte periactive,bool simulate2d,unsigned np,unsigned ini,tmatrix4d m,double dt
  ,const unsigned *ridpmv,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  if(periactive){ const bool peri=true;
    if(simulate2d)KerMoveMatBound<peri,true>  <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,posxy,posz,dcell,velrhop,code);
    else          KerMoveMatBound<peri,false> <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,posxy,posz,dcell,velrhop,code);
  }
  else{ const bool peri=false;
    if(simulate2d)KerMoveMatBound<peri,true>  <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,posxy,posz,dcell,velrhop,code);
    else          KerMoveMatBound<peri,false> <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,posxy,posz,dcell,velrhop,code);
  }
}

//##############################################################################
//# Kernels for Floating bodies.
//##############################################################################
//==============================================================================
/// Calcula distancia entre pariculas floating y centro segun condiciones periodicas.
/// Computes distance between floating and centre particles according to periodic conditions.
//==============================================================================
template<bool periactive> __device__ void KerFtPeriodicDist(double px,double py,double pz,double cenx,double ceny,double cenz,float radius,float &dx,float &dy,float &dz){
  if(periactive){
    double ddx=px-cenx;
    double ddy=py-ceny;
    double ddz=pz-cenz;
    const unsigned peri=CTE.periactive;
    if((peri&1) && fabs(ddx)>radius){
      if(ddx>0){ ddx+=CTE.xperincx; ddy+=CTE.xperincy; ddz+=CTE.xperincz; }
      else{      ddx-=CTE.xperincx; ddy-=CTE.xperincy; ddz-=CTE.xperincz; }
    }
    if((peri&2) && fabs(ddy)>radius){
      if(ddy>0){ ddx+=CTE.yperincx; ddy+=CTE.yperincy; ddz+=CTE.yperincz; }
      else{      ddx-=CTE.yperincx; ddy-=CTE.yperincy; ddz-=CTE.yperincz; }
    }
    if((peri&4) && fabs(ddz)>radius){
      if(ddz>0){ ddx+=CTE.zperincx; ddy+=CTE.zperincy; ddz+=CTE.zperincz; }
      else{      ddx-=CTE.zperincx; ddy-=CTE.zperincy; ddz-=CTE.zperincz; }
    }
    dx=float(ddx);
    dy=float(ddy);
    dz=float(ddz);
  }
  else{
    dx=float(px-cenx);
    dy=float(py-ceny);
    dz=float(pz-cenz);
  }
}

//------------------------------------------------------------------------------
/// Calcula fuerzas sobre floatings.
/// Computes forces on floatings.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerFtCalcForces( //fdata={pini,np,radius,mass}
  float3 gravity,const float4 *ftodata,const float *ftomassp,const double3 *ftocenter,const unsigned *ftridp
  ,const double2 *posxy,const double *posz,const float3 *ace
  ,float3 *ftoforces)
{
  extern __shared__ float rfacex[];
  float *rfacey=rfacex+blockDim.x;
  float *rfacez=rfacey+blockDim.x;
  float *rfomegavelx=rfacez+blockDim.x;
  float *rfomegavely=rfomegavelx+blockDim.x;
  float *rfomegavelz=rfomegavely+blockDim.x;
  float *rinert11=rfomegavelz+blockDim.x;
  float *rinert12=rinert11+blockDim.x;
  float *rinert13=rinert12+blockDim.x;
  float *rinert21=rinert13+blockDim.x;
  float *rinert22=rinert21+blockDim.x;
  float *rinert23=rinert22+blockDim.x;
  float *rinert31=rinert23+blockDim.x;
  float *rinert32=rinert31+blockDim.x;
  float *rinert33=rinert32+blockDim.x;

  const unsigned tid=threadIdx.x;                      //-Numero de thread. //-Thread number.
  const unsigned cf=blockIdx.y*gridDim.x + blockIdx.x; //-Numero de floating. //-Floating number
  
  //-Carga datos de floatings.
  //-Loads floating data.
  float4 rfdata=ftodata[cf];
  const unsigned fpini=(unsigned)__float_as_int(rfdata.x);
  const unsigned fnp=(unsigned)__float_as_int(rfdata.y);
  const float fradius=rfdata.z;
  const float fmassp=ftomassp[cf];
  const double3 rcenter=ftocenter[cf];

  //-Inicializa memoria shared a Zero.
  //-Initialises shared memory to zero.
  const unsigned ntid=(fnp<blockDim.x? fnp: blockDim.x); //-Numero de threads utilizados. //-Number of used threads.
  if(tid<ntid){
    rfacex[tid]=rfacey[tid]=rfacez[tid]=0;
    rfomegavelx[tid]=rfomegavely[tid]=rfomegavelz[tid]=0;
    rinert11[tid]=rinert12[tid]=rinert13[tid]=0;
    rinert21[tid]=rinert22[tid]=rinert23[tid]=0;
    rinert31[tid]=rinert32[tid]=rinert33[tid]=0;
  }

  //-Calcula datos en memoria shared.
  //-Computes data in shared memory.
  const unsigned nfor=unsigned((fnp+blockDim.x-1)/blockDim.x);
  for(unsigned cfor=0;cfor<nfor;cfor++){
    unsigned p=cfor*blockDim.x+tid;
    if(p<fnp){
      const unsigned rp=ftridp[p+fpini];
      if(rp!=UINT_MAX){
        float3 race=ace[rp];
        race.x-=gravity.x; race.y-=gravity.y; race.z-=gravity.z;
        rfacex[tid]+=race.x; rfacey[tid]+=race.y; rfacez[tid]+=race.z;
        //-Calcula distancia al centro.
		//-Computes distance from the centre.
        double2 rposxy=posxy[rp];
        float dx,dy,dz;
        KerFtPeriodicDist<periactive>(rposxy.x,rposxy.y,posz[rp],rcenter.x,rcenter.y,rcenter.z,fradius,dx,dy,dz);
        //-Calcula omegavel.
		//-Computes omegavel.
        rfomegavelx[tid]+=(race.z*dy - race.y*dz);
        rfomegavely[tid]+=(race.x*dz - race.z*dx);
        rfomegavelz[tid]+=(race.y*dx - race.x*dy);
        //-Calcula inertia tensor.
		//-Computes inertia tensor.
        rinert11[tid]+= (dy*dy+dz*dz)*fmassp;
        rinert12[tid]+=-(dx*dy)*fmassp;
        rinert13[tid]+=-(dx*dz)*fmassp;
        rinert21[tid]+=-(dx*dy)*fmassp;
        rinert22[tid]+= (dx*dx+dz*dz)*fmassp;
        rinert23[tid]+=-(dy*dz)*fmassp;
        rinert31[tid]+=-(dx*dz)*fmassp;
        rinert32[tid]+=-(dy*dz)*fmassp;
        rinert33[tid]+= (dx*dx+dy*dy)*fmassp;
      }
    }
  }

  //-Reduce datos de memoria shared y guarda resultados.
  //-reduces data in shared memory and stores results.
  __syncthreads();
  if(!tid){
    float3 face=make_float3(0,0,0);
    float3 fomegavel=make_float3(0,0,0);
    float3 inert1=make_float3(0,0,0);
    float3 inert2=make_float3(0,0,0);
    float3 inert3=make_float3(0,0,0);
    for(unsigned c=0;c<ntid;c++){
      face.x+=rfacex[c];  face.y+=rfacey[c];  face.z+=rfacez[c];
      fomegavel.x+=rfomegavelx[c]; fomegavel.y+=rfomegavely[c]; fomegavel.z+=rfomegavelz[c];
      inert1.x+=rinert11[c];  inert1.y+=rinert12[c];  inert1.z+=rinert13[c];
      inert2.x+=rinert21[c];  inert2.y+=rinert22[c];  inert2.z+=rinert23[c];
      inert3.x+=rinert31[c];  inert3.y+=rinert32[c];  inert3.z+=rinert33[c];
    }
    //-Calculates the inverse of the intertia matrix to compute the I^-1 * L= W
    float3 invinert1=make_float3(0,0,0);
    float3 invinert2=make_float3(0,0,0);
    float3 invinert3=make_float3(0,0,0);
    const float detiner=(inert1.x*inert2.y*inert3.z+inert1.y*inert2.z*inert3.x+inert2.x*inert3.y*inert1.z-(inert3.x*inert2.y*inert1.z+inert2.x*inert1.y*inert3.z+inert2.z*inert3.y*inert1.x));
    if(detiner){
      invinert1.x= (inert2.y*inert3.z-inert2.z*inert3.y)/detiner;
      invinert1.y=-(inert1.y*inert3.z-inert1.z*inert3.y)/detiner;
      invinert1.z= (inert1.y*inert2.z-inert1.z*inert2.y)/detiner;
      invinert2.x=-(inert2.x*inert3.z-inert2.z*inert3.x)/detiner;
      invinert2.y= (inert1.x*inert3.z-inert1.z*inert3.x)/detiner;
      invinert2.z=-(inert1.x*inert2.z-inert1.z*inert2.x)/detiner;
      invinert3.x= (inert2.x*inert3.y-inert2.y*inert3.x)/detiner;
      invinert3.y=-(inert1.x*inert3.y-inert1.y*inert3.x)/detiner;
      invinert3.z= (inert1.x*inert2.y-inert1.y*inert2.x)/detiner;
    }
    //-Calcula omega a partir de fomegavel y invinert.
	//-Computes omega from fomegavel and invinert.
    {
      float3 omega;
      omega.x=(fomegavel.x*invinert1.x+fomegavel.y*invinert1.y+fomegavel.z*invinert1.z);
      omega.y=(fomegavel.x*invinert2.x+fomegavel.y*invinert2.y+fomegavel.z*invinert2.z);
      omega.z=(fomegavel.x*invinert3.x+fomegavel.y*invinert3.y+fomegavel.z*invinert3.z);
      fomegavel=omega;
    }
    //-Guarda resultados en ftoforces[].
	//-Stores results in ftoforces[].
    ftoforces[cf*2]=face;
    ftoforces[cf*2+1]=fomegavel;
  }
}

//==============================================================================
/// Calcula fuerzas sobre floatings.
/// Computes forces on floatings.
//==============================================================================
void FtCalcForces(bool periactive,unsigned ftcount
  ,tfloat3 gravity,const float4 *ftodata,const float *ftomassp,const double3 *ftocenter,const unsigned *ftridp
  ,const double2 *posxy,const double *posz,const float3 *ace
  ,float3 *ftoforces)
{
  if(ftcount){
    const unsigned bsize=128;
    const unsigned smem=sizeof(float)*(3+3+9)*bsize;
    dim3 sgrid=GetGridSize(ftcount*bsize,bsize);
    if(periactive)KerFtCalcForces<true>  <<<sgrid,bsize,smem>>> (Float3(gravity),ftodata,ftomassp,ftocenter,ftridp,posxy,posz,ace,ftoforces);
    else          KerFtCalcForces<false> <<<sgrid,bsize,smem>>> (Float3(gravity),ftodata,ftomassp,ftocenter,ftridp,posxy,posz,ace,ftoforces);
  }
}

//------------------------------------------------------------------------------
/// Updates information and particles of floating bodies.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerFtUpdate(bool predictor,bool simulate2d//fdata={pini,np,radius,mass}
  ,double dt,float3 gravity,const float4 *ftodata,const unsigned *ftridp
  ,const float3 *ftoforces,double3 *ftocenter,float3 *ftovel,float3 *ftoomega
  ,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  const unsigned tid=threadIdx.x;                      //-Numero de thread. //-Thread number.
  const unsigned cf=blockIdx.y*gridDim.x + blockIdx.x; //-Numero de floating. //floating number.
  //-Obtiene datos de floating.
  //-Obtains floating data.
  float4 rfdata=ftodata[cf];
  const unsigned fpini=(unsigned)__float_as_int(rfdata.x);
  const unsigned fnp=(unsigned)__float_as_int(rfdata.y);
  const float fradius=rfdata.z;
  const float fmass=rfdata.w;
  //-Calculo de face.
  //-Face computation.
  float3 face=ftoforces[cf*2];
  face.x=(face.x+fmass*gravity.x)/fmass;
  face.y=(face.y+fmass*gravity.y)/fmass;
  face.z=(face.z+fmass*gravity.z)/fmass;
  //-Calculo de fomega.
  //-fomega computation.
  float3 fomega=ftoomega[cf];
  {
    const float3 omega=ftoforces[cf*2+1];
    fomega.x=float(dt*omega.x+fomega.x);
    fomega.y=float(dt*omega.y+fomega.y);
    fomega.z=float(dt*omega.z+fomega.z);
  }
  float3 fvel=ftovel[cf];
  //-Anula componentes para 2D.
  //-Remove y components for 2D.
  if(simulate2d){ face.y=0; fomega.x=0; fomega.z=0; fvel.y=0; }
  //-Calculo de fcenter.
  //-fcenter computation.
  double3 fcenter=ftocenter[cf];
  fcenter.x+=dt*fvel.x;
  fcenter.y+=dt*fvel.y;
  fcenter.z+=dt*fvel.z;
  //-Calculo de fvel.
  //-fvel computation.
  fvel.x=float(dt*face.x+fvel.x);
  fvel.y=float(dt*face.y+fvel.y);
  fvel.z=float(dt*face.z+fvel.z);

  //-Updates floating particles.
  const unsigned nfor=unsigned((fnp+blockDim.x-1)/blockDim.x);
  for(unsigned cfor=0;cfor<nfor;cfor++){
    unsigned fp=cfor*blockDim.x+tid;
    if(fp<fnp){
      const unsigned p=ftridp[fp+fpini];
      if(p!=UINT_MAX){
        double2 rposxy=posxy[p];
        double rposz=posz[p];
        float4 rvel=velrhop[p];
        //-Calcula y graba desplazamiento de posicion.
		//-Computes and stores position displacement.
        const double dx=dt*double(rvel.x);
        const double dy=dt*double(rvel.y);
        const double dz=dt*double(rvel.z);
        KerUpdatePos<periactive>(rposxy,rposz,dx,dy,dz,false,p,posxy,posz,dcell,code);
        //-Calcula y graba nueva velocidad.
		//-Computes and stores new velocity.
        float disx,disy,disz;
        KerFtPeriodicDist<periactive>(rposxy.x+dx,rposxy.y+dy,rposz+dz,fcenter.x,fcenter.y,fcenter.z,fradius,disx,disy,disz);
        rvel.x=fvel.x+(fomega.y*disz-fomega.z*disy);
        rvel.y=fvel.y+(fomega.z*disx-fomega.x*disz);
        rvel.z=fvel.z+(fomega.x*disy-fomega.y*disx);
        velrhop[p]=rvel;
      }
    }
  }

  //-Stores floating data.
  __syncthreads();
  if(!tid && !predictor){
    ftocenter[cf]=(periactive? KerUpdatePeriodicPos(fcenter): fcenter);
    ftovel[cf]=fvel;
    ftoomega[cf]=fomega;
  }
}

//==============================================================================
/// Updates information and particles of floating bodies.
//==============================================================================
void FtUpdate(bool periactive,bool predictor,bool simulate2d,unsigned ftcount
  ,double dt,tfloat3 gravity,const float4 *ftodata,const unsigned *ftridp
  ,const float3 *ftoforces,double3 *ftocenter,float3 *ftovel,float3 *ftoomega
  ,double2 *posxy,double *posz,unsigned *dcell,float4 *velrhop,word *code)
{
  if(ftcount){
    const unsigned bsize=128;
    dim3 sgrid=GetGridSize(ftcount*bsize,bsize);
    if(periactive)KerFtUpdate<true>  <<<sgrid,bsize>>> (predictor,simulate2d,dt,Float3(gravity),ftodata,ftridp,ftoforces,ftocenter,ftovel,ftoomega,posxy,posz,dcell,velrhop,code);
    else          KerFtUpdate<false> <<<sgrid,bsize>>> (predictor,simulate2d,dt,Float3(gravity),ftodata,ftridp,ftoforces,ftocenter,ftovel,ftoomega,posxy,posz,dcell,velrhop,code);
  }
}


//##############################################################################
//# Kernels para Periodic conditions
//# Kernels for Periodic conditions
//##############################################################################
//------------------------------------------------------------------------------
/// Marca las periodicas actuales como ignorar.
/// Marks current periodics to be ignored.
//------------------------------------------------------------------------------
__global__ void KerPeriodicIgnore(unsigned n,word *code)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    //-Comprueba codigo de particula.
	//-Checks code of particles.
    const word rcode=code[p];
    if(CODE_GetSpecialValue(rcode)==CODE_PERIODIC)code[p]=CODE_SetOutIgnore(rcode);
  }
}

//==============================================================================
/// Marca las periodicas actuales como ignorar.
/// Marks current periodics to be ignored.
//==============================================================================
void PeriodicIgnore(unsigned n,word *code){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerPeriodicIgnore <<<sgrid,SPHBSIZE>>> (n,code);
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Crea lista de nuevas particulas periodicas a duplicar y con delper activado
/// marca las periodicas viejas para ignorar.
/// - EN:
/// Create list of new periodic particles to be duplicated and 
/// marks old periodics to be ignored.
//------------------------------------------------------------------------------
__global__ void KerPeriodicMakeList(unsigned n,unsigned pini,unsigned nmax
  ,double3 mapposmin,double3 mapposmax,double3 perinc
  ,const double2 *posxy,const double *posz,const word *code,unsigned *listp)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    const unsigned p2=p+pini;
    //-Se queda con particulas normales o periodicas.
	//-Inteacts with normal or periodic particles.
    if(CODE_GetSpecialValue(code[p2])<=CODE_PERIODIC){
      //-Obtiene posicion de particula.
	  //-Obtains particle position.
      const double2 rxy=posxy[p2];
      const double rx=rxy.x,ry=rxy.y;
      const double rz=posz[p2];
      double rx2=rx+perinc.x,ry2=ry+perinc.y,rz2=rz+perinc.z;
      if(mapposmin.x<=rx2 && mapposmin.y<=ry2 && mapposmin.z<=rz2 && rx2<mapposmax.x && ry2<mapposmax.y && rz2<mapposmax.z){
        unsigned cp=atomicAdd(slist,1);  slist[cp+1]=p2;
      }
      rx2=rx-perinc.x; ry2=ry-perinc.y; rz2=rz-perinc.z;
      if(mapposmin.x<=rx2 && mapposmin.y<=ry2 && mapposmin.z<=rz2 && rx2<mapposmax.x && ry2<mapposmax.y && rz2<mapposmax.z){
        unsigned cp=atomicAdd(slist,1);  slist[cp+1]=(p2|0x80000000);
      }
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+nmax),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    unsigned cp=slist[0]+threadIdx.x;
    if(cp<nmax)listp[cp]=slist[threadIdx.x+1];
  }
  if(blockDim.x+threadIdx.x<ns){//-Puede haber el doble de periodicas que threads. //-There may be twice as many periodics per thread
    unsigned cp=blockDim.x+slist[0]+threadIdx.x;
    if(cp<nmax)listp[cp]=slist[blockDim.x+threadIdx.x+1];
  }
}

//==============================================================================
/// ES:
/// Crea lista de nuevas particulas periodicas a duplicar.
/// Con stable activado reordena lista de periodicas.
/// - EN:
/// Create list of new periodic particles to be duplicated.
/// With stable activated reorders perioc list.
//==============================================================================
unsigned PeriodicMakeList(unsigned n,unsigned pini,bool stable,unsigned nmax
  ,tdouble3 mapposmin,tdouble3 mapposmax,tdouble3 perinc
  ,const double2 *posxy,const double *posz,const word *code,unsigned *listp)
{
  unsigned count=0;
  if(n){
    //-Inicializa tamaño de lista lspg a cero.
	//-Lspg size list initialized to zero.
    cudaMemset(listp+nmax,0,sizeof(unsigned));
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE*2+1)*sizeof(unsigned); //-De cada particula pueden salir 2 nuevas periodicas mas la posicion del contador. //-Each particle can leave two new periodic over the counter position.
    KerPeriodicMakeList <<<sgrid,SPHBSIZE,smem>>> (n,pini,nmax,Double3(mapposmin),Double3(mapposmax),Double3(perinc),posxy,posz,code,listp);
    cudaMemcpy(&count,listp+nmax,sizeof(unsigned),cudaMemcpyDeviceToHost);
    //-Reordena lista si es valida y stable esta activado.
	//-Reorders list if it is valid and stable has been activated.
    if(stable && count && count<=nmax){
      thrust::device_ptr<unsigned> dev_list(listp);
      thrust::sort(dev_list,dev_list+count);
    }
  }
  return(count);
}

//------------------------------------------------------------------------------
/// ES:
/// Duplica la posicion de la particula indicada aplicandole un desplazamiento.
/// Las particulas duplicadas se considera que siempre son validas y estan dentro
/// del dominio.
/// Este kernel vale para single-gpu y multi-gpu porque los calculos se hacen 
/// a partir de domposmin.
/// Se controla que las coordendas de celda no sobrepasen el maximo.
/// - EN:
/// Doubles the position of the indicated particle using a displacement.
/// Duplicate particles are considered valid and are always within
/// the domain.
/// This kernel applies to single-GPU and multi-GPU because the calculations are made
/// from domposmin.
/// It controls the cell coordinates not exceed the maximum.
//------------------------------------------------------------------------------
__device__ void KerPeriodicDuplicatePos(unsigned pnew,unsigned pcopy
  ,bool inverse,double dx,double dy,double dz,uint3 cellmax
  ,double2 *posxy,double *posz,unsigned *dcell)
{
  //-Obtiene pos de particula a duplicar.
  //-Obtainsposition of the particle to be duplicated.
  double2 rxy=posxy[pcopy];
  double rz=posz[pcopy];
  //-Aplica desplazamiento.
  //-Applies displacement.
  rxy.x+=(inverse? -dx: dx);
  rxy.y+=(inverse? -dy: dy);
  rz+=(inverse? -dz: dz);
  //-Calcula coordendas de celda dentro de dominio.
  //-Computes cell coordinates within the domain.
  unsigned cx=unsigned((rxy.x-CTE.domposminx)/CTE.scell);
  unsigned cy=unsigned((rxy.y-CTE.domposminy)/CTE.scell);
  unsigned cz=unsigned((rz-CTE.domposminz)/CTE.scell);
  //-Ajusta las coordendas de celda si sobrepasan el maximo.
  //-Adjust cell coordinates if they exceed the maximum.
  cx=(cx<=cellmax.x? cx: cellmax.x);
  cy=(cy<=cellmax.y? cy: cellmax.y);
  cz=(cz<=cellmax.z? cz: cellmax.z);
  //-Graba posicion y celda de nuevas particulas.
  //-Stores position and cell of the new particles.
  posxy[pnew]=rxy;
  posz[pnew]=rz;
  dcell[pnew]=PC__Cell(CTE.cellcode,cx,cy,cz);
}

//------------------------------------------------------------------------------
/// ES:
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
/// - EN:
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
//------------------------------------------------------------------------------
__global__ void KerPeriodicDuplicateVerlet(unsigned n,unsigned pini,uint3 cellmax,double3 perinc
  ,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,tsymatrix3f *spstau,float4 *velrhopm1)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Ajusta posicion y celda de nueva particula.
	//-Adjusts cell position of the new particles.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copia el resto de datos.
	//-Copies the remaining data.
    idp[pnew]=idp[pcopy];
    code[pnew]=CODE_SetPeriodic(code[pcopy]);
    velrhop[pnew]=velrhop[pcopy];
    velrhopm1[pnew]=velrhopm1[pcopy];
    if(spstau)spstau[pnew]=spstau[pcopy];
  }
}

//==============================================================================
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Creates periodic particles from a list of particles to duplicate.
//==============================================================================
void PeriodicDuplicateVerlet(unsigned n,unsigned pini,tuint3 domcells,tdouble3 perinc
  ,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,tsymatrix3f *spstau,float4 *velrhopm1)
{
  if(n){
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerPeriodicDuplicateVerlet <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,spstau,velrhopm1);
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
/// - EN:
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
//------------------------------------------------------------------------------
template<bool varspre> __global__ void KerPeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,uint3 cellmax,double3 perinc,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,tsymatrix3f *spstau,double2 *posxypre,double *poszpre,float4 *velrhoppre)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Ajusta posicion y celda de nueva particula.
	//-Adjusts cell position of the new particles.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copia el resto de datos.
	//-Copies the remaining data.
    idp[pnew]=idp[pcopy];
    code[pnew]=CODE_SetPeriodic(code[pcopy]);
    velrhop[pnew]=velrhop[pcopy];
    if(varspre){
      posxypre[pnew]=posxypre[pcopy];
      poszpre[pnew]=poszpre[pcopy];
      velrhoppre[pnew]=velrhoppre[pcopy];
    }
    if(spstau)spstau[pnew]=spstau[pcopy];
  }
}

//==============================================================================
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Creates periodic particles from a list of particles to duplicate.
//==============================================================================
void PeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,tuint3 domcells,tdouble3 perinc,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,tsymatrix3f *spstau,double2 *posxypre,double *poszpre,float4 *velrhoppre)
{
  if(n){
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    if(posxypre!=NULL)KerPeriodicDuplicateSymplectic<true>  <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,spstau,posxypre,poszpre,velrhoppre);
    else              KerPeriodicDuplicateSymplectic<false> <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,spstau,posxypre,poszpre,velrhoppre);
  }
}


//##############################################################################
//# Kernels para external forces (JSphVarAcc)
//# Kernels for external forces (JSphVarAcc)
//##############################################################################
//------------------------------------------------------
/// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddVarAccAng(unsigned n,unsigned pini,word codesel,float3 gravity
  ,bool setgravity,double3 acclin,double3 accang,double3 centre,double3 velang,double3 vellin
  ,const word *code,const double2 *posxy,const double *posz,const float4 *velrhop,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
    p+=pini;
    //Check if the current particle is part of the particle set by its Mk
    if(CODE_GetTypeValue(code[p])==codesel){
      const float3 accf=ace[p]; //-Gets the current particles acceleration value.
      double accx=accf.x,accy=accf.y,accz=accf.z;
      //-Adds linear acceleration.
      accx+=acclin.x;  accy+=acclin.y;  accz+=acclin.z;
      //-Subtract global gravity from the acceleration if it is set in the input file
      if(!setgravity){
        accx-=gravity.x;  accy-=gravity.y;  accz-=gravity.z; 
      }

      //-Adds angular acceleration.
      const double2 rxy=posxy[p];
      const double dcx=rxy.x-centre.x;
      const double dcy=rxy.y-centre.y;
      const double dcz=posz[p]-centre.z;
      //-Get the current particle's velocity
      const float4 rvel=velrhop[p];
      const double velx=rvel.x-vellin.x;
      const double vely=rvel.y-vellin.y;
      const double velz=rvel.z-vellin.z;

      //-Calculate angular acceleration ((Dw/Dt) x (r_i - r)) + (w x (w x (r_i - r))) + (2w x (v_i - v))
      //(Dw/Dt) x (r_i - r) (term1)
      accx+=(accang.y*dcz)-(accang.z*dcy);
      accy+=(accang.z*dcx)-(accang.x*dcz);
      accz+=(accang.x*dcy)-(accang.y*dcx);

      //Centripetal acceleration (term2)
      //First find w x (r_i - r))
      const double innerx=(velang.y*dcz)-(velang.z*dcy);
      const double innery=(velang.z*dcx)-(velang.x*dcz);
      const double innerz=(velang.x*dcy)-(velang.y*dcx);
      //Find w x inner
      accx+=(velang.y*innerz)-(velang.z*innery);
      accy+=(velang.z*innerx)-(velang.x*innerz);
      accz+=(velang.x*innery)-(velang.y*innerx);

      //Coriolis acceleration 2w x (v_i - v) (term3)
      accx+=((2.0*velang.y)*velz)-((2.0*velang.z)*vely);
      accy+=((2.0*velang.z)*velx)-((2.0*velang.x)*velz);
      accz+=((2.0*velang.x)*vely)-((2.0*velang.y)*velx);

      //-Stores the new acceleration value.
      ace[p]=make_float3(float(accx),float(accy),float(accz));
    }
  }
}

//------------------------------------------------------
/// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddVarAccLin(unsigned n,unsigned pini,word codesel,float3 gravity
  ,bool setgravity,double3 acclin,const word *code,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
    p+=pini;
    //Check if the current particle is part of the particle set by its Mk
    if(CODE_GetTypeValue(code[p])==codesel){
      const float3 accf=ace[p]; //-Gets the current particles acceleration value.
      double accx=accf.x,accy=accf.y,accz=accf.z;
      //-Adds linear acceleration.
      accx+=acclin.x;  accy+=acclin.y;  accz+=acclin.z;
      //-Subtract global gravity from the acceleration if it is set in the input file
      if(!setgravity){
        accx-=gravity.x;  accy-=gravity.y;  accz-=gravity.z; 
      }
      //-Stores the new acceleration value.
      ace[p]=make_float3(float(accx),float(accy),float(accz));
    }
  }
}

//==================================================================================================
/// Adds variable acceleration forces for particle MK groups that have an input file.
//==================================================================================================
void AddVarAcc(unsigned n,unsigned pini,word codesel
  ,tdouble3 acclin,tdouble3 accang,tdouble3 centre,tdouble3 velang,tdouble3 vellin,bool setgravity
  ,tfloat3 gravity,const word *code,const double2 *posxy,const double *posz,const float4 *velrhop,float3 *ace)
{
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    const bool withaccang=(accang.x!=0 || accang.y!=0 || accang.z!=0);
    if(withaccang)KerAddVarAccAng <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(gravity),setgravity,Double3(acclin),Double3(accang),Double3(centre),Double3(velang),Double3(vellin),code,posxy,posz,velrhop,ace);
    else          KerAddVarAccLin <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(gravity),setgravity,Double3(acclin),code,ace);
  }
}

//==========================
///Initial advection - r*
//==========================
template<bool floating> __global__ void KerComputeRStar(unsigned npf,unsigned npb,const float4 *velrhoppre,double dtm,word *code,double2 *movxy,double *movz)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
    unsigned p1=p+npb;
    //-Particulas: Fixed & Moving //-Particles: Fixed & Moving
      float4 rvelrhop=velrhoppre[p1];
      
      if(!floating || CODE_GetType(code[p1])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
        //-Calcula y graba desplazamiento de posicion.
		//-Computes and stores position displacement.
        double dx=double(rvelrhop.x)*dtm;
        double dy=double(rvelrhop.y)*dtm;
        double dz=double(rvelrhop.z)*dtm;
        
        movxy[p1]=make_double2(dx,dy);
        movz[p1]=dz;  
    }
  }
}

void ComputeRStar(bool floating,unsigned npf,unsigned npb,const float4 *velrhoppre,double dtm,word *code,double2 *movxy,double *movz)
{
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    if(floating)KerComputeRStar<true> <<<sgrid,SPHBSIZE>>> (npf,npb,velrhoppre,dtm,code,movxy,movz);
    else        KerComputeRStar<false><<<sgrid,SPHBSIZE>>> (npf,npb,velrhoppre,dtm,code,movxy,movz);
  }
}

//==============================================================================
///Find Irelation
//==============================================================================
//------------------------------------------------------------------------------
/// Devuelve drx, dry y drz entre dos particulas.
/// Returns drx, dry and drz between the particles.
//------------------------------------------------------------------------------
__device__ void KerGetParticlesIrelationDr(int p2
  ,const double2 *posxy,const double *posz,const double3 &posdp1
  ,float &drx,float &dry,float &drz)
{
  double2 posp2=posxy[p2];
  drx=float(posdp1.x-posp2.x);
  dry=float(posdp1.y-posp2.y);
  drz=float(posdp1.z-posz[p2]);
}
//------------------------------------------------------------------------------
/// Returns postion and vel of a particle.
//------------------------------------------------------------------------------
__device__ void KerGetParticleDataIrelation(unsigned p1
  ,const double2 *posxy,const double *posz,double3 &posdp1)
{
  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
}

//==============================================================================
///Irelation
//==============================================================================

__device__ void KerFindIrelationCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz
  ,const word *code,const unsigned *idp,float massp2,double3 posdp1,unsigned idpg1,unsigned *irelationg,float &closestr)
{
  for(int p2=pini;p2<pfin;p2++)if(CODE_GetTypeValue(code[p2])==0){
    float drx,dry,drz;
    KerGetParticlesIrelationDr(p2,posxy,posz,posdp1,drx,dry,drz);
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=closestr){
	    closestr=rr2;
      irelationg[idpg1]=idp[p2];
    }
  }
}

__global__ void KerFindIrelation
  (unsigned npb,const double2 *posxy,const double *posz,const word *code,const unsigned *idp,unsigned *irelationg)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
  if(p1<npb){
    if(CODE_GetTypeValue(code[p1])==1){
      //-Carga datos de particula p1.
	    //-Loads particle p1 data.
      double3 posdp1;
      KerGetParticleDataIrelation(p1,posxy,posz,posdp1);
      unsigned idpg1=idp[p1];
      irelationg[idpg1]=npb;
      float closestr=CTE.fourh2;

      KerFindIrelationCalc(p1,0,npb,posxy,posz,code,idp,CTE.massf,posdp1,idpg1,irelationg,closestr);
    }
  }
}

void FindIrelation(const unsigned bsbound,unsigned npb,const double2 *posxy
  ,const double *posz,const word *code,const unsigned *idp,unsigned *irelationg){
  if(npb){
    dim3 sgridb=GetGridSize(npb,bsbound);
    KerFindIrelation <<<sgridb,bsbound>>> (npb,posxy,posz,code,idp,irelationg);
  }
}

//==============================================================================
///Kernel Correction
//==============================================================================   
template<bool psimple> __device__ void KerKernelCorCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,float massp2,double3 posdp1,float3 posp1,double3 &dwxcorrgp1,double3 &dwzcorrgp1,const word *code)
{
  for(int p2=pini;p2<pfin;p2++)if(CODE_GetTypeValue(code[p2])==0||CODE_GetTypeValue(code[p2])==2){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<psimple> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
	  
      const double volumep2=massp2/CTE.rhopzero; //Volume of particle j 
      /*float3 tempX=make_float3(-volumep2*frx*drx-errorX.x,0,-volumep2*frz*drx-errorX.z);
      float3 tempZ=make_float3(-volumep2*frx*drz-errorZ.x,0,-volumep2*frz*drz-errorZ.z);
      float3 tempTx=make_float3(dwxcorrgp1.x+tempX.x,0,dwxcorrgp1.z+tempX.z);
      float3 tempTz=make_float3(dwzcorrgp1.x+tempZ.x,0,dwzcorrgp1.z+tempZ.z);
      errorX.x=(tempTx.x-dwxcorrgp1.x)-tempX.x; errorX.z=(tempTx.z-dwxcorrgp1.z)-tempX.z;
      errorZ.x=(tempTz.x-dwzcorrgp1.x)-tempZ.x; errorZ.z=(tempTz.z-dwzcorrgp1.z)-tempZ.z;
      dwxcorrgp1.x=tempTx.x; dwxcorrgp1.z=tempTx.z;
      dwzcorrgp1.x=tempTz.x; dwzcorrgp1.z=tempTz.z;*/
      dwxcorrgp1.x-=volumep2*frx*drx; dwxcorrgp1.z-=volumep2*frz*drx;
      dwzcorrgp1.x-=volumep2*frx*drz; dwzcorrgp1.z-=volumep2*frz*drz;
    }
  }
}

__global__ void KerInverseKernelCor(unsigned n,unsigned pinit,double3 *dwxcorrg,double3 *dwzcorrg,const word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
      const float det=1.0f/(dwxcorrg[p1].x*dwzcorrg[p1].z-dwxcorrg[p1].z*dwzcorrg[p1].x);

      if(det){
	      const float temp=dwxcorrg[p1].x;
        dwxcorrg[p1].x=dwzcorrg[p1].z*det;
	      dwxcorrg[p1].z=-dwxcorrg[p1].z*det; 
	      dwzcorrg[p1].x=-dwzcorrg[p1].x*det;
	      dwzcorrg[p1].z=temp*det;
      }
    }
  }
}

template<bool psimple> __global__ void KerKernelCorrection
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,double3 *dwxcorrg,double3 *dwzcorrg,const word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
      unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
      if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
      //-Obtiene datos basicos de particula p1.
  	  //-Obtains basic data of particle p1.
      double3 posdp1;
      float3 posp1,velp1;
      float rhopp1,pressp1;
      KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    
      double3 dwxcorrgp1=make_double3(0,0,0);
      double3 dwzcorrgp1=make_double3(0,0,0);
      //-Obtiene limites de interaccion
	    //-Obtains interaction limits
      int cxini,cxfin,yini,yfin,zini,zfin;
      KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Interaccion con Fluidas.
	    //-Interaction with fluids.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin){
            KerKernelCorCalc<psimple> (p1,pini,pfin,posxy,posz,pospress,CTE.massf,posdp1,posp1,dwxcorrgp1,dwzcorrgp1,code);
          }
	      }
      }

      if(dwxcorrgp1.x||dwxcorrgp1.y||dwxcorrgp1.z||dwzcorrgp1.x||dwzcorrgp1.y||dwzcorrgp1.z){
        double3 dwx=dwxcorrg[p1];
        double3 dwz=dwzcorrg[p1];
        dwx.x+=dwxcorrgp1.x; dwx.y+=dwxcorrgp1.y; dwx.z+=dwxcorrgp1.z;
        dwz.x+=dwzcorrgp1.x; dwz.y+=dwzcorrgp1.y; dwz.z+=dwzcorrgp1.z;
        dwxcorrg[p1]=dwx;
        dwzcorrg[p1]=dwz;
        dwxcorrgp1=make_double3(0,0,0);
        dwzcorrgp1=make_double3(0,0,0);
      }

      //-Interaccion con contorno.
	    //-Interaction with boundaries.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z;
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin){
            KerKernelCorCalc<psimple> (p1,pini,pfin,posxy,posz,pospress,CTE.massf,posdp1,posp1,dwxcorrgp1,dwzcorrgp1,code);
          }
        }
      }

      if(dwxcorrgp1.x||dwxcorrgp1.y||dwxcorrgp1.z||dwzcorrgp1.x||dwzcorrgp1.y||dwzcorrgp1.z){
        double3 dwx=dwxcorrg[p1];
        double3 dwz=dwzcorrg[p1];
        dwx.x+=dwxcorrgp1.x; dwx.y+=dwxcorrgp1.y; dwx.z+=dwxcorrgp1.z;
        dwz.x+=dwzcorrgp1.x; dwz.y+=dwzcorrgp1.y; dwz.z+=dwzcorrgp1.z;
        dwxcorrg[p1]=dwx;
        dwzcorrg[p1]=dwz;
      }
    }
  }
}

//==============================================================================
/// Kernel Correction
//==============================================================================
void KernelCorrection(bool psimple,TpCellMode cellmode
  ,const unsigned bsfluid,const unsigned bsbound,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,double3 *dwxcorrg,double3 *dwzcorrg,const word *code){

  const unsigned npf=np-npb;
  int hdiv=(cellmode==CELLMODE_H? 2: 1);
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  unsigned cellfluid=nc.w*nc.z+1;
  int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(psimple){
      KerKernelCorrection<true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,code);
      KerKernelCorrection<true> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,code);
    }
    else{
      KerKernelCorrection<false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,code);
      KerKernelCorrection<false> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,code);
    }
    KerInverseKernelCor <<<sgridf,bsfluid>>> (npf,npb,dwxcorrg,dwzcorrg,code);
    KerInverseKernelCor <<<sgridb,bsbound>>> (npbok,0,dwxcorrg,dwzcorrg,code);
  }
}




//==============================================================================
///Matrix Order 
//==============================================================================
__global__ void KerMatrixOrderFluid
  (unsigned npf,unsigned npb,unsigned *porder,const unsigned matrixBound)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
  if(p<npf){
    unsigned p1=p+npb;
    porder[p1]=matrixBound+p;
  }
}

void MatrixOrderFluid(const unsigned bsfluid,unsigned np,unsigned npb, unsigned *porder,const unsigned matrixBound){
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    KerMatrixOrderFluid <<<sgridf,bsfluid>>> (npf,npb,porder,matrixBound);
  }
}

__device__ void KerMatrixOrderDummyCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const word *code,const unsigned *idp,unsigned idpg1,unsigned irelationg1,unsigned *porder)
{
  for(int p2=pini;p2<pfin;p2++)if(irelationg1==idp[p2])porder[p1]=porder[p2];
}

__global__ void KerMatrixOrderDummy
  (unsigned np,unsigned n,int hdiv,uint4 nc,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const word *code,const unsigned *idp,const unsigned *irelationg,unsigned *porder)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
  if(p1<n){
    if(CODE_GetTypeValue(code[p1])==1){
      porder[p1]=np;
      //-Carga datos de particula p1.
	    //-Loads particle p1 data.
      unsigned idpg1=idp[p1];
      unsigned irelationg1=irelationg[idpg1];
      //-Obtiene limites de interaccion
	    //-Obtains interaction limits
      int cxini,cxfin,yini,yfin,zini,zfin;
      KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Interaccion de Contorno con Fluidas.
	    //-Boundary-Bound interaction.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z;
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin)KerMatrixOrderDummyCalc(p1,pini,pfin,code,idp,idpg1,irelationg1,porder);
        }
      }
    }
  }
}

void MatrixOrderDummy(TpCellMode cellmode
  ,const unsigned bsbound,unsigned np,unsigned npb,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const word *code,const unsigned *idp,const unsigned *irelationg,unsigned *porder){

  int hdiv=(cellmode==CELLMODE_H? 2: 1);
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Bound-Bound
  if(npb){
    dim3 sgridb=GetGridSize(npb,bsbound);
    KerMatrixOrderDummy <<<sgridb,bsbound>>> (np,npb,hdiv,nc,begincell,cellzero,dcell,code,idp,irelationg,porder);
  }
}

//==============================================================================
///Free Surface Identification
//==============================================================================
template<bool psimple> __device__ void KerFreeSurfaceCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,float massp2,double3 posdp1,float3 posp1,double &divr1)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<false> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
	  
      const double volumep2=massp2/CTE.rhopzero; //Volume of particle j 
      const double rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
			divr1-=volumep2*rDivW;
    }
  }
}

template<bool psimple> __global__ void KerFreeSurfaceFind
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop,double *divr,const word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
      unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
      if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
      //-Obtiene datos basicos de particula p1.
  	  //-Obtains basic data of particle p1.
      double3 posdp1;
      float3 posp1,velp1;
      float rhopp1,pressp1;
      double divr1=0.0;
      KerGetParticleData<false>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    
      //-Obtiene limites de interaccion
	    //-Obtains interaction limits
      int cxini,cxfin,yini,yfin,zini,zfin;
      KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

      //-Interaccion con Fluidas.
	    //-Interaction with fluids.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin){
		        KerFreeSurfaceCalc<false> (p1,pini,pfin,posxy,posz,pospress,CTE.massf,posdp1,posp1,divr1);
          }
	      }
      }

      if(divr1){
        divr[p1]+=divr1;
        divr1=0;
      }

      //-Interaccion con contorno.
	    //-Interaction with boundaries.
      for(int z=zini;z<zfin;z++){
        int zmod=(nc.w)*z;
        for(int y=yini;y<yfin;y++){
          int ymod=zmod+nc.x*y;
          unsigned pini,pfin=0;
          for(int x=cxini;x<cxfin;x++){
            int2 cbeg=begincell[x+ymod];
            if(cbeg.y){
              if(!pfin)pini=cbeg.x;
              pfin=cbeg.y;
            }
          }
          if(pfin){
            KerFreeSurfaceCalc<false> (p1,pini,pfin,posxy,posz,pospress,CTE.massf,posdp1,posp1,divr1);
		      }
        }
      }

      if(divr1) divr[p1]+=divr1;
    }
  }
}

void FreeSurfaceFind(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop
  ,const word *code,const unsigned *idp,double *divr){
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);
    
    if(psimple){
      KerFreeSurfaceFind<true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code);
      KerFreeSurfaceFind<true> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code);
    }
    else{
      KerFreeSurfaceFind<false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code);
      KerFreeSurfaceFind<false> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code);
    }
  }
}

//==============================================================================
///Free Surface Mark
//==============================================================================
template<bool psimple> __global__ void KerFreeSurfaceMark
  (unsigned n,unsigned pinit,double *divr,double *matrixInd, double *matrixb,unsigned int *row,const unsigned *porder,const word *code,const double pi,const double freesurface)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
      unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
      if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
        unsigned oi=porder[p1];
        const int Mark=row[oi]+1;
        if(divr[p1]>=freesurface && divr[p1]<=freesurface+0.2f){
          double alpha=0.5*(1.0-cos(pi*double(divr[p1]-freesurface)/0.2));

          matrixb[oi]=matrixb[oi]*alpha;

          for(int index=Mark;index<row[oi+1];index++) matrixInd[index]=matrixInd[index]*alpha;
        }
    }
  }
}

void FreeSurfaceMark(bool psimple,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,double *divr
  ,double *matrixInd, double *matrixb,unsigned int *row,const unsigned *porder,const word *code,const double pi,const double freesurface){
  const unsigned npf=np-npb;

  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);

    if(psimple){
      KerFreeSurfaceMark<true> <<<sgridf,bsfluid>>> (npf,npb,divr,matrixInd,matrixb,row,porder,code,pi,freesurface);
      KerFreeSurfaceMark<true> <<<sgridb,bsbound>>> (npbok,0,divr,matrixInd,matrixb,row,porder,code,pi,freesurface);
    }
    else{
      KerFreeSurfaceMark<false> <<<sgridf,bsfluid>>> (npf,npb,divr,matrixInd,matrixb,row,porder,code,pi,freesurface);
      KerFreeSurfaceMark<false> <<<sgridb,bsbound>>> (npbok,0,divr,matrixInd,matrixb,row,porder,code,pi,freesurface);
    }
  }
}

//==============================================================================
///Matrix B Population
//==============================================================================
template<bool psimple> __device__ void KerPopMatrixBCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,const float3 velp1,const double massp2,const double3 posdp1,const float3 posp1,const double3 dwxcorrgp1,const double3 dwzcorrgp1,double &matrixbp1,const word *code)
{
  for(int p2=pini;p2<pfin;p2++)if(CODE_GetTypeValue(code[p2])==0||CODE_GetTypeValue(code[p2])==2){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<false> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
	  
      const double volumep2=massp2/double(CTE.rhopzero); //Volume of particle j 

      const double dvx=velp1.x-double(velrhop[p2].x), dvz=velp1.z-double(velrhop[p2].z);
			const double temp_x=frx*dwxcorrgp1.x+frz*dwzcorrgp1.x;
			const double temp_z=frx*dwxcorrgp1.z+frz*dwzcorrgp1.z;
			double temp=dvx*temp_x+dvz*temp_z;

      matrixbp1-=volumep2*temp;
    }
  }
}

template<bool psimple> __global__ void KerPopulateMatrixB
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const double3 *dwxcorrg,const double3 *dwzcorrg
  ,double *matrixb,const unsigned *porder,const unsigned *idp,const double dt,const unsigned ppedim,const double *divr,const word *code,const double freesurface)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
      if(divr[p1]>freesurface){
        //-Obtiene datos basicos de particula p1.
  	    //-Obtains basic data of particle p1.
        double3 posdp1;
        float3 velp1;
        float3 posp1;
        float rhopp1,pressp1;
        const double3 dwxcorrgp1=dwxcorrg[p1];
        const double3 dwzcorrgp1=dwzcorrg[p1];
        double matrixbp1=0.0;
        const unsigned oi=porder[p1];
        KerGetParticleData<false>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);

        //-Obtiene limites de interaccion
	      //-Obtains interaction limits
        int cxini,cxfin,yini,yfin,zini,zfin;
        KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

        //-Interaccion con Fluidas.
	      //-Interaction with fluids.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
		          KerPopMatrixBCalc<psimple> (p1,pini,pfin,posxy,posz,pospress,velrhop,velp1,double(CTE.massf),posdp1,posp1,dwxcorrgp1,dwzcorrgp1,matrixbp1,code);
            }
	        }
        }

        matrixb[oi]+=double(matrixbp1);
        matrixbp1=0;

        //-Interaccion con contorno.
	      //-Interaction with boundaries.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z;
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
              KerPopMatrixBCalc<psimple> (p1,pini,pfin,posxy,posz,pospress,velrhop,velp1,CTE.massf,posdp1,posp1,dwxcorrgp1,dwzcorrgp1,matrixbp1,code);
		        }
          }
        }

        matrixb[oi]+=double(matrixbp1);
        matrixb[oi]=matrixb[oi]/dt;
      }
    }
  }
}

void PopulateMatrixB(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,const float4 *velrhop,double3 *dwxcorrg,double3 *dwzcorrg,double *matrixb
  ,const unsigned *porder,const unsigned *idp,const double dt,const unsigned ppedim,const double *divr,const word *code,const double freesurface){

  const unsigned npf=np-npb;
  int hdiv=(cellmode==CELLMODE_H? 2: 1);
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  unsigned cellfluid=nc.w*nc.z+1;
  int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);

    if(psimple){
      KerPopulateMatrixB<true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,matrixb,porder,idp,dt,ppedim,divr,code,freesurface);
      KerPopulateMatrixB<true> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,matrixb,porder,idp,dt,ppedim,divr,code,freesurface);
    }
    else{
      KerPopulateMatrixB<false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,matrixb,porder,idp,dt,ppedim,divr,code,freesurface);
      KerPopulateMatrixB<false> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,dwxcorrg,dwzcorrg,matrixb,porder,idp,dt,ppedim,divr,code,freesurface);
    }
  }
}

//==============================================================================
///Matrix Storage
//==============================================================================
template<bool psimple> __device__ void KerMatrixStorageCalc
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,double3 posdp1,float3 posp1,const word *code,unsigned &index)
{
  for(int p2=pini;p2<pfin;p2++)if(CODE_GetTypeValue(code[p2])==0||CODE_GetTypeValue(code[p2])==2){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<false> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      index++;
    }
  }
}

template<bool psimple> __global__ void KerMatrixStorage
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,double *divr,const word *code,const unsigned *porder,unsigned int *row,const double freesurface)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
      unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
      if(CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2){
        if(divr[p1]>freesurface){
        //-Obtiene datos basicos de particula p1.
  	    //-Obtains basic data of particle p1.
        double3 posdp1;
        float3 posp1,velp1;
        float rhopp1,pressp1;
        unsigned index=0;
        unsigned oi=porder[p1];
        KerGetParticleData<false>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    
        //-Obtiene limites de interaccion
	      //-Obtains interaction limits
        int cxini,cxfin,yini,yfin,zini,zfin;
        KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

        //-Interaccion con Fluidas.
	      //-Interaction with fluids.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
		          KerMatrixStorageCalc<false> (p1,pini,pfin,posxy,posz,pospress,posdp1,posp1,code,index);
            }
	        }
        }

        //-Interaccion con contorno.
	      //-Interaction with boundaries.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z;
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
              KerMatrixStorageCalc<false> (p1,pini,pfin,posxy,posz,pospress,posdp1,posp1,code,index);
		        }
          }
        }
        row[oi]+=index;
      }
    }
  }
}

void MatrixStorage(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop
  ,const word *code,const unsigned *idp,double *divr,const unsigned *porder,unsigned int *row,const double freesurface){
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(psimple){
      KerMatrixStorage<true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code,porder,row,freesurface);
      KerMatrixStorage<true> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code,porder,row,freesurface);
    }
    else{
      KerMatrixStorage<false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code,porder,row,freesurface);
      KerMatrixStorage<false> <<<sgridb,bsbound>>> (npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,pospress,velrhop,divr,code,porder,row,freesurface);
    }
  }
}

//==============================================================================
///Matrix A Population
//==============================================================================
template<bool psimple> __device__ void KerMatrixAFluid
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,double3 posdp1,float3 posp1,const double massp2,const double RhopZero,const word *code,unsigned &index,unsigned int *col,double *matrixInd,const int diag,const unsigned *porder)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<false> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);

      const unsigned oj=porder[p2];
	    double volumep2=massp2/RhopZero; //Volume of particle j 

      double rDivW=drx*frx+dry*fry+drz*frz;
      double temp=2.0*rDivW/(RhopZero*(rr2+double(CTE.eta2)));

      matrixInd[index]=-temp*volumep2;
      col[index]=oj;
      matrixInd[diag]+=temp*volumep2;
      index++;
    }
  }
}

template<bool psimple> __device__ void KerMatrixABound
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,const float4 *pospress
  ,double3 posdp1,float3 posp1,const unsigned *idp,const double massp2,const double RhopZero,tdouble3 gravity,const word *code,unsigned int *col,unsigned int *row
  ,double *matrixInd,double *matrixb,const int diag,const unsigned *porder,const unsigned oi,const unsigned ppedim,const unsigned npb,const unsigned *irelationg)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float  pressp2;
    KerGetParticlesDr<false> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);

      const unsigned idp2=idp[p2];
      const unsigned mkp2=CODE_GetTypeValue(code[p2]);
      const unsigned oj=porder[p2];
	    const double volumep2=massp2/RhopZero; //Volume of particle j 

      const double rDivW=drx*frx+dry*fry+drz*frz;
      double temp=2.0*rDivW/(RhopZero*(rr2+double(CTE.eta2)));

      if(oi!=oj){
        for(unsigned pk=diag;pk<unsigned(row[oi+1]);pk++){
          if(col[pk]==ppedim){
            matrixInd[pk]=-temp*volumep2;
            col[pk]=oj;
            matrixInd[diag]+=temp*volumep2;
            break;
          }
          else if(col[pk]==oj){
            matrixInd[pk]-=temp*volumep2;
            matrixInd[diag]+=temp*volumep2;
            break;
          }
        }
      }

      if(mkp2==1){
        unsigned p2k;
        for(unsigned k=0;k<npb;k++) if(idp[k]==irelationg[idp2]){
          p2k=k;
          break;
        }

        double dist = posz[p2k]-posz[p2];
			  temp = temp * RhopZero * abs(gravity.z) * dist;
			  matrixb[oi]+=volumep2*temp; 
      }
    }
  }
}

template<bool psimple> __global__ void KerPopulateMatrixA
  (unsigned np,unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell,tdouble3 gravity
  ,const double2 *posxy,const double *posz,const float4 *pospress,const float4 *velrhop,const double *divr,const word *code
  ,const unsigned *idp,const unsigned *porder,unsigned int *row,unsigned int *col,double *matrixInd,double *matrixb,const unsigned ppedim,const unsigned npb,const unsigned *irelationg,const double freesurface)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    if((CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2)&&porder[p1]!=int(np)){
      unsigned oi=porder[p1];
      const unsigned diag=row[oi];
      col[diag]=oi;
      unsigned index=diag+1;
      if(divr[p1]>freesurface){
        //-Obtiene datos basicos de particula p1.
  	    //-Obtains basic data of particle p1.
        double3 posdp1;
        float3 posp1,velp1;
        float rhopp1,pressp1;
        KerGetParticleData<false>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    
        //-Obtiene limites de interaccion
	      //-Obtains interaction limits
        int cxini,cxfin,yini,yfin,zini,zfin;
        KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
        
        //-Interaccion con Fluidas.
	      //-Interaction with fluids.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
		          KerMatrixAFluid<false> (p1,pini,pfin,posxy,posz,pospress,posdp1,posp1,double(CTE.massf),double(CTE.rhopzero),code,index,col,matrixInd,diag,porder);
            }
	        }
        }

        //-Interaccion con contorno.
	      //-Interaction with boundaries.
        for(int z=zini;z<zfin;z++){
          int zmod=(nc.w)*z;
          for(int y=yini;y<yfin;y++){
            int ymod=zmod+nc.x*y;
            unsigned pini,pfin=0;
            for(int x=cxini;x<cxfin;x++){
              int2 cbeg=begincell[x+ymod];
              if(cbeg.y){
                if(!pfin)pini=cbeg.x;
                pfin=cbeg.y;
              }
            }
            if(pfin){
              KerMatrixABound<false> (p1,pini,pfin,posxy,posz,pospress,posdp1,posp1,idp,double(CTE.massb),double(CTE.rhopzero),gravity,code,col,row,matrixInd,matrixb,diag,porder,oi,ppedim,npb,irelationg);           
		        }
          }
        }
      }
      else matrixInd[diag]=1.0;
    }
  }
}

void PopulateMatrixA(bool psimple,TpCellMode cellmode
  ,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok
  ,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell,tdouble3 gravity,const double2 *posxy
  ,const double *posz,const float4 *pospress,const float4 *velrhop,double *matrixInd,double *matrixb
  ,unsigned int *row,unsigned int *col,const unsigned *porder,const unsigned *idp,const unsigned ppedim
  ,const double *divr,const word *code,const unsigned *irelationg,const double freesurface){
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(psimple){
      KerPopulateMatrixA<true> <<<sgridf,bsfluid>>> (np,npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,pospress,velrhop,divr,code,idp,porder,row,col,matrixInd,matrixb,ppedim,npb,irelationg,freesurface);
      KerPopulateMatrixA<true> <<<sgridb,bsbound>>> (np,npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,pospress,velrhop,divr,code,idp,porder,row,col,matrixInd,matrixb,ppedim,npb,irelationg,freesurface);
    }
    else{
      KerPopulateMatrixA<false> <<<sgridf,bsfluid>>> (np,npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,pospress,velrhop,divr,code,idp,porder,row,col,matrixInd,matrixb,ppedim,npb,irelationg,freesurface);
      KerPopulateMatrixA<false> <<<sgridb,bsbound>>> (np,npbok,0,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,pospress,velrhop,divr,code,idp,porder,row,col,matrixInd,matrixb,ppedim,npb,irelationg,freesurface);
    }
  }
}

//==============================================================================
/// Pressure Assign
//==============================================================================
//------------------------------------------------------------------------------
///Pressure Assign 
//------------------------------------------------------------------------------
template<bool psimple> __global__ void KerPressureAssignCode0
  (unsigned np,unsigned n,unsigned pinit,float4 *velrhop,const word *code,const unsigned *porder,double *press)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    if((CODE_GetTypeValue(code[p1])==0||CODE_GetTypeValue(code[p1])==2)&&porder[p1]!=np) velrhop[p1].w=float(press[porder[p1]]);
  }
}

template<bool psimple> __global__ void KerPressureAssignCode1
  (unsigned np,unsigned npb,unsigned npbok,unsigned pinit,const tfloat3 gravity,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,double *press,const unsigned *porder,const unsigned *idp,const word *code,const unsigned *irelationg,const double *divr)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npbok){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    if(CODE_GetTypeValue(code[p1])==1&&porder[p1]!=np){
      //-Obtiene datos basicos de particula p1.
  	  //-Obtains basic data of particle p1.
      double3 posdp1;
      float3 posp1,velp1;
      float rhopp1,pressp1;
      KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
      
      const unsigned j=irelationg[idp[p1]];
      if(j!=npb){
        unsigned p2k;
        for(int k=int(pinit);k<int(np);k++) if(idp[k]==j){
          p2k=k;
          break;
        }

        const double drz=posz[p2k]-posdp1.z;
        if(divr[p2k])velrhop[p1].w=float(press[porder[p1]]+CTE.rhopzero*fabs(gravity.z)*drz);
        else velrhop[p1].w=CTE.rhopzero*fabs(gravity.z)*drz;
      }
    }
  }
}

void PressureAssign(bool psimple,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok
  ,const tfloat3 gravity,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,double *press,const unsigned *porder,const unsigned *idp,const word *code,const unsigned *irelationg,const double *divr){
  const unsigned npf=np-npb;

  if(np){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npb,bsbound);
    if(psimple){
      KerPressureAssignCode0<true> <<<sgridf,bsfluid>>> (np,npf,npb,velrhop,code,porder,press);
      KerPressureAssignCode0<true> <<<sgridb,bsbound>>> (np,npbok,0,velrhop,code,porder,press);
      KerPressureAssignCode1<true> <<<sgridb,bsbound>>> (np,npb,npbok,0,gravity,posxy,posz,pospress,velrhop,press,porder,idp,code,irelationg,divr); 
    }
    else{
      KerPressureAssignCode0<false> <<<sgridf,bsfluid>>> (np,npf,npb,velrhop,code,porder,press);
      KerPressureAssignCode0<false> <<<sgridb,bsbound>>> (np,npbok,0,velrhop,code,porder,press);
      KerPressureAssignCode1<false> <<<sgridb,bsbound>>> (np,npb,npbok,0,gravity,posxy,posz,pospress,velrhop,press,porder,idp,code,irelationg,divr); 
    }
  }
}

//==============================================================================
/// Initialises array with the indicated value.
//==============================================================================
__global__ void KerInitArrayPOrder(unsigned n,unsigned *v,unsigned value)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la particula //-NI of the particle
  if(p<n)v[p]=value;
}

void InitArrayPOrder(unsigned n,unsigned *v, unsigned value){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerInitArrayPOrder <<<sgrid,SPHBSIZE>>> (n,v,value);
  }
}

__global__ void KerInitArrayCol(unsigned n,unsigned int *v,int value)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la particula //-NI of the particle
  if(p<n)v[p]=value;
}
void InitArrayCol(unsigned n,unsigned int *v,int value){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerInitArrayCol <<<sgrid,SPHBSIZE>>> (n,v,value);
  }
}

//==============================================================================
/// Solve matrix with ViennaCL
//==============================================================================

template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solverCPU(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,std::vector<double> &vectorx,const unsigned ppedim){ 
  VectorType result(rhs);
  VectorType residual(rhs);
  viennacl::tools::timer timer;
  timer.start();   
  result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  viennacl::backend::finish();    
  std::cout << "  > Solver time: " << timer.get() << std::endl;   
  residual -= viennacl::linalg::prod(matrix, result); 
  std::cout << "  > Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;  
  std::cout << "  > Iterations: " << solver.iters() << std::endl;

  copy(result,vectorx);
}

void solveViennaCPU(double *matrixa, double *matrixb, double *matrixx,unsigned int *row, unsigned int *col,const unsigned ppedim,const unsigned nnz){
    viennacl::context ctx;
   
    typedef double ScalarType;

    double *matrixaCPU; matrixaCPU=new double[nnz]; cudaMemcpy(matrixaCPU,matrixa,sizeof(double)*nnz,cudaMemcpyDeviceToHost);
    double *matrixbCPU; matrixbCPU=new double[ppedim]; cudaMemcpy(matrixbCPU,matrixb,sizeof(double)*ppedim,cudaMemcpyDeviceToHost);
    double *matrixxCPU; matrixxCPU=new double[ppedim]; cudaMemcpy(matrixxCPU,matrixx,sizeof(double)*ppedim,cudaMemcpyDeviceToHost);
    unsigned int *rowCPU; rowCPU=new unsigned int[ppedim+1]; cudaMemcpy(rowCPU,row,sizeof(unsigned int)*(ppedim+1),cudaMemcpyDeviceToHost);
    unsigned int *colCPU; colCPU=new unsigned int[nnz]; cudaMemcpy(colCPU,col,sizeof(unsigned int)*nnz,cudaMemcpyDeviceToHost);
    
    std::vector<double>vectora; vectora.resize(nnz); for(int i=0;i<nnz;i++) vectora[i]=matrixaCPU[i];
    std::vector<double>vectorb; vectorb.resize(ppedim); for(int i=0;i<ppedim;i++) vectorb[i]=matrixbCPU[i];
    std::vector<double>vectorx; vectorx.resize(ppedim); for(int i=0;i<ppedim;i++) vectorx[i]=matrixxCPU[i];
    std::vector<int>vectorrow; vectorrow.resize(ppedim+1); for(int i=0;i<(ppedim+1);i++) vectorrow[i]=rowCPU[i];
    std::vector<int>vectorcol; vectorcol.resize(nnz); for(int i=0;i<nnz;i++) vectorcol[i]=colCPU[i];

    viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ctx);
    vcl_compressed_matrix.set(&vectorrow[0],&vectorcol[0],&vectora[0],ppedim,ppedim,nnz);

    viennacl::vector<ScalarType> vcl_vec(vectorb.size(),ctx);
    
    copy(vectorb,vcl_vec);

    viennacl::linalg::bicgstab_tag bicgstab(1e-5,2000);

    std::cout<<"JACOBI PRECOND GPUCPU" <<std::endl;
    viennacl::vector<ScalarType> vcl_result(vcl_compressed_matrix.size1(),ctx);
    viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_compressed_matrix,viennacl::linalg::jacobi_tag());
    run_solverCPU(vcl_compressed_matrix,vcl_vec,bicgstab,vcl_jacobi,vectorx,ppedim);

    for(int i=0;i<ppedim;i++) matrixxCPU[i]=vectorx[i]; cudaMemcpy(matrixx,matrixxCPU,sizeof(double)*ppedim,cudaMemcpyHostToDevice);
    delete[] matrixaCPU; delete[] matrixbCPU; delete[] matrixxCPU; delete[] rowCPU; delete[] colCPU;
    vectora.clear(); vectorb.clear(); vectorx.clear(); vectorrow.clear();vectorcol.clear();
}    

template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag,typename ScalarType>
void run_solver(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,viennacl::vector<ScalarType> & vcl_result){ 
  VectorType result(rhs);
  VectorType residual(rhs);
  viennacl::tools::timer timer;
  timer.start();
  result = viennacl::linalg::solve(matrix, rhs, solver, precond);   
  viennacl::backend::finish();  
  std::cout << "  > Solver time: " << timer.get() << std::endl;   
  residual -= viennacl::linalg::prod(matrix, result); 
  std::cout << "  > Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;  
  std::cout << "  > Iterations: " << solver.iters() << std::endl;
  viennacl::copy(result,vcl_result);
}

template<typename ScalarType>
void run_amg(viennacl::linalg::bicgstab_tag & bicgstab_solver,viennacl::vector<ScalarType> & vcl_vec,viennacl::compressed_matrix<ScalarType> & matrix,
             std::string info,viennacl::linalg::amg_tag & amg_tag,viennacl::vector<ScalarType> & vcl_result){
  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg(matrix, amg_tag);
  std::cout << " * Setup phase (ViennaCL types)..." << std::endl;
  viennacl::tools::timer timer; 
  timer.start(); 
  vcl_amg.setup(); 
  //std::cout << "levels = " << vcl_amg.levels() << "\n";
  //for(int i =0; i< vcl_amg.levels();i++) std::cout << "level " << i << "\t" << "size = " << vcl_amg.size(i) << "\n";
  viennacl::backend::finish();
  std::cout << "  > Setup time: " << timer.get() << std::endl;
  std::cout << " * CG solver (ViennaCL types)..." << std::endl;
  run_solver(matrix,vcl_vec,bicgstab_solver,vcl_amg,vcl_result);
}

void solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,double *matrixa,double *matrixx,double *matrixb,unsigned int *row,unsigned int *col,const unsigned nnz,const unsigned ppedim){
  viennacl::context CudaCtx(viennacl::CUDA_MEMORY);
  typedef double       ScalarType;

  viennacl::compressed_matrix<ScalarType> vcl_A_cuda(row, col, matrixa, viennacl::CUDA_MEMORY, ppedim, ppedim, nnz);
  
  viennacl::vector<ScalarType> vcl_vec(matrixb, viennacl::CUDA_MEMORY, ppedim);
  viennacl::vector<ScalarType> vcl_result(matrixx, viennacl::CUDA_MEMORY, ppedim);

  viennacl::linalg::bicgstab_tag bicgstab(1e-5,2000);
  //viennacl::linalg::gmres_tag gmres(tolerance,iterations,20);

  //if(tprecond==PRECOND_Jacobi){
    std::cout<<"JACOBI PRECOND" <<std::endl;
    viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_A_cuda,viennacl::linalg::jacobi_tag());
    run_solver(vcl_A_cuda,vcl_vec,bicgstab,vcl_jacobi,vcl_result);
  /*}
  else if(tprecond==PRECOND_AMG){
    std::cout<<"AMG PRECOND"<<std::endl;

    viennacl::linalg::amg_tag amg_tag_agg_pmis;
    amg_tag_agg_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION); std::cout<<"COARSENING: MIS2 AGGREGATION"<<std::endl;
    if(tamginter==AMGINTER_AG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION); std::cout<<"INTERPOLATION: AGGREGATION "<<std::endl; }
    else if(tamginter==AMGINTER_SAG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION); std::cout<<"INTERPOLATION: SMOOTHED AGGREGATION"<<std::endl; }
    
    amg_tag_agg_pmis.set_strong_connection_threshold(strongconnection);
    amg_tag_agg_pmis.set_jacobi_weight(jacobiweight);
    amg_tag_agg_pmis.set_presmooth_steps(presmooth);
    amg_tag_agg_pmis.set_postsmooth_steps(postsmooth); 
    amg_tag_agg_pmis.set_coarsening_cutoff(coarsecutoff); 
    run_amg(bicgstab,vcl_vec,vcl_A_cuda,"MIS2 AGGREGATION COARSENING, AGGREGATION INTERPOLATION",amg_tag_agg_pmis,vcl_result);
  }*/
}

//------------------------------------------------------------------------------
/// Shifting
//------------------------------------------------------------------------------
template<bool psimple,TpFtMode ftmode> __device__ void KerInteractionForcesShifting2
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco,const float *ftomassp
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop,const word *code
  ,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,float3 posp1,float3 velp1,float pressp1,float rhopp1
  ,TpShifting tshifting,float3 &shiftposp1,float Wab1,const float tensilen, const float tensiler,float &divrp1)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    float pressp2;
    KerGetParticlesDr<psimple> (p2,posxy,posz,pospress,posdp1,posp1,drx,dry,drz,pressp2);
    double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
      //-Wendland kernel.
      double frx,fry,frz;
      KerGetKernel(rr2,drx,dry,drz,frx,fry,frz);
	  
      //-Obtiene masa de particula p2 en caso de existir floatings.
	  //-Obtains mass of particle p2 if any floating bodies exist.
      bool ftp2;         //-Indica si es floating. //-indicates if it is floating.
      float ftmassp2;    //-Contiene masa de particula floating o massp2 si es bound o fluid. //-Contains mass of floating body or massf if fluid.
     // bool compute=true; //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
      if(USE_FLOATING){
        const word cod=code[p2];
        ftp2=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        //if(ftp2 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
        if(ftp2 && tshifting==SHIFT_NoBound)shiftposp1.x=FLT_MAX; //-Con floatings anula shifting. //-Cancels shifting with floating bodies
        //compute=!(USE_DEM && ftp1 && (boundp2 || ftp2)); //-Se desactiva cuando se usa DEM y es float-float o float-bound. //-Deactivated when DEM is used and is float-float or float-bound.
      }

      //-Shifting correction
      const float massrhop=(USE_FLOATING? ftmassp2: massp2)/CTE.rhopzero;
      const float tensile=tensilen*powf(KerGetKernelWab(rr2)/Wab1,tensiler);
        //const bool noshift=(boundp2 && (tshifting==SHIFT_NoBound || (tshifting==SHIFT_NoFixed && CODE_GetType(code[p2])==CODE_TYPE_FIXED)));
        shiftposp1.x+=massrhop*(1.0f+tensile)*frx; 
        shiftposp1.y+=massrhop*(1.0f+tensile)*fry;
        shiftposp1.z+=massrhop*(1.0f+tensile)*frz;
        divrp1-=massrhop*(drx*frx+dry*fry+drz*frz);
    }
  }
}

template<bool psimple,TpFtMode ftmode> __global__ void KerInteractionForcesShifting1
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,float viscob,float viscof
  ,const int2 *begincell,int3 cellzero,const unsigned *dcell,const float *ftomassp
  ,const double2 *posxy,const double *posz,const float4 *pospress,float4 *velrhop,const word *code
  ,TpShifting tshifting,float3 *shiftpos,double *divr,const float tensilen,const float tensiler)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    //float visc=0,arp1=0,deltap1=0;
   // float3 acep1=make_float3(0,0,0);
     const float Wab1=KerGetKernelWab(CTE.dp*CTE.dp);
    //-Vars para Shifting.
	//-Variables for Shifting.
    float3 shiftposp1=make_float3(0,0,0);
    float  divrp1=0;
   /* if(shift){
      shiftposp1=make_float3(0,0,0);
      shiftdetectp1=0;
    }*/

    //-Obtiene datos de particula p1 en caso de existir floatings.
	//-Obtains data of particle p1 in case there are floating bodies.
    bool ftp1;       //-Indica si es floating. //-Indicates if it is floating.
    float ftmassp1;  //-Contiene masa de particula floating o 1.0f si es fluid. //-Contains floating particle mass or 1.0f if it is fluid.
    /*if(USE_FLOATING){
      const word cod=code[p1];
      ftp1=(CODE_GetType(cod)==CODE_TYPE_FLOATING);
      ftmassp1=(ftp1? ftomassp[CODE_GetTypeValue(cod)]: 1.f);
      //if(ftp1 && (tdelta==DELTA_Dynamic || tdelta==DELTA_DynamicExt))deltap1=FLT_MAX;
      //if(ftp1 && shift)shiftposp1.x=FLT_MAX;  //-Para floatings no se calcula shifting. //-Shifting is not calculated for floating bodies.
    }*/

    //-Obtiene datos basicos de particula p1.
	//-Obtains basic data of particle p1.
    double3 posdp1;
    float3 posp1,velp1;
    float rhopp1,pressp1;
    KerGetParticleData<psimple>(p1,posxy,posz,pospress,velrhop,velp1,rhopp1,posdp1,posp1,pressp1);
    pressp1=velrhop[p1].w;
    //-Vars para Laminar+SPS
	//-Variables for Laminar+SPS
   // float2 taup1_xx_xy,taup1_xz_yy,taup1_yz_zz;
    /*if(lamsps){
      taup1_xx_xy=tauff[p1*3];
      taup1_xz_yy=tauff[p1*3+1];
      taup1_yz_zz=tauff[p1*3+2];
    }*/
    //-Vars para Laminar+SPS (calculo).
	//-Variables for Laminar+SPS (computation).
   // float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
    /*if(lamsps){
      grap1_xx_xy=make_float2(0,0);
      grap1_xz_yy=make_float2(0,0);
      grap1_yz_zz=make_float2(0,0);
    }*/

    //-Obtiene limites de interaccion
	//-Obtains interaction limits
    int cxini,cxfin,yini,yfin,zini,zfin;
    KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);

    //-Interaccion con Fluidas.
	//-Interaction with fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid; //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin){
		      KerInteractionForcesShifting2<psimple,ftmode> (false,p1,pini,pfin,viscob,ftomassp,posxy,posz,pospress,velrhop,code,CTE.massf,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,tshifting,shiftposp1,Wab1,tensilen,tensiler,divrp1);
        }
	    }
    }

    if(shiftposp1.x||shiftposp1.y||shiftposp1.z||divrp1){
      float3 s=shiftpos[p1];
      s.x+=shiftposp1.x; s.y+=shiftposp1.y; s.z+=shiftposp1.z;
      shiftpos[p1]=s;
      shiftposp1=make_float3(0,0,0);
      divr[p1]+=divrp1;
      divrp1=0;
    }

    //-Interaccion con contorno.
	//-Interaction with boundaries.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin){
		      KerInteractionForcesShifting2<psimple,ftmode> (true,p1,pini,pfin,viscof,ftomassp,posxy,posz,pospress,velrhop,code,CTE.massf,ftmassp1,ftp1,posdp1,posp1,velp1,pressp1,rhopp1,tshifting,shiftposp1,Wab1,tensilen,tensiler,divrp1);
        }
      }
    }
    
    if(shiftposp1.x||shiftposp1.y||shiftposp1.z||divrp1){
      float3 s=shiftpos[p1];
      s.x+=shiftposp1.x; s.y+=shiftposp1.y; s.z+=shiftposp1.z;
      shiftpos[p1]=s;
      divr[p1]+=divrp1;
    }
  }
}

void Interaction_Shifting
  (bool psimple,bool floating,bool usedem,TpCellMode cellmode,float viscob,float viscof,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,const float4 *pospress
  ,float4 *velrhop,const word *code,const float *ftomassp
  ,TpShifting tshifting,float3 *shiftpos,double *divr,const float tensilen,const float tensiler)
{
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);

    if(psimple){ const bool psimple=true;
      if(!floating)   KerInteractionForcesShifting1<psimple,FTMODE_None> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
      else if(!usedem)KerInteractionForcesShifting1<psimple,FTMODE_Sph> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
      else            KerInteractionForcesShifting1<psimple,FTMODE_Dem> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
    }
    else{ const bool psimple=false;
      if(!floating)   KerInteractionForcesShifting1<psimple,FTMODE_None> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
      else if(!usedem)KerInteractionForcesShifting1<psimple,FTMODE_Sph> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
      else            KerInteractionForcesShifting1<psimple,FTMODE_Dem> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,pospress,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler);
    }
  }
}

template<bool floating> __global__ void KerComputeShift
  (unsigned npf,unsigned npb,const float3 *shiftpos,word *code,double2 *movxy,double *movz)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
    unsigned p1=p+npb;
    const float3 rshiftpos=shiftpos[p1];
    if(!floating || CODE_GetType(code[p1])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
      double dx=double(rshiftpos.x);
      double dy=double(rshiftpos.y);
      double dz=double(rshiftpos.z);
      /*if(shift){
        const float3 rshiftpos=shiftpos[p];
        dx+=double(rshiftpos.x);
        dy+=double(rshiftpos.y);
        dz+=double(rshiftpos.z);
      }*/
      movxy[p1]=make_double2(dx,dy);
      movz[p1]=dz;
    }
  }
}

void ComputeShift(bool floating,const unsigned bsfluid,unsigned np,unsigned npb,const float3 *shiftpos
  ,word *code,double2 *movxy,double *movz)
{
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    if(floating)KerComputeShift<true>  <<<sgridf,bsfluid>>> (npf,npb,shiftpos,code,movxy,movz);
    else        KerComputeShift<false> <<<sgridf,bsfluid>>> (npf,npb,shiftpos,code,movxy,movz);
  }
}
}