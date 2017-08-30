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
#include "JBlockSizeAuto.h"
#include "JLog2.h"
#include <cfloat>
#include <math_constants.h>
//#include "JDgKerPrint.h"
//#include "JDgKerPrint_ker.h"

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#pragma warning(disable : 4503) //Cancels "warning C4503: decorated name length exceeded, name was truncated"
#include <cuda_runtime.h>
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
	float resf;
  if(1){
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
	  
		if(ndata>1)cudaMemcpy(&resf,res,sizeof(float),cudaMemcpyDeviceToHost);
		else cudaMemcpy(&resf,data,sizeof(float),cudaMemcpyDeviceToHost);
  }

	//else{//-Using Thrust library is slower than ReduMasFloat() with ndata < 5M.
  //  thrust::device_ptr<float> dev_ptr(data);
  //  resf=thrust::reduce(dev_ptr,dev_ptr+ndata,-FLT_MAX,thrust::maximum<float>());
  //}
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
__global__ void KerResetVely(unsigned npb,unsigned npf,float4 *v)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf)v[p+npb].y=0;
}

//==============================================================================
// Pone v[].y a cero.
/// Sets v[].y to zero.
//==============================================================================
void ResetVely(unsigned npb,unsigned npf,float4 *v){
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerResetVely <<<sgrid,SPHBSIZE>>> (npb,npf,v);
  }
}

//------------------------------------------------------------------------------
/// Pone v[].y a cero.
/// Sets v[].y to zero.
//------------------------------------------------------------------------------
__global__ void KerResetAcey(unsigned npf,double3 *v)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf)v[p].y=0;
}

//==============================================================================
// Pone v[].y a cero.
/// Sets v[].y to zero.
//==============================================================================
void ResetAcey(unsigned npf,double3 *v){
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerResetAcey <<<sgrid,SPHBSIZE>>> (npf,v);
  }
}

__global__ void KerResetBoundVel(unsigned n,double3 *vel,double3 *velpre)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
		vel[p].x=velpre[p].x;
		vel[p].y=velpre[p].y;
		vel[p].z=velpre[p].z;
	}
}
void ResetBoundVel(const unsigned npbok,const unsigned bsbound,double3 *vel,double3 *velpre){
  if(npbok){
    dim3 sgridb=GetGridSize(npbok,bsbound);
    KerResetBoundVel <<<sgridb,bsbound>>> (npbok,vel,velpre);
  }
}

__global__ void KerResetrowIndg(const unsigned npplus,unsigned *row,const unsigned npb)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npplus){
		row[p]=npb;
	}
}

void ResetrowIndg(const unsigned npplus,unsigned *row,const unsigned npb){
	if(npplus){
    dim3 sgrid=GetGridSize(npplus,SPHBSIZE);
    KerResetrowIndg <<<sgrid,SPHBSIZE>>> (npplus,row,npb);
  }
}
//------------------------------------------------------------------------------
/// Calculates module^2 of ace.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const float3 *ace,float *acemod)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const float3 r=ace[p];
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace.
//==============================================================================
void ComputeAceMod(unsigned n,const float3 *ace,float *acemod){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerComputeAceMod <<<sgrid,SPHBSIZE>>> (n,ace,acemod);
  }
}

//------------------------------------------------------------------------------
/// Calculates module^2 of ace, comprobando que la particula sea normal.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const word *code,const float3 *ace,float *acemod)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const float3 r=(CODE_GetSpecialValue(code[p])==CODE_NORMAL? ace[p]: make_float3(0,0,0));
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace, comprobando que la particula sea normal.
//==============================================================================
void ComputeAceMod(unsigned n,const word *code,const float3 *ace,float *acemod){
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
//# Kernels auxiliares para interaccion.
//# Auxiliary kernels for the interaction.
//##############################################################################
//------------------------------------------------------------------------------
/// Devuelve posicion y vel de particula.
/// Returns postion and vel of a particle.
//------------------------------------------------------------------------------
/*__device__ void KerGetParticleData(unsigned p1
  ,const double2 *posxy,const double *posz,const float4 *velrhop
  ,float3 &velp1,double3 &posdp1)
{
  float4 r=velrhop[p1];
  velp1=make_float3(r.x,r.y,r.z);

  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
}*/


__device__ void KerGetParticleData(unsigned p1
  ,const double2 *posxy,const double *posz,const double3 *velrhop
  ,double3 &velp1,double3 &posdp1)
{
  double3 r=velrhop[p1];
  velp1=make_double3(r.x,r.y,r.z);

  double2 pxy=posxy[p1];
  posdp1=make_double3(pxy.x,pxy.y,posz[p1]);
}
//------------------------------------------------------------------------------
/// Devuelve drx, dry y drz entre dos particulas.
/// Returns drx, dry and drz between the particles.
//------------------------------------------------------------------------------
__device__ void KerGetParticlesDr(unsigned count,int p2
  ,const double2 *posxy,const double *posz,bool &interact,const double3 &posdp1,double3 fluidVel,double fluidPress
  ,double &drx,double &dry,double &drz,double3 &velp2,double &pressp2,const double PistonPos,const double PistonVel,double &NeumannDist,double RightWall,tfloat3 gravity)
{
	double dp05=0.5*CTE.dp;
	double LeftWall=PistonPos; double LeftWallVel=PistonVel;
	double Bottom=dp05;
	double3 posp2=make_double3(posxy[p2].x,posxy[p2].y,posz[p2]);
	if(count==0){//Fluid particle p2
		interact=true;
		drx=posdp1.x-posp2.x;
		dry=0;
		drz=posdp1.z-posp2.z;
		velp2=fluidVel;
		pressp2=fluidPress;
	}
	else if(count==1){//Boundary LeftWall
		double dist=posp2.x-LeftWall;
		if(dist*dist<=CTE.fourh2){
			interact=true;
			posp2.x=2.0*LeftWall-posp2.x;
			drx=posdp1.x-posp2.x;
			dry=0;
			drz=posdp1.z-posp2.z;
			velp2.x=2.0*LeftWallVel-fluidVel.x;
			velp2.z=fluidVel.z;
			pressp2=fluidPress;
		}
	}
	else if(count==2){//Boundary Bottom
		double dist=posp2.z-Bottom;
		if(dist*dist<=CTE.fourh2){
			interact=true;
			NeumannDist=-2.0*dist;
			posp2.z=2.0*Bottom-posp2.z;
			drx=posdp1.x-posp2.x;
			dry=0;
			drz=posdp1.z-posp2.z;
			velp2.x=fluidVel.x;
			velp2.z=-fluidVel.z;
			pressp2=fluidPress+NeumannDist*gravity.z*CTE.rhopzero;
		}
	}
	else if(count==3){//Boundary RightWall
		double dist=posp2.x-RightWall;
		if(dist*dist<=CTE.fourh2){
			interact=true;
			posp2.x=2.0*RightWall-posp2.x;
			drx=posdp1.x-posp2.x;
			dry=0;
			drz=posdp1.z-posp2.z;
			velp2.x=-fluidVel.x;
			velp2.z=fluidVel.z;
			pressp2=fluidPress;
		}
	}
	else if(count==4){//Boundary BottomLeft
		double distx=posp2.x-LeftWall;
		double distz=posp2.z-Bottom;
		double dist=distx*distx+distz*distz;
		if(dist<=CTE.fourh2){
			interact=true;
			NeumannDist=-2.0*distz;
			posp2.x=2.0*LeftWall-posp2.x;
			posp2.z=2.0*Bottom-posp2.z;
			drx=posdp1.x-posp2.x;
			dry=0;
			drz=posdp1.z-posp2.z;
			velp2.x=-fluidVel.x;
			velp2.z=-fluidVel.z;
			pressp2=fluidPress+NeumannDist*gravity.z*CTE.rhopzero;
		}
	}
	else if(count==5){//Boundary BottomRight
		double distx=posp2.x-RightWall;
		double distz=posp2.z-Bottom;
		double dist=distx*distx+distz*distz;
		if(dist<=CTE.fourh2){
			interact=true;
			NeumannDist=-2.0*distz;
			posp2.x=2.0*RightWall-posp2.x;
			posp2.z=2.0*Bottom-posp2.z;
			drx=posdp1.x-posp2.x;
			dry=0;
			drz=posdp1.z-posp2.z;
			velp2.x=-fluidVel.x;
			velp2.z=-fluidVel.z;
			pressp2=fluidPress+NeumannDist*gravity.z*CTE.rhopzero;
		}
	}
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
/*__device__ void KerGetKernelQuintic(float rr2,float drx,float dry,float drz
  ,float &frx,float &fry,float &frz)
{
  const float rad=sqrt(rr2);
  const float qq=rad/CTE.h;
  const float Bwen=CTE.bwen;

	//-Quintic Spline
  float fac;
  if(qq<1.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f)-75.0f*powf(1.0f-qq,4.0f));
  else if(qq<2.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f)+30.0f*powf(2.0f-qq,4.0f));
  else if(qq<3.0f)fac=Bwen*(-5.0f*powf(3.0f-qq,4.0f));
  else fac=0;
  fac=fac/rad;

  frx=fac*drx; fry=fac*dry; frz=fac*drz;

}*/

__device__ void KerGetKernelQuintic(double rr2,double drx,double dry,double drz
  ,double &frx,double &fry,double &frz)
{
  const double rad=sqrt(rr2);
	const double h=CTE.h;
  const double qq=rad/h;
  const double Bwen=7.0/(478.0*PI*h*h*h);

	//-Quintic Spline
  double fac;
  if(qq<1.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0)+30.0*pow(2.0-qq,4.0)-75.0*pow(1.0-qq,4.0));
  else if(qq<2.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0)+30.0*pow(2.0-qq,4.0));
  else if(qq<3.0)fac=Bwen*(-5.0*pow(3.0-qq,4.0));
  else fac=0;
  fac=fac/rad;

  frx=fac*drx; fry=fac*dry; frz=fac*drz;

}

__device__ void KerGetKernelWendland(float rr2,float drx,float dry,float drz
  ,float &frx,float &fry,float &frz)
{
  const float rad=sqrt(rr2);
  const float qq=rad/CTE.h;

  //-Wendland kernel.
  const float wqq1=1.f-0.5f*qq;
  const float fac=CTE.bwen*qq*wqq1*wqq1*wqq1/rad;

  frx=fac*drx; fry=fac*dry; frz=fac*drz;
}

//------------------------------------------------------------------------------
/// Devuelve valores de kernel: wab.
/// returns kernel values: wab.
//------------------------------------------------------------------------------
/*__device__ float KerGetKernelQuinticWab(float rr2)
{
  const float rad=sqrt(rr2);
  const float qq=rad/CTE.h;
	const float Awen=CTE.awen;

	//-Quintic Spline
  float wab;
  if(qq<1.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f)+15.0f*powf(1.0f-qq,5.0f));
  else if(qq<2.0f)wab=Awen*(powf(3.0f-qq,5.0f)-6.0f*powf(2.0f-qq,5.0f));
  else if(qq<3.0f)wab=Awen*(powf(3.0f-qq,5.0f));
  else wab=0;
  return(wab);
}*/

__device__ double KerGetKernelQuinticWab(double rr2)
{
  const double rad=sqrt(rr2);
  const double qq=rad/CTE.h;
	const double h=CTE.h;
	const double Awen=7.0/(478.0*PI*h*h);

	//-Quintic Spline
  double wab;
  if(qq<1.0)wab=Awen*(pow(3.0-qq,5.0)-6.0*pow(2.0-qq,5.0)+15.0*pow(1.0-qq,5.0));
  else if(qq<2.0)wab=Awen*(pow(3.0-qq,5.0)-6.0*pow(2.0-qq,5.0));
  else if(qq<3.0)wab=Awen*(pow(3.0-qq,5.0));
  else wab=0;
  return(wab);
}

__device__ float KerGetKernelWendlandWab(float rr2)
{
  const float rad=sqrt(rr2);
  const float qq=rad/CTE.h;

  //-Wendland kernel.
  const float wqq=2.f*qq+1.f;
  const float wqq1=1.f-0.5f*qq;
  const float wqq2=wqq1*wqq1;

  return(CTE.awen*wqq*wqq2*wqq2);
}

//------------------------------------------------------------------------------
///  Inverse Kernel Correction Matrix
//------------------------------------------------------------------------------

__global__ void KerInverseKernelCor2D(unsigned n,unsigned pinit,double3 *dwxcorrg,double3 *dwzcorrg,const word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
		double3 dwx; dwx.x=dwxcorrg[p].x; dwx.z=dwxcorrg[p].z;
		double3 dwz; dwz.x=dwzcorrg[p].x; dwz.z=dwzcorrg[p].z;
		const double det=1.0/(dwx.x*dwz.z-dwz.x*dwx.z);
	
    if(det){
      dwxcorrg[p].x=dwz.z*det;
	    dwxcorrg[p].z=-dwx.z*det; 
	    dwzcorrg[p].x=-dwz.x*det;
	    dwzcorrg[p].z=dwx.x*det;
	  }
  }
}

__global__ void KerInverseKernelCor3D(unsigned n,unsigned pinit,float3 *dwxcorrg,double3 *dwycorrg,float3 *dwzcorrg,const word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
		double3 dwx; dwx.x=dwxcorrg[p].x; dwx.y=dwxcorrg[p].y; dwx.z=dwxcorrg[p].z; //  dwx.x   dwx.y   dwx.z
		double3 dwy; dwy.x=dwycorrg[p].x; dwy.y=dwycorrg[p].y; dwy.z=dwycorrg[p].z; //  dwy.x   dwy.y   dwy.z
		double3 dwz; dwz.x=dwzcorrg[p].x; dwz.y=dwzcorrg[p].y; dwz.z=dwzcorrg[p].z; //  dwz.x   dwz.y   dwz.z

		const double det=(dwx.x*dwy.y*dwz.z+dwx.y*dwy.z*dwz.x+dwy.x*dwz.y*dwx.z)-(dwz.x*dwy.y*dwx.z+dwy.x*dwx.y*dwz.z+dwy.z*dwz.y*dwx.x);
		dwxcorrg[p].x=float((dwy.y*dwz.z-dwy.z*dwz.y)/det);
		dwxcorrg[p].y=-float((dwx.y*dwz.z-dwx.z*dwz.y)/det);
		dwxcorrg[p].z=float((dwx.y*dwy.z-dwx.z*dwy.y)/det);
		dwycorrg[p].x=-float((dwy.x*dwz.z-dwy.z*dwz.x)/det);
		dwycorrg[p].y=float((dwx.x*dwz.z-dwx.z*dwz.x)/det);
		dwycorrg[p].z=-float((dwx.x*dwy.z-dwx.z*dwy.x)/det);
		dwzcorrg[p].x=float((dwy.x*dwz.y-dwy.y*dwz.x)/det);
		dwzcorrg[p].y=-float((dwx.x*dwz.y-dwx.y*dwz.x)/det);
		dwzcorrg[p].z=float((dwx.x*dwy.y-dwx.y*dwy.x)/det);
  }
}

template<TpKernel tker,TpFtMode ftmode> __device__ void KerInteractionForcesFluidVisc
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco
  ,const float *ftomassp,const double2 *posxy,const double *posz,const double3 *velrhop,const word *code,const unsigned *idp
  ,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,double3 velp1,double pressp1,double3 &acep1,float &divrp1,double3 &dwxp1,double3 &dwyp1,double3 &dwzp1,unsigned &rowCount,const float *divr,const float boundaryfs,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall,tfloat3 gravity)
{
	const float volume=massp2/CTE.rhopzero; //Volume of particle j 
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
		bool row=false;
		for(int count=0;count<=5;count++){
			bool interact=false;
			double3 velp2;
			double pressp2;
			double NeumannDist=0;
			KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
			if(interact){
				double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
					row=true;
					//-Wendland kernel.
					double frx,fry,frz;
					if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
																					//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
      
				//===== Aceleration ===== 
					const double rDivW=drx*frx+dry*fry+drz*frz;//R.Div(W)
					const double temp=volume*2.0f*visco*rDivW/(rr2+CTE.eta2);
					const double dvx=velp1.x-velp2.x, dvy=velp1.y-velp2.y, dvz=velp1.z-velp2.z;
					acep1.x+=temp*dvx; acep1.y+=temp*dvy; acep1.z+=temp*dvz;

					divrp1-=volume*rDivW;

					dwxp1.x-=volume*frx*drx; dwxp1.y-=volume*frx*dry; dwxp1.z-=volume*frx*drz;
					dwyp1.x-=volume*fry*drx; dwyp1.y-=volume*fry*dry; dwyp1.z-=volume*fry*drz;
					dwzp1.x-=volume*frz*drx; dwzp1.y-=volume*frz*dry; dwzp1.z-=volume*frz*drz;
				}
			}
		}
		if(row&&p1!=p2)rowCount++;
  }
}

//------------------------------------------------------------------------------
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode> __device__ void KerInteractionForcesFluidPresGrad
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco
  ,const float *ftomassp,const double2 *posxy,const double *posz,const double3 *velrhop,const word *code,const unsigned *idp
  ,double3 dwxcorrg,double3 dwycorrg,double3 dwzcorrg,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,double3 velp1,double pressp1
  ,double3 &acep1,unsigned *row,float &nearestBound,const float *divr,const float boundaryfs,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall,tfloat3 gravity)
{
	const float volumep2=massp2/CTE.rhopzero; //Volume of particle j
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
		for(int count=0;count<=5;count++){
			bool interact=false;
			double3 velp2;
			double pressp2;
			double NeumannDist=0;
			KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
			if(interact){
				double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
					//-Wendland kernel.
					double frx,fry,frz;
					if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
					//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);

					//===== Aceleration ===== 
					const double temp_x=frx*dwxcorrg.x+fry*dwycorrg.x+frz*dwzcorrg.x;
					//const double temp_y=frx*dwxcorrg.y+fry*dwycorrg.y+frz*dwzcorrg.y;
					const double temp_z=frx*dwxcorrg.z+fry*dwycorrg.z+frz*dwzcorrg.z;
					const double temp=volumep2*(pressp2-pressp1);

					acep1.x+=temp*temp_x;	/*acep1.y+=temp*temp_y;*/	acep1.z+=temp*temp_z;

				}
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
template<TpKernel tker,TpFtMode ftmode> __global__ void KerInteractionForcesFluid
  (TpInter tinter,unsigned npf,unsigned npb,int hdiv,uint4 nc,unsigned cellfluid,float viscob,float viscof
  ,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const float *ftomassp,const double2 *posxy,const double *posz,const double3 *velrhop,const word *code,const unsigned *idp
  ,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg,double3 *ace,float *divr,unsigned *row,const unsigned matOrder,const float boundaryfs,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall,tfloat3 gravity)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf){
    unsigned p1=p+npb;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
		float divrp1=0.0;
    double3 acep1=make_double3(0,0,0);
		double3 dwxp1=make_double3(0,0,0); double3 dwyp1=make_double3(0,0,0); double3 dwzp1=make_double3(0,0,0);
		unsigned rowCount=0; 
		float nearestBound=float(CTE.dp*CTE.dp);
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
    double3 velp1;
    //const float pressp1=velrhop[p1].w;
		const double pressp1=pressure[p1];
    KerGetParticleData(p1,posxy,posz,velrhop,velp1,posdp1);
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
					if(tinter==1)KerInteractionForcesFluidVisc<tker,ftmode> (false,p1,pini,pfin,viscof,ftomassp,posxy,posz,velrhop,code,idp,CTE.massf,ftmassp1,ftp1,posdp1,velp1,pressp1,acep1,divrp1,dwxp1,dwyp1,dwzp1,rowCount,divr,boundaryfs,pressure,PistonPos,PistonVel,RightWall,gravity);
					else if(tinter==2) KerInteractionForcesFluidPresGrad<tker,ftmode> (false,p1,pini,pfin,viscof,ftomassp,posxy,posz,velrhop,code,idp,dwxcorrg[Correctp1],dwycorrg[Correctp1],dwzcorrg[Correctp1],CTE.massf,ftmassp1,ftp1,posdp1,velp1,pressp1,acep1,row,nearestBound,divr,boundaryfs,pressure,PistonPos,PistonVel,RightWall,gravity);
				}
			}
    }

    if(acep1.x||acep1.y||acep1.z||divrp1
			||dwxp1.x||dwxp1.y||dwxp1.z
			||dwyp1.x||dwyp1.y||dwyp1.z
			||dwzp1.x||dwzp1.y||dwzp1.z||rowCount){
      double3 r=ace[Correctp1]; 
      if(tinter==1){ 
				r.x=acep1.x; r.y=acep1.y; r.z=acep1.z;
				
				divr[p1]=divrp1;
				row[p1-matOrder]=rowCount;
				dwxcorrg[Correctp1].x=dwxp1.x; dwxcorrg[Correctp1].y=dwxp1.y; dwxcorrg[Correctp1].z=dwxp1.z; 
				dwycorrg[Correctp1].x=dwyp1.x; dwycorrg[Correctp1].y=dwyp1.y; dwycorrg[Correctp1].z=dwyp1.z; 
				dwzcorrg[Correctp1].x=dwzp1.x; dwzcorrg[Correctp1].y=dwzp1.y; dwzcorrg[Correctp1].z=dwzp1.z; 
			} 

	    if(tinter==2){ const float rho0=CTE.rhopzero; r.x+=(acep1.x/rho0); r.y+=(acep1.y/rho0); r.z+=(acep1.z/rho0);}
      ace[Correctp1]=r;
    }
  }
}

template<TpKernel tker,TpFtMode ftmode> __device__ void KerViscousSchwaigerCalc
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,const float *ftomassp,const double2 *posxy,const double *posz
	,const double3 *velrhop,const word *code,const unsigned *idp,float massp2,float ftmassp1,bool ftp1,double3 posdp1,double3 velp1,double3 &sumfr
	,double3 &dud,double3 &dvd,double3 &dwd,double3 dwxp1,double3 dwyp1,double3 dwzp1,const float *divr,const float boundaryfs,double3 &taop1,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall,tfloat3 gravity)
{
	const float volume=massp2/CTE.rhopzero; //Volume of particle j 

  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    for(int count=0;count<=5;count++){
			bool interact=false;
			double3 velp2;
			double pressp2;
			double NeumannDist=0;
			KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
			if(interact){
				double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
					//-Wendland kernel.
					double frx,fry,frz;
					if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
					//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
					const double rDivW=drx*frx/*+dry*fry*/+drz*frz;//R.Div(W)
					const double taotemp=volume*rDivW/(rr2+CTE.eta2);
					taop1.x+=taotemp*drx*drx; taop1.z+=taotemp*drz*drz;
					sumfr.x+=volume*frx; sumfr.y+=volume*fry; sumfr.z+=volume*frz;
			
					//const double3 velrhop2=velrhop[p2];
					const double dvx=velp1.x-velp2.x, dvy=velp1.y-velp2.y, dvz=velp1.z-velp2.z;

					double temp_x=frx*dwxp1.x+fry*dwyp1.x+frz*dwzp1.x; temp_x=temp_x*volume;
					double temp_y=frx*dwxp1.y+fry*dwyp1.y+frz*dwzp1.y; temp_y=temp_y*volume;
					double temp_z=frx*dwxp1.z+fry*dwyp1.z+frz*dwzp1.z; temp_z=temp_z*volume;
			
					dud.x+=dvx*temp_x; dud.y+=dvx*temp_y; dud.z+=dvx*temp_z;
					dvd.x+=dvy*temp_x; dvd.y+=dvy*temp_y; dvd.z+=dvy*temp_z;
					dwd.x+=dvz*temp_x; dwd.y+=dvz*temp_y; dwd.z+=dvz*temp_z;
				}
			}
		}
	}
}

template<TpKernel tker,TpFtMode ftmode> __global__ void KerViscousSchwaiger
  (unsigned npf,unsigned npb,int hdiv,uint4 nc,unsigned cellfluid,float viscob,float viscof,const int2 *begincell
	,int3 cellzero,const unsigned *dcell,const float *ftomassp,const double2 *posxy,const double *posz
	,const double3 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg
	,double3 *ace,double3 *SumFr,double *tao,const float *divr,const float boundaryfs,const float freesurface,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall,tfloat3 gravity)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf){
    unsigned p1=p+npb;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
		double3 sumfr=make_double3(0,0,0);
    double3 dud=make_double3(0,0,0);
		double3 dvd=make_double3(0,0,0);
		double3 dwd=make_double3(0,0,0);
		double3 dwxp1=dwxcorrg[Correctp1];
		double3 dwyp1=dwycorrg[Correctp1];
		double3 dwzp1=dwzcorrg[Correctp1];
		double3 taop1=make_double3(0,0,0);
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
    double3 velp1;
    KerGetParticleData(p1,posxy,posz,velrhop,velp1,posdp1);
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
        if(pfin) KerViscousSchwaigerCalc<tker,ftmode> (false,p1,pini,pfin,ftomassp,posxy,posz,velrhop,code,idp,CTE.massf,ftmassp1,ftp1,posdp1,velp1,sumfr,dud,dvd,dwd,dwxp1,dwyp1,dwzp1,divr,boundaryfs,taop1,pressure,PistonPos,PistonVel,RightWall,gravity);
			}
    }

		double mu2=2.0*viscof;
    double3 r=ace[Correctp1]; 
    r.x-=mu2*(dud.x*sumfr.x+dud.y*sumfr.y+dud.z*sumfr.z); 
		r.y-=mu2*(dvd.x*sumfr.x+dvd.y*sumfr.y+dvd.z*sumfr.z); 
		r.z-=mu2*(dwd.x*sumfr.x+dwd.y*sumfr.y+dwd.z*sumfr.z);
		taop1.x=1.0/(taop1.x+CTE.eta2); taop1.z=1.0/(taop1.z+CTE.eta2);
		double taoFinal=-0.5*(taop1.x+taop1.z);
    ace[Correctp1]=r;
		if(divr[p1]>freesurface){ ace[Correctp1].x=ace[Correctp1].x*taoFinal; ace[Correctp1].z=ace[Correctp1].z*taoFinal;}
		tao[Correctp1]=taoFinal;
		SumFr[Correctp1].x+=sumfr.x; SumFr[Correctp1].y+=sumfr.y; SumFr[Correctp1].z+=sumfr.z;
  }
}

//==============================================================================
/// Collects kernel information.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode> void Interaction_ForcesT_KerInfo(StKerInfo *kerinfo)
{
	#if CUDART_VERSION >= 6050
  {
    typedef void (*fun_ptr)(TpInter,unsigned,unsigned,int,uint4,unsigned,float,float,const int2*,int3,const unsigned*,const float*,const double2*,const double*,const double3*,const word*,const unsigned*,double3*,double3*,double3*,double3*,float*,unsigned*,const unsigned,const float,const double*,const double,const double,const double,tfloat3);
    fun_ptr ptr=&KerInteractionForcesFluid<tker,ftmode>;
    int qblocksize=0,mingridsize=0;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize,&qblocksize,(void*)ptr,0,0);
    struct cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,(void*)ptr);
    kerinfo->forcesfluid_bs=qblocksize;
    kerinfo->forcesfluid_rg=attr.numRegs;
    kerinfo->forcesfluid_bsmax=attr.maxThreadsPerBlock;
    //printf(">> KerInteractionForcesFluid  blocksize:%u (%u)\n",qblocksize,0);
  }

  CheckErrorCuda("Error collecting kernel information.");
#endif
}

//==============================================================================
/// Interaccion para el calculo de fuerzas.
/// Interaction for the force computation.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,bool schwaiger> void Interaction_ForcesT
  (TpSlipCond tslipcond,TpCellMode cellmode,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,double3 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg
  ,const float *ftomassp,double3 *ace,bool simulate2d,float *divr,const double3 *mirrorPos,const unsigned *mirrorCell,double4 *mls,unsigned *row
	,double3 *SumFr,double *tao,const float boundaryfs,const float freesurface,const double pistonposx,StKerInfo *kerinfo,JBlockSizeAuto *bsauto,const double *pressure,const double pistonvel,const double RightWall,const tfloat3 gravity)
{
	const unsigned npf=np-npb;
	if(kerinfo)Interaction_ForcesT_KerInfo<tker,ftmode>(kerinfo);
  //else if(bsauto)Interaction_ForcesT_BsAuto<psimple,tker,ftmode,lamsps,tdelta,shift>(cellmode,viscob,viscof,bsbound,bsfluid,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,pospress,velrhop,code,idp,ftomassp,tau,gradvel,viscdt,ar,ace,delta,tshifting,shiftpos,shiftdetect,simulate2d,bsauto);
  else if(npf){
		//-Executes particle interactions.
		const int hdiv=(cellmode==CELLMODE_H? 2: 1);
		const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
		const unsigned cellfluid=nc.w*nc.z+1;
		const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
	
		dim3 sgridb=GetGridSize(npbok,bsbound);
		dim3 sgridf=GetGridSize(npf,bsfluid);

		//-Interaccion Fluid-Fluid & Fluid-Bound
		//-Interaction Fluid-Fluid & Fluid-Bound
		const unsigned matOrder=npb;

		KerInteractionForcesFluid<tker,ftmode> <<<sgridf,bsfluid>>> (tinter,npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ace,divr,row,matOrder,boundaryfs,pressure,pistonposx,pistonvel,RightWall,gravity);

		if(tinter==1){
			if(simulate2d) KerInverseKernelCor2D <<<sgridf,bsfluid>>> (npf,npb,dwxcorrg,dwzcorrg,code);
		}

		if(schwaiger&&tinter==1) KerViscousSchwaiger<tker,ftmode> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ace,SumFr,tao,divr,boundaryfs,freesurface,pressure,pistonposx,pistonvel,RightWall,gravity);
	}
}
//==============================================================================
void Interaction_Forces(TpKernel tkernel,bool floating,bool usedem,TpSlipCond tslipcond,bool schwaiger,TpCellMode cellmode
  ,float viscob,float viscof,unsigned bsbound,unsigned bsfluid
  ,TpInter tinter,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz,double3 *velrhop,const word *code,const unsigned *idp,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg
  ,const float *ftomassp,double3 *ace,bool simulate2d,float *divr,const double3 *mirrorPos,const unsigned *mirrorCell
	,double4 *mls,unsigned *row,double3 *SumFr,double *tao,const float boundaryfs,const float freesurface,const double pistonposx,StKerInfo *kerinfo,JBlockSizeAuto *bsauto,const double *pressure,const double pistonvel,const double RightWall,tfloat3 gravity)
{
	if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
		if(!schwaiger){
			if(!floating)   Interaction_ForcesT<tker,FTMODE_None,false> (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else if(!usedem)Interaction_ForcesT<tker,FTMODE_Sph,false>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else            Interaction_ForcesT<tker,FTMODE_Dem,false>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
		}else{
			if(!floating)   Interaction_ForcesT<tker,FTMODE_None,true> (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else if(!usedem)Interaction_ForcesT<tker,FTMODE_Sph,true>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else            Interaction_ForcesT<tker,FTMODE_Dem,true>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);	
		}
	}
	else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
		if(!schwaiger){
			if(!floating)   Interaction_ForcesT<tker,FTMODE_None,false> (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else if(!usedem)Interaction_ForcesT<tker,FTMODE_Sph,false>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else            Interaction_ForcesT<tker,FTMODE_Dem,false>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,NULL,NULL,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
		}else{
			if(!floating)   Interaction_ForcesT<tker,FTMODE_None,true> (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else if(!usedem)Interaction_ForcesT<tker,FTMODE_Sph,true>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
			else            Interaction_ForcesT<tker,FTMODE_Dem,true>  (tslipcond,cellmode,viscob,viscof,bsbound,bsfluid,tinter,np,npb,npbok,ncells,begincell,cellmin,dcell,posxy,posz,velrhop,code,idp,dwxcorrg,dwycorrg,dwzcorrg,ftomassp,ace,simulate2d,divr,mirrorPos,mirrorCell,mls,row,SumFr,tao,boundaryfs,freesurface,pistonposx,kerinfo,bsauto,pressure,pistonvel,RightWall,gravity);
		}
	}
}

/*//##############################################################################
//# Kernels para interaccion DEM.
//# Kernels for DEM interaction
//##########################################.####################################
//------------------------------------------------------------------------------
/// Realiza la interaccion DEM de una particula con un conjunto de ellas. (Float-Float/Bound)
/// DEM interaction of a particle with a set of particles (Float-Float/Bound).
//------------------------------------------------------------------------------
template<bool psimple> __device__ void KerInteractionForcesDemBox 
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
}
*/

//##############################################################################
//# Kernels para Shifting.
//# Shifting kernels.
//##############################################################################
//------------------------------------------------------------------------------
/// Calcula Shifting final para posicion de particulas.
/// Computes final shifting for the particle position.
//------------------------------------------------------------------------------
__global__ void KerRunShifting(const bool simulate2d,unsigned n,unsigned pini,double dt
  ,float shiftcoef,float freesurface,double3 *velrhop,const float *divr,double3 *shiftpos
	,const float ShiftOffset,const double alphashift,const bool maxShift,double3 *sumtensile,const double beta0,const double beta1)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<n){
    const unsigned p1=p+pini;
		const unsigned Correctp1=p;
    double3 rshiftpos=shiftpos[Correctp1];
    float divrp1=divr[p1];
		double h=double(CTE.h);
		double dp=double(CTE.dp);
    //double umagn=-double(shiftcoef)*double(CTE.h)*dt*sqrt(velrhop[p1].x*velrhop[p1].x+velrhop[p1].y*velrhop[p1].y+velrhop[p1].z*velrhop[p1].z);
		double umagn=-double(shiftcoef)*h*h;

		double3 norm=make_double3(-rshiftpos.x,-rshiftpos.y,-rshiftpos.z);

		if(simulate2d) norm.y=0;
	  double3 tang=make_double3(0,0,0);
	  double3 bitang=make_double3(0,0,0);
		rshiftpos.x+=sumtensile[Correctp1].x; rshiftpos.y+=sumtensile[Correctp1].y; rshiftpos.z+=sumtensile[Correctp1].z;
		if(simulate2d)  rshiftpos.y=0;

	  //-tangent and bitangent calculation
	  tang.x=norm.z+norm.y;		
	  if(!simulate2d)tang.y=-(norm.x+norm.z);	
	  tang.z=-norm.x+norm.y;
		if(!simulate2d){
			bitang.x=tang.y*norm.z-norm.y*tang.z;
			bitang.y=norm.x*tang.z-tang.x*norm.z;
			bitang.z=tang.x*norm.y-norm.x*tang.y;
		}

	  //-unit normal vector
	  double temp=norm.x*norm.x+norm.y*norm.y+norm.z*norm.z;
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
    else {tang.x=0.f; tang.y=0.f; tang.z=0.f;}

	  //-unit bitangent vector
		if(!simulate2d){
			temp=bitang.x*bitang.x+bitang.y*bitang.y+bitang.z*bitang.z;
			if(temp){
				temp=sqrt(temp);
				bitang.x=bitang.x/temp; bitang.y=bitang.y/temp; bitang.z=bitang.z/temp;
			}
			else {bitang.x=0.f; bitang.y=0.f; bitang.z=0.f;}
		}

	  //-gradient calculation
	  double dcds=tang.x*rshiftpos.x+tang.z*rshiftpos.z+tang.y*rshiftpos.y;
	  double dcdn=norm.x*rshiftpos.x+norm.z*rshiftpos.z+norm.y*rshiftpos.y;
	  double dcdb=bitang.x*rshiftpos.x+bitang.z*rshiftpos.z+bitang.y*rshiftpos.y;

    if(divrp1<freesurface){
			dcdn-=beta0;
			double factorNormShift=0;//alphashift;
      rshiftpos.x=dcds*tang.x+dcdb*bitang.x+(dcdn*norm.x)*factorNormShift;
      if(!simulate2d) rshiftpos.y=dcds*tang.y+dcdb*bitang.y+(dcdn*norm.y)*factorNormShift;
      rshiftpos.z=dcds*tang.z+dcdb*bitang.z+(dcdn*norm.z)*factorNormShift;
    }
    else if(divrp1<=freesurface+ShiftOffset){ 
			dcdn-=beta1;
			double factorNormShift=alphashift;
			rshiftpos.x=dcds*tang.x+dcdb*bitang.x+dcdn*norm.x*factorNormShift;
      if(!simulate2d) rshiftpos.y=dcds*tang.y+dcdb*bitang.y+(dcdn*norm.y)*factorNormShift;
      rshiftpos.z=dcds*tang.z+dcdb*bitang.z+dcdn*norm.z*factorNormShift;
    }

    rshiftpos.x=rshiftpos.x*umagn;
    if(!simulate2d) rshiftpos.y=rshiftpos.y*umagn;
    rshiftpos.z=rshiftpos.z*umagn;

    //Max Shifting
		if(maxShift){
      double absShift=sqrt(rshiftpos.x*rshiftpos.x+rshiftpos.y*rshiftpos.y+rshiftpos.z*rshiftpos.z);
			double maxDist=0.1*dp;
      if(abs(rshiftpos.x)>maxDist) rshiftpos.x=maxDist*rshiftpos.x/absShift;
      if(abs(rshiftpos.y)>maxDist) rshiftpos.y=maxDist*rshiftpos.y/absShift;
      if(abs(rshiftpos.z)>maxDist) rshiftpos.z=maxDist*rshiftpos.z/absShift;
		}

    shiftpos[Correctp1]=rshiftpos;
  }
}

//==============================================================================
/// Calcula Shifting final para posicion de particulas.
/// Computes final shifting for the particle position.
//==============================================================================
void RunShifting(const bool simulate2d,unsigned np,unsigned npb,double dt
  ,double shiftcoef,float freesurface,double3 *velrhop,const float *divr,double3 *shiftpos
	,bool maxShift,double3 *sumtensile,const float shiftoffset,const double alphashift,const double betashift0,const double betashift1){
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerRunShifting <<<sgrid,SPHBSIZE>>> (simulate2d,npf,npb,dt,shiftcoef,freesurface,velrhop,divr,shiftpos,shiftoffset,alphashift,maxShift,sumtensile,betashift0,betashift1);
  }
}

//------------------------------------------------------------------------------
/// Calcula los nuevos valores de Pos, Vel y Rhop (usando para Symplectic-Predictor)
/// Computes new values for Pos, Check, Vel and Ros (used with Symplectic-Predictor).
//------------------------------------------------------------------------------
template<bool floating> __global__ void KerComputeStepSymplecticPre
  (unsigned npf,unsigned npb,const double3 *ace,double dtm,word *code,double3 *velrhop)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
		const unsigned p1=p+npb;
		const unsigned Correctp1=p;
    double3 rvelrhop=velrhop[p1];

    const double3 race=ace[Correctp1];
    rvelrhop.x+=race.x*dtm;
    rvelrhop.y+=race.y*dtm;
    rvelrhop.z+=race.z*dtm;

    velrhop[p1]=rvelrhop;
  }
}

//==============================================================================
/// Actualizacion de particulas usando Symplectic-Predictor.
/// Updates particles using Symplectic-Predictor.
//==============================================================================   
void ComputeStepSymplecticPre(const unsigned bsfluid,bool floating,unsigned np,unsigned npb,const double3 *ace,double dtm
  ,word *code,double3 *velrhop)
{
  if(np){
		const unsigned npf=np-npb;
    dim3 sgrid=GetGridSize(npf,bsfluid);
    if(floating)KerComputeStepSymplecticPre<true>  <<<sgrid,bsfluid>>> (npf,npb,ace,dtm,code,velrhop);
    else        KerComputeStepSymplecticPre<false> <<<sgrid,bsfluid>>> (npf,npb,ace,dtm,code,velrhop);
  }
}

//------------------------------------------------------------------------------
/// ES:
/// Calcula los nuevos valores de Pos, Vel y Rhop (usandopara Symplectic-Corrector)
/// Pone vel de contorno a cero.
/// - EN:
/// Computes new values for Pos, Check, Vel and Ros (using Symplectic-Corrector).
/// The value of Vel always set to be reset.
//------------------------------------------------------------------------------
__device__ void KerDampingZone(const double xp1,double3 &rvelrhop,const double dampingpoint,const double dampinglength)
{
	if(xp1>dampingpoint){
		double fx=1.0-exp(-2.0*(dampinglength-(xp1-dampingpoint)));
		rvelrhop.x=rvelrhop.x*fx;
		rvelrhop.y=rvelrhop.y*fx;
		rvelrhop.z=rvelrhop.z*fx;
	}
}
__device__ void KerCorrectVelocity(const unsigned p1,const unsigned nearestBound,const double2 *posxy,const double *posz,double3 &rvelrhop,double3 *velrhop,const unsigned *idpg,const double3 *mirrorPos)
{
	double3 NormDir=make_double3(0,0,0), NormVelWall=make_double3(0,0,0), NormVelp1=make_double3(0,0,0);
	const unsigned nearestID=idpg[nearestBound];
	const double3 velwall=make_double3(velrhop[nearestBound].x,velrhop[nearestBound].y,velrhop[nearestBound].z);
	const double3 velp1=make_double3(rvelrhop.x,rvelrhop.y,rvelrhop.z);
	NormDir.x=mirrorPos[nearestID].x-posxy[nearestBound].x;
	NormDir.y=mirrorPos[nearestID].y-posxy[nearestBound].y;
	NormDir.z=mirrorPos[nearestID].z-posz[nearestBound];
	double MagNorm=NormDir.x*NormDir.x+NormDir.y*NormDir.y+NormDir.z*NormDir.z;
	if(MagNorm){MagNorm=sqrt(MagNorm); NormDir.x=NormDir.x/MagNorm; NormDir.y=NormDir.y/MagNorm; NormDir.z=NormDir.z/MagNorm;}
	double NormProdVelWall=velwall.x*NormDir.x+velwall.y*NormDir.y+velwall.z*NormDir.z;
	double NormProdVelp1=velp1.x*NormDir.x+velp1.y*NormDir.y+velp1.z*NormDir.z;

	NormVelWall.x=NormDir.x*NormProdVelWall; NormVelp1.x=NormDir.x*NormProdVelp1;
	NormVelWall.y=NormDir.y*NormProdVelWall; NormVelp1.y=NormDir.y*NormProdVelp1;
	NormVelWall.z=NormDir.z*NormProdVelWall; NormVelp1.z=NormDir.z*NormProdVelp1;

	double dux=NormVelp1.x-NormVelWall.x;
	double duy=NormVelp1.y-NormVelWall.y;
	double duz=NormVelp1.z-NormVelWall.z;

	double VelNorm=dux*NormDir.x+duy*NormDir.y+duz*NormDir.z;
	if(VelNorm<0){
		rvelrhop.x-=VelNorm*NormDir.x;
		rvelrhop.y-=VelNorm*NormDir.y;
		rvelrhop.z-=VelNorm*NormDir.z;
	}
}

template<bool floating> __global__ void KerComputeStepSymplecticCor
  (unsigned npf,unsigned npb
  ,const double3 *velrhoppre,const double3 *ace,double dtm,double dt,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,double3 *velrhop,tfloat3 gravity,const unsigned *row,const double2 *posxy,const double *posz,const unsigned *idp,const double3 *mirrorPos
	,const bool wavegen,const double dampingpoint,const double dampinglength)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
		const unsigned p1=p+npb;
		const unsigned Correctp1=p;
    double3 rvelrhop=velrhop[p1];
	  double3 rvelp=velrhoppre[p1];
    //-Actualiza velocidad.
		//-Updates velocity.

    double3 race=ace[Correctp1];
    rvelrhop.x-=(race.x-gravity.x)*dt;
    rvelrhop.y-=(race.y-gravity.y)*dt;
    rvelrhop.z-=(race.z-gravity.z)*dt;

		//if(row[p1]!=npb) KerCorrectVelocity(p1,row[p1],posxy,posz,rvelrhop,velrhop,idp,mirrorPos);
		if(wavegen){
			const double xp1=posxy[p1].x;
			KerDampingZone(xp1,rvelrhop,dampingpoint,dampinglength);
		}
		//-Computes and stores position displacement.
    double dx=(rvelp.x+rvelrhop.x)*dtm;
    double dy=(rvelp.y+rvelrhop.y)*dtm;
    double dz=(rvelp.z+rvelrhop.z)*dtm;
			
    movxy[p1]=make_double2(dx,dy);
    movz[p1]=dz;
		velrhop[p1]=rvelrhop;
  }
}

//==============================================================================
// Actualizacion de particulas usando Symplectic-Corrector.
/// Updates particles using Symplectic-Corrector.
//==============================================================================   
void ComputeStepSymplecticCor(const unsigned bsfluid,bool floating,unsigned np,unsigned npb
  ,const double3 *velrhoppre,const double3 *ace,double dtm,double dt,float rhopoutmin,float rhopoutmax
  ,word *code,double2 *movxy,double *movz,double3 *velrhop,tfloat3 gravity,const unsigned *row,const double2 *posxy,const double *posz,const unsigned *idp,const double3 *mirrorPos
	,const bool wavegen,const double dampingpoint,const double dampinglength)
{
  if(np){
		const unsigned npf=np-npb;
    dim3 sgrid=GetGridSize(npf,bsfluid);
    if(floating)KerComputeStepSymplecticCor<true>  <<<sgrid,bsfluid>>> (npf,npb,velrhoppre,ace,dtm,dt,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop,gravity,row,posxy,posz,idp,mirrorPos,wavegen,dampingpoint,dampinglength);
    else        KerComputeStepSymplecticCor<false> <<<sgrid,bsfluid>>> (npf,npb,velrhoppre,ace,dtm,dt,rhopoutmin,rhopoutmax,code,movxy,movz,velrhop,gravity,row,posxy,posz,idp,mirrorPos,wavegen,dampingpoint,dampinglength);
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
void ComputeStepPos2(const unsigned bsfluid,byte periactive,bool floating,unsigned np,unsigned npb
  ,const double2 *posxypre,const double *poszpre,const double2 *movxy,const double *movz
  ,double2 *posxy,double *posz,unsigned *dcell,word *code)
{
  const unsigned pini=npb;
  const unsigned npf=np-pini;
  if(npf){
    dim3 sgrid=GetGridSize(npf,bsfluid);
    if(periactive){ const bool peri=true;
      if(floating)KerComputeStepPos2<peri,true>  <<<sgrid,bsfluid>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos2<peri,false> <<<sgrid,bsfluid>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
    }
    else{ const bool peri=false;
      if(floating)KerComputeStepPos2<peri,true>  <<<sgrid,bsfluid>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else        KerComputeStepPos2<peri,false> <<<sgrid,bsfluid>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
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
  ,const unsigned *ridpmv,double2 *posxy,double *posz,unsigned *dcell,double3 *velrhop,word *code,const unsigned *idpg,double3 *mirrorPos,unsigned *mirrorCell)
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
      velrhop[pid]=make_double3(mvvel.x,mvvel.y,mvvel.z);
    }
  }
}

//==============================================================================
/// Aplica un movimiento lineal a un conjunto de particulas.
/// Applies a linear movement to a set of particles.
//==============================================================================
void MoveLinBound(byte periactive,unsigned np,unsigned ini,tdouble3 mvpos,tfloat3 mvvel
  ,const unsigned *ridp,double2 *posxy,double *posz,unsigned *dcell,double3 *velrhop,word *code,const unsigned *idpg,double3 *mirrorPos,unsigned *mirrorCell)
{
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  if(periactive)KerMoveLinBound<true>  <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridp,posxy,posz,dcell,velrhop,code,idpg,mirrorPos,mirrorCell);
  else          KerMoveLinBound<false> <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridp,posxy,posz,dcell,velrhop,code,idpg,mirrorPos,mirrorCell);
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

__global__ void KerPistonCorner(const unsigned npb,double2 *posxy,const double *posz,const unsigned *idp,double3 *mirrorpos,word *code,const double pistonposX,const double pistonposZ,const double pistonYmin,const double pistonYmax,const bool simulate2d,double3 *velrhop,const float pistonvelx)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
  if(p1<npb){
		if(CODE_GetType(code[p1])!=CODE_TYPE_MOVING){
			const unsigned idp1=idp[p1];
			double3 posdp1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
			double x2=pistonposX-posdp1.x; x2=x2*x2;
			if(x2<=CTE.fourh2){
 				if(posdp1.x<pistonposX){
 					mirrorpos[idp1].x=2.0*pistonposX-posxy[p1].x;
					mirrorpos[idp1].z=2.0*pistonposZ-posz[p1];
					velrhop[p1].x=pistonvelx;
  			}
 				else{
					mirrorpos[idp1].x=posxy[p1].x;
					mirrorpos[idp1].z=2.0*pistonposZ-posz[p1];
					velrhop[p1].x=0;
				}
			}
		}
	}
}

//==============================================================================
/// Recalculate mirror points for wavegen boudary
//==============================================================================
void PistonCorner(const unsigned bsbound,const unsigned npb,double2 *posxy,const double *posz,const unsigned *idp,double3 *mirrorpos,word *code,const double pistonposX,const double pistonposZ,const double pistonYmin,const double pistonYmax,const bool simulate2d,double3 *velrhop,const float pistonvelx)
{
	dim3 sgridb=GetGridSize(npb,bsbound);
	KerPistonCorner <<<sgridb,bsbound>>> (npb,posxy,posz,idp,mirrorpos,code,pistonposX,pistonposZ,pistonYmin,pistonYmax,simulate2d,velrhop,pistonvelx);
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
template<bool varspre> __global__ void KerPeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,uint3 cellmax,double3 perinc,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,double2 *posxypre,double *poszpre,float4 *velrhoppre)
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
  }
}

//==============================================================================
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Creates periodic particles from a list of particles to duplicate.
//==============================================================================
void PeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,tuint3 domcells,tdouble3 perinc,const unsigned *listp,unsigned *idp,word *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,double2 *posxypre,double *poszpre,float4 *velrhoppre)
{
  if(n){
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    if(posxypre!=NULL)KerPeriodicDuplicateSymplectic<true>  <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,posxypre,poszpre,velrhoppre);
    else              KerPeriodicDuplicateSymplectic<false> <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,posxypre,poszpre,velrhoppre);
  }
}


//##############################################################################
//# Kernels para external forces (JSphVarAcc)
//# Kernels for external forces (JSphVarAcc)
//##############################################################################
//------------------------------------------------------
/// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddAccInputAng(unsigned n,unsigned pini,word codesel,float3 gravity
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
      const double velx=rvel.x;
      const double vely=rvel.y;
      const double velz=rvel.z;

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
      accx+=((2.0*velang.y)*velz)-((2.0*velang.z)*(vely-vellin.y));
      accy+=((2.0*velang.z)*velx)-((2.0*velang.x)*(velz-vellin.z));
      accz+=((2.0*velang.x)*vely)-((2.0*velang.y)*(velx-vellin.x));

      //-Stores the new acceleration value.
      ace[p]=make_float3(float(accx),float(accy),float(accz));
    }
  }
}

//------------------------------------------------------
/// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddAccInputLin(unsigned n,unsigned pini,word codesel,float3 gravity
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
void AddAccInput(unsigned n,unsigned pini,word codesel
  ,tdouble3 acclin,tdouble3 accang,tdouble3 centre,tdouble3 velang,tdouble3 vellin,bool setgravity
  ,tfloat3 gravity,const word *code,const double2 *posxy,const double *posz,const float4 *velrhop,float3 *ace)
{
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    const bool withaccang=(accang.x!=0 || accang.y!=0 || accang.z!=0);
    if(withaccang)KerAddAccInputAng <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(gravity),setgravity,Double3(acclin),Double3(accang),Double3(centre),Double3(velang),Double3(vellin),code,posxy,posz,velrhop,ace);
    else          KerAddAccInputLin <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(gravity),setgravity,Double3(acclin),code,ace);
  }
}

//==========================
///Initial advection - r*
//==========================
template<bool floating> __global__ void KerComputeRStar(unsigned npf,unsigned npb,const double3 *velrhoppre,double dtm,word *code,double2 *movxy,double *movz)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
    unsigned p1=p+npb;
    //-Particulas: Fixed & Moving //-Particles: Fixed & Moving
      double3 rvelrhop=velrhoppre[p1];
      double dx=rvelrhop.x*dtm;
      double dy=rvelrhop.y*dtm;
      double dz=rvelrhop.z*dtm;
        
      movxy[p1]=make_double2(dx,dy);
      movz[p1]=dz;  
  }
}

void ComputeRStar(const unsigned bsfluid,bool floating,unsigned npf,unsigned npb,const double3 *velrhoppre,double dtm,word *code,double2 *movxy,double *movz)
{
  if(npf){
    dim3 sgrid=GetGridSize(npf,bsfluid);
    if(floating)KerComputeRStar<true> <<<sgrid,bsfluid>>> (npf,npb,velrhoppre,dtm,code,movxy,movz);
    else        KerComputeRStar<false><<<sgrid,bsfluid>>> (npf,npb,velrhoppre,dtm,code,movxy,movz);
  }
}

//==============================================================================
///Matrix A Setup
//==============================================================================
__global__ void kerMatrixASetup(const unsigned end,const unsigned start,const unsigned matOrder,const unsigned ppedim,unsigned int *row,unsigned *nnz,unsigned *numfreesurface,const float *divr,const float freesurface){
   unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of particle.
   if(p==0){
	   for(int p1=int(start);p1<int(end);p1++){
			 const unsigned oi=p1-matOrder;
			 if(divr[p1]<=freesurface){
				 row[oi]=0;
				 numfreesurface[0]++;
			 }
       unsigned nnzOld=nnz[0];
       nnz[0] +=row[oi]+1;
       row[oi]=nnzOld;  
     }
     row[ppedim]=nnz[0];
   }
}

void MatrixASetup(const unsigned np,const unsigned npb,const unsigned npbok,const unsigned ppedim,unsigned int*row,unsigned *nnz,unsigned *numfreesurface,const float *divr,const float freesurface){
  const unsigned npf=np-npb;
	const unsigned matOrder=npb;
	if(npf){
		kerMatrixASetup <<<1,1>>> (np,npb,matOrder,ppedim,row,nnz,numfreesurface,divr,freesurface);
  }
}

//==============================================================================
///Matrix A Population
//==============================================================================
template<TpKernel tker,bool schwaiger> __device__ void KerMatrixAFluidSelf
  (const unsigned p1,const unsigned matOrder,const double2 *posxy,const double *posz,double3 posdp1
	,const double3 velp1,const double3 *velrhop,const double3 dwx,const double3 dwy,const double3 dwz,tfloat3 gravity,const float massp2,const float RhopZero,const word *code,const unsigned *idp,unsigned &index
	,unsigned int *col,double *matrixInd,double *matrixb,const int diag,const double3 *mirrorPos,const unsigned oi,double &divU,double &Neumann,const double3 sumfr,const float *divr,const float boundaryfs,const double taop1,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall)
{
	float volumep2=massp2/RhopZero; //Volume of particle j
	double drx,dry,drz;
  for(int count=1;count<=5;count++){
		bool interact=false;
		double3 velp2;
		double pressp2;
		double NeumannDist=0;
		KerGetParticlesDr(count,p1,posxy,posz,interact,posdp1,velrhop[p1],pressure[p1],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
		if(interact){
			double rr2=drx*drx+dry*dry+drz*drz;
			if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
				//-Wendland kernel.
				double frx,fry,frz;
				if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
				//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);

				const double temp_x=frx*dwx.x/*+fry*dwy.x*/+frz*dwz.x;
				//const double temp_y=frx*dwx.y+fry*dwy.y+frz*dwz.y;
				const double temp_z=frx*dwx.z/*+fry*dwy.z*/+frz*dwz.z;

				//===== Laplacian operator =====
				const double rDivW=drx*frx+dry*fry+drz*frz;
				double temp=rDivW/(rr2+CTE.eta2);

				if(schwaiger){
					const double Schwaigergrad=temp_x*sumfr.x+/*temp_y*sumfr.y+*/temp_z*sumfr.z;
					temp+=Schwaigergrad;
					temp=temp*taop1;
				}

				temp=temp*volumep2*2.0/RhopZero;

				//=====Divergence of velocity==========
				double dvx=velp1.x-velp2.x, /*dvy=velp1.y-velrhop[p2].y,*/ dvz=velp1.z-velp2.z;
				const double tempDivU=dvx*temp_x+/*dvy*temp_y+*/dvz*temp_z;
				divU-=volumep2*tempDivU;

				//=====dp/dn=====
				double temp2=temp*RhopZero*gravity.z*NeumannDist;
				Neumann+=temp2; 
			}
		}
  }
}

template<TpKernel tker,bool schwaiger> __device__ void KerMatrixAFluid
  (const unsigned p1,const bool fluid,const unsigned matOrder,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,double3 posdp1
	,const double3 velp1,const double3 *velrhop,const double3 dwx,const double3 dwy,const double3 dwz,tfloat3 gravity,const float massp2,const float RhopZero,const word *code,const unsigned *idp,unsigned &index
	,unsigned int *col,double *matrixInd,double *matrixb,const int diag,const double3 *mirrorPos,const unsigned oi,double &divU,double &Neumann,const double3 sumfr,const float *divr,const float boundaryfs,const double taop1,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall)
{
	float volumep2=massp2/RhopZero; //Volume of particle j
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
		bool entry=false;
		if(p1!=p2){
			for(int count=0;count<=5;count++){
				bool interact=false;
				double3 velp2;
				double pressp2;
				double NeumannDist=0;
				KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
				if(interact){
					double rr2=drx*drx+dry*dry+drz*drz;
					if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
						entry=true;
						unsigned oj=p2;
						if(fluid)oj-=matOrder;
						//-Wendland kernel.
						double frx,fry,frz;
						if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
						//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);

						const double temp_x=frx*dwx.x/*+fry*dwy.x*/+frz*dwz.x;
						//const double temp_y=frx*dwx.y+fry*dwy.y+frz*dwz.y;
						const double temp_z=frx*dwx.z/*+fry*dwy.z*/+frz*dwz.z;

						//===== Laplacian operator =====
						const double rDivW=drx*frx/*+dry*fry*/+drz*frz;
						double temp=rDivW/(rr2+CTE.eta2);

						if(schwaiger){
							const double Schwaigergrad=temp_x*sumfr.x+/*temp_y*sumfr.y+*/temp_z*sumfr.z;
							temp+=Schwaigergrad;
							temp=temp*taop1;
						}

						temp=temp*volumep2*2.0/RhopZero;

						matrixInd[index]-=temp;
						col[index]=oj;
						matrixInd[diag]+=temp;

						//=====Divergence of velocity==========
						double dvx=velp1.x-velp2.x, /*dvy=velp1.y-velrhop[p2].y,*/ dvz=velp1.z-velp2.z;
						const double tempDivU=dvx*temp_x+/*dvy*temp_y+*/dvz*temp_z;
						divU-=volumep2*tempDivU;

						//=====dp/dn=====
						double temp2=temp*RhopZero*gravity.z*NeumannDist;
						Neumann+=temp2; 
					}
				}
			}
			if(entry) index++;
		}
  }
}

template<TpKernel tker,bool schwaiger> __global__ void KerPopulateMatrixAFluid
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell,tfloat3 gravity
  ,const double2 *posxy,const double *posz,const double3 *velrhop,const double3 *dwxCorr,const double3 *dwyCorr,const double3 *dwzCorr,const float *divr,const word *code
  ,const unsigned *idp,unsigned int *row,unsigned int *col,double *matrixInd,double *matrixb,const float freesurface,const double3 *mirrorPos,const double dt,const unsigned matOrder,const double3 *SumFr,const double *tao,const float boundaryfs,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
    if(CODE_GetSpecialValue(code[p1])!=CODE_PERIODIC){
      unsigned oi=p1-matOrder;
      const unsigned diag=row[oi];
      col[diag]=oi;
      unsigned index=diag+1;
			double divU=0;
			double Neumann=0;
      if(divr[p1]>freesurface){
				double3 sumfr=SumFr[Correctp1];
				const double taop1=tao[Correctp1];
        //-Obtiene datos basicos de particula p1.
  	    //-Obtains basic data of particle p1.
        double3 posdp1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
				double3 velp1=make_double3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
        //-Obtiene limites de interaccion
	      //-Obtains interaction limits
        int cxini,cxfin,yini,yfin,zini,zfin;
        KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
        
        //for(int fluid=0;fluid<=1;fluid++){
					for(int z=zini;z<zfin;z++){
						int zmod=(nc.w)*z+(cellfluid);//*fluid); //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
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
								KerMatrixAFluid<tker,schwaiger> (p1,1,matOrder,pini,pfin,posxy,posz,posdp1,velp1,velrhop,dwxCorr[Correctp1],dwyCorr[Correctp1],dwzCorr[Correctp1],gravity,CTE.massf,CTE.rhopzero,code,idp,index,col,matrixInd,matrixb,diag,mirrorPos,oi,divU,Neumann,sumfr,divr,boundaryfs,taop1,pressure,PistonPos,PistonVel,RightWall);
							}
						}
					}
					KerMatrixAFluidSelf<tker,schwaiger> (p1,matOrder,posxy,posz,posdp1,velp1,velrhop,dwxCorr[Correctp1],dwyCorr[Correctp1],dwzCorr[Correctp1],gravity,CTE.massf,CTE.rhopzero,code,idp,index,col,matrixInd,matrixb,diag,mirrorPos,oi,divU,Neumann,sumfr,divr,boundaryfs,taop1,pressure,PistonPos,PistonVel,RightWall);

				//}
      }
      else matrixInd[diag]=1.0;

			matrixb[oi]=Neumann+(divU/dt);
    }
  }
}

void PopulateMatrix(TpKernel tkernel,bool schwaiger,TpCellMode cellmode,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const int2 *begincell,tuint3 cellmin
	,const unsigned *dcell,tfloat3 gravity,const double2 *posxy,const double *posz,const double3 *velrhop,const double3 *dwxCorr,const double3 *dwyCorr,const double3 *dwzCorr,double *matrixInd,double *matrixb
  ,unsigned int *row,unsigned int *col,const unsigned *idp,const float *divr,const word *code,const float freesurface,const double3 *mirrorPos,const unsigned *mirrorCell,const double4 *mls,const double dt,const double3 *SumFr,const double *tao,const float boundaryfs,const double *pressure,const double PistonPos,const double PistonVel,const double RightWall){
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
		const unsigned matOrder=npb;

		if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
			if(!schwaiger) KerPopulateMatrixAFluid<tker,false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,row,col,matrixInd,matrixb,freesurface,mirrorPos,dt,matOrder,NULL,NULL,boundaryfs,pressure,PistonPos,PistonVel,RightWall); 
			else KerPopulateMatrixAFluid<tker,true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,row,col,matrixInd,matrixb,freesurface,mirrorPos,dt,matOrder,SumFr,tao,boundaryfs,pressure,PistonPos,PistonVel,RightWall); 
		}
		else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
			if(!schwaiger) KerPopulateMatrixAFluid<tker,false> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,row,col,matrixInd,matrixb,freesurface,mirrorPos,dt,matOrder,NULL,NULL,boundaryfs,pressure,PistonPos,PistonVel,RightWall); 
			else KerPopulateMatrixAFluid<tker,true> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,gravity,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,row,col,matrixInd,matrixb,freesurface,mirrorPos,dt,matOrder,SumFr,tao,boundaryfs,pressure,PistonPos,PistonVel,RightWall); 
		}
	}
}

__global__ void KerFreeSurfaceMark
  (unsigned n,unsigned pinit,const unsigned matOrder,float *divr,double *matrixInd, double *matrixb,unsigned int *row,const word *code,const double pi,const float freesurface,const float shiftoffset)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
    unsigned oi=p1-matOrder;
    const int Mark=row[oi]+1;
    if(divr[p1]>=freesurface && divr[p1]<=freesurface+shiftoffset){
      double alpha=0.5*(1.0-cos(pi*double(divr[p1]-freesurface)/shiftoffset));

			matrixb[oi]=matrixb[oi]*alpha;

      for(int index=Mark;index<row[oi+1];index++) matrixInd[index]=matrixInd[index]*alpha;
    }
  }
}

void FreeSurfaceMark(const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,float *divr
  ,double *matrixInd, double *matrixb,unsigned int *row,const word *code,const double pi,const float freesurface,const float shiftoffset){
  const unsigned npf=np-npb;

  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npbok,bsbound);
		const unsigned matOrder=npb-npbok;
    KerFreeSurfaceMark <<<sgridf,bsfluid>>> (npf,npb,matOrder,divr,matrixInd,matrixb,row,code,pi,freesurface,shiftoffset);
  }
}

//==============================================================================
/// Pressure Assign
//==============================================================================
//------------------------------------------------------------------------------
///Pressure Assign 
//------------------------------------------------------------------------------
__global__ void KerPressureSort
  (unsigned npf,const unsigned npb,const unsigned matOrder,double *pressure,double *pressure2)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf){
    unsigned p1=p+npb;      //-Nº de particula. //-NI of particle
		pressure2[p1]=pressure[p1-matOrder];
		pressure[p1-matOrder]=0;
  }
}

__global__ void KerPressureAssignFluid
  (unsigned npf,const unsigned npb,double *pressure,double *pressure2)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npf){
    unsigned p1=p+npb;      //-Nº de particula. //-NI of particle
		pressure[p1]=pressure2[p1];
  }
}

/*__global__ void KerPressureAssignBound
  (unsigned npbok,const tfloat3 gravity,const double *posz
  ,double *pressure,const unsigned *idp,const word *code,bool negpresbound,const double3 *mirrorPos)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<npbok){
    unsigned p1=p;      //-Nº de particula. //-NI of particle
		double dist=mirrorPos[idp[p1]].z-posz[p1];
		double Neumann=double(CTE.rhopzero)*abs(gravity.z)*dist;
		pressure[p1]+=Neumann;
  }
}*/

void PressureAssign(const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok
  ,const tfloat3 gravity,const double *posz,float4 *velrhop,double *pressure,const unsigned *idp
	,const word *code,bool negpresbound,const double3 *mirrorPos,double *pressure2){
  const unsigned npf=np-npb;

  if(np){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    dim3 sgridb=GetGridSize(npb,bsbound);
		
		const unsigned matOrder=npb;
		//KerPressureAssignBound <<<sgridb,bsbound>>> (npbok,gravity,posz,pressure,idp,code,negpresbound,mirrorPos); 
		KerPressureSort <<<sgridf,bsfluid>>> (npf,npb,matOrder,pressure,pressure2);
		KerPressureAssignFluid <<<sgridf,bsfluid>>> (npf,npb,pressure,pressure2);
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
template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag,typename ScalarType>
void run_solver(MatrixType const & matrix, VectorType const & rhs,SolverTag const & solver, PrecondTag const & precond,viennacl::vector<ScalarType> & vcl_result){ 
  VectorType result(rhs);
  //VectorType residual(rhs);
  viennacl::tools::timer timer;
  timer.start();
  result = viennacl::linalg::solve(matrix, rhs, solver, precond);   
  viennacl::backend::finish();  
  //std::cout << "  > Solver time: " << timer.get() << std::endl;   
  //residual -= viennacl::linalg::prod(matrix, result); 
	//double normResidual=viennacl::linalg::norm_2(residual);
  //if(normResidual){
		//std::cout << "  > Relative residual: " << normResidual / viennacl::linalg::norm_2(rhs) << std::endl;  
		//std::cout << "  > Iterations: " << solver.iters() << std::endl;
	//}
  viennacl::copy(result,vcl_result);
}

void solveVienna(TpPrecond tprecond,TpAMGInter tamginter,double tolerance,int iterations,int restart,float strongconnection,float jacobiweight, int presmooth,int postsmooth,int coarsecutoff,int coarselevels,double *matrixa,double *matrixx,double *matrixb,unsigned int *row,unsigned int *col,const unsigned nnz,const unsigned ppedim,const unsigned numfreesurface){
  viennacl::context CudaCtx(viennacl::CUDA_MEMORY);
  typedef double       ScalarType;

  viennacl::compressed_matrix<ScalarType> vcl_A_cuda(row, col, matrixa, viennacl::CUDA_MEMORY, ppedim, ppedim, nnz);

  viennacl::vector<ScalarType> vcl_vec(matrixb, viennacl::CUDA_MEMORY, ppedim);
  viennacl::vector<ScalarType> vcl_result(matrixx, viennacl::CUDA_MEMORY, ppedim);

  viennacl::linalg::bicgstab_tag bicgstab(tolerance,iterations);

	if(viennacl::linalg::norm_2(vcl_vec)){
		if(tprecond==PRECOND_Jacobi){
			//std::cout<<"JACOBI PRECOND" <<std::endl;
			viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<ScalarType> > vcl_jacobi(vcl_A_cuda,viennacl::linalg::jacobi_tag());
			run_solver(vcl_A_cuda,vcl_vec,bicgstab,vcl_jacobi,vcl_result);
		}
		else if(tprecond==PRECOND_AMG){
			//std::cout<<"AMG PRECOND"<<std::endl;

			viennacl::linalg::amg_tag amg_tag_agg_pmis;
			amg_tag_agg_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION); //std::cout<<"COARSENING: MIS2 AGGREGATION"<<std::endl;
			if(tamginter==AMGINTER_AG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION);} //std::cout<<"INTERPOLATION: AGGREGATION "<<std::endl; }
			else if(tamginter==AMGINTER_SAG){ amg_tag_agg_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION);} //std::cout<<"INTERPOLATION: SMOOTHED AGGREGATION"<<std::endl; }
    
			amg_tag_agg_pmis.set_strong_connection_threshold(strongconnection);
			amg_tag_agg_pmis.set_jacobi_weight(jacobiweight);
			amg_tag_agg_pmis.set_presmooth_steps(presmooth);
			amg_tag_agg_pmis.set_postsmooth_steps(postsmooth);
			amg_tag_agg_pmis.set_coarsening_cutoff(numfreesurface);
      //amg_tag_agg_pmis.set_coarse_levels(coarselevels); 
			viennacl::linalg::amg_precond<viennacl::compressed_matrix<double> > vcl_AMG(vcl_A_cuda, amg_tag_agg_pmis);
			//std::cout << " * Setup phase (ViennaCL types)..." << std::endl;
			viennacl::tools::timer timer; 
			timer.start();
			vcl_AMG.setup();
			//std::cout << "levels = " << vcl_AMG.levels() << "\n";
			//for(int i =0; i< vcl_AMG.levels();i++) std::cout << "level " << i << "\t" << "size = " << vcl_AMG.size(i) << "\n";
			viennacl::backend::finish();
			//std::cout << "  > Setup time: " << timer.get() << std::endl;
			run_solver(vcl_A_cuda,vcl_vec,bicgstab,vcl_AMG,vcl_result);
		}
	}
	else std::cout << "norm(b)=0" << std::endl;
}

//------------------------------------------------------------------------------
/// Shifting
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode> __device__ void KerInteractionForcesShifting2
  (bool boundp2,unsigned p1,const unsigned &pini,const unsigned &pfin,float visco,const float *ftomassp
  ,const double2 *posxy,const double *posz,double3 *velrhop,const word *code
  ,float massp2,float ftmassp1,bool ftp1
  ,double3 posdp1,double3 velp1
  ,TpShifting tshifting,double3 &shiftposp1,double Wab1,const double tensilen, const float tensiler,float &divrp1,double3 &sumtensilep1,const float *divr,const float freesurface,const float boundaryfs,double3 &dwxp1,double3 &dwyp1,double3 &dwzp1,const double PistonPos,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure)
{
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
    for(int count=0;count<=5;count++){
			bool interact=false;
			double3 velp2;
			double pressp2;
			double NeumannDist=0;
			KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
			if(interact){
				double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
					//-Wendland kernel.
					double frx,fry,frz,Wab;
					if(tker==KERNEL_Quintic){
						KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
						Wab=KerGetKernelQuinticWab(rr2);
					}
					else if(tker==KERNEL_Wendland){
				//		KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);
					//	Wab=KerGetKernelWendlandWab(rr2);
					}

					//-Shifting correction
					const double volume=massp2/CTE.rhopzero;
					const double tensile=tensiler*pow(Wab/Wab1,tensilen);
					shiftposp1.x+=volume*frx; //-For boundary do not use shifting / Con boundary anula shifting.
					shiftposp1.y+=volume*fry;
					shiftposp1.z+=volume*frz;
					sumtensilep1.x+=volume*tensile*frx;
					sumtensilep1.y+=volume*tensile*fry;
					sumtensilep1.z+=volume*tensile*frz;
					divrp1-=volume*(drx*frx+dry*fry+drz*frz);
					dwxp1.x-=volume*frx*drx; dwxp1.y-=volume*frx*dry; dwxp1.z-=volume*frx*drz;
					dwyp1.x-=volume*fry*drx; dwyp1.y-=volume*fry*dry; dwyp1.z-=volume*fry*drz;
					dwzp1.x-=volume*frz*drx; dwzp1.y-=volume*frz*dry; dwzp1.z-=volume*frz*drz;
				}
			}
		}
  }
}

template<TpKernel tker,TpFtMode ftmode> __global__ void KerInteractionForcesShifting1
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,float viscob,float viscof
  ,const int2 *begincell,int3 cellzero,const unsigned *dcell,const float *ftomassp
  ,const double2 *posxy,const double *posz,double3 *velrhop,const word *code
  ,TpShifting tshifting,double3 *shiftpos,float *divr,const float tensilen,const float tensiler,double3 *sumtensile,const float freesurface,const float boundaryfs,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg,const double PistonPos,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
    double Wab1;
		if(tker==KERNEL_Quintic) Wab1=KerGetKernelQuinticWab(CTE.dp*CTE.dp);
		//else if(tker==KERNEL_Wendland) Wab1=KerGetKernelWendlandWab(CTE.dp*CTE.dp);

    //-Vars para Shifting.
		//-Variables for Shifting.
    double3 shiftposp1=make_double3(0,0,0);
		double3 sumtensilep1=make_double3(0,0,0);
    float  divrp1=0;
		double3 dwxp1=make_double3(0,0,0); double3 dwyp1=make_double3(0,0,0); double3 dwzp1=make_double3(0,0,0);
    //-Obtiene datos de particula p1 en caso de existir floatings.
	//-Obtains data of particle p1 in case there are floating bodies.
    bool ftp1;       //-Indica si es floating. //-Indicates if it is floating.
    float ftmassp1;  //-Contiene masa de particula floating o 1.0f si es fluid. //-Contains floating particle mass or 1.0f if it is fluid.

    //-Obtiene datos basicos de particula p1.
	//-Obtains basic data of particle p1.
    double3 posdp1;
    double3 velp1;
    KerGetParticleData(p1,posxy,posz,velrhop,velp1,posdp1);

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
		      KerInteractionForcesShifting2<tker,ftmode> (false,p1,pini,pfin,viscob,ftomassp,posxy,posz,velrhop,code,CTE.massf,ftmassp1,ftp1,posdp1,velp1,tshifting,shiftposp1,Wab1,tensilen,tensiler,divrp1,sumtensilep1,divr,freesurface,boundaryfs,dwxp1,dwyp1,dwzp1,PistonPos,PistonVel,RightWall,gravity,pressure);
        }
	    }
    }

    /*if(shiftposp1.x||shiftposp1.y||shiftposp1.z||divrp1){
      shiftpos[Correctp1].x=shiftposp1.x; shiftpos[Correctp1].y=shiftposp1.y; shiftpos[Correctp1].z=shiftposp1.z;
      shiftposp1=make_float3(0,0,0);
      divr[p1]=divrp1;
			divrp1=0;
			sumtensile[Correctp1].x=sumtensilep1.x; sumtensile[Correctp1].y=sumtensilep1.y; sumtensile[Correctp1].z=sumtensilep1.z;
			sumtensilep1=make_float3(0,0,0);
			dwxcorrg[Correctp1].x=float(dwxp1.x); dwxcorrg[Correctp1].y=float(dwxp1.y); dwxcorrg[Correctp1].z=float(dwxp1.z); 
			dwycorrg[Correctp1].x=float(dwyp1.x); dwycorrg[Correctp1].y=float(dwyp1.y); dwycorrg[Correctp1].z=float(dwyp1.z); 
			dwzcorrg[Correctp1].x=float(dwzp1.x); dwzcorrg[Correctp1].y=float(dwzp1.y); dwzcorrg[Correctp1].z=float(dwzp1.z); 
			dwxp1=make_double3(0,0,0); dwyp1=make_double3(0,0,0); dwzp1=make_double3(0,0,0);
    }*/

    //-Interaccion con contorno.
	//-Interaction with boundaries.
   /* for(int z=zini;z<zfin;z++){
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
		      KerInteractionForcesShifting2<tker,ftmode> (true,p1,pini,pfin,viscof,ftomassp,posxy,posz,velrhop,code,CTE.massf,ftmassp1,ftp1,posdp1,velp1,tshifting,shiftposp1,Wab1,tensilen,tensiler,divrp1,sumtensilep1,divr,freesurface,boundaryfs,dwxp1,dwyp1,dwzp1);
        }
      }
    }*/
    
    if(shiftposp1.x||shiftposp1.y||shiftposp1.z||divrp1){
			shiftpos[Correctp1].x+=shiftposp1.x; shiftpos[Correctp1].y+=shiftposp1.y; shiftpos[Correctp1].z+=shiftposp1.z;
      divr[p1]+=divrp1;
			sumtensile[Correctp1].x=sumtensilep1.x; sumtensile[Correctp1].y+=sumtensilep1.y; sumtensile[Correctp1].z+=sumtensilep1.z;
			dwxcorrg[Correctp1].x+=dwxp1.x; dwxcorrg[Correctp1].y+=dwxp1.y; dwxcorrg[Correctp1].z+=dwxp1.z; 
			dwycorrg[Correctp1].x+=dwyp1.x; dwycorrg[Correctp1].y+=dwyp1.y; dwycorrg[Correctp1].z+=dwyp1.z; 
			dwzcorrg[Correctp1].x+=dwzp1.x; dwzcorrg[Correctp1].y+=dwzp1.y; dwzcorrg[Correctp1].z+=dwzp1.z; 
		}
  }
}

void Interaction_Shifting
  (TpKernel tkernel,TpSlipCond tslipcond,bool simulate2d,bool floating,bool usedem,TpCellMode cellmode,float viscob,float viscof,unsigned bsfluid,unsigned bsbound
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells
  ,const int2 *begincell,tuint3 cellmin,const unsigned *dcell
  ,const double2 *posxy,const double *posz
  ,double3 *velrhop,const word *code,const float *ftomassp
  ,TpShifting tshifting,double3 *shiftpos,float *divr,const float tensilen,const float tensiler
	,double3 *sumtensile,const float freesurface,const float boundaryfs,const unsigned *idp
	,const double3 *mirrorPos,const unsigned *mirrorCell,double3 *dwxcorrg,double3 *dwycorrg,double3 *dwzcorrg,double4 *mls,unsigned *row,const double pistonposx,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure)
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
		dim3 sgridb=GetGridSize(npb,bsbound);

		if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
			if(!floating)   KerInteractionForcesShifting1<tker,FTMODE_None> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			else if(!usedem)KerInteractionForcesShifting1<tker,FTMODE_Sph> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			else            KerInteractionForcesShifting1<tker,FTMODE_Dem> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			
			if(simulate2d) KerInverseKernelCor2D <<<sgridf,bsfluid>>> (npf,npb,dwxcorrg,dwzcorrg,code);
		}
		else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
			if(!floating)   KerInteractionForcesShifting1<tker,FTMODE_None> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			else if(!usedem)KerInteractionForcesShifting1<tker,FTMODE_Sph> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			else            KerInteractionForcesShifting1<tker,FTMODE_Dem> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,viscob,viscof,begincell,cellzero,dcell,ftomassp,posxy,posz,velrhop,code,tshifting,shiftpos,divr,tensilen,tensiler,sumtensile,freesurface,boundaryfs,dwxcorrg,dwycorrg,dwzcorrg,pistonposx,PistonVel,RightWall,gravity,pressure);
			
			if(simulate2d) KerInverseKernelCor2D <<<sgridf,bsfluid>>> (npf,npb,dwxcorrg,dwzcorrg,code);
		}
	}
}

template<bool floating> __global__ void KerComputeShift
  (unsigned npf,unsigned npb,const double3 *shiftpos,word *code,double2 *movxy,double *movz)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle.
  if(p<npf){
    unsigned p1=p+npb;
		const unsigned Correctp1=p;
    const double3 rshiftpos=shiftpos[Correctp1];
    if(!floating || CODE_GetType(code[p1])==CODE_TYPE_FLUID){//-Particulas: Fluid //-Particles: Fluid
      double dx=rshiftpos.x;
      double dy=rshiftpos.y;
      double dz=rshiftpos.z;
      movxy[p1]=make_double2(dx,dy);
      movz[p1]=dz;
    }
  }
}

void ComputeShift(bool floating,const unsigned bsfluid,unsigned np,unsigned npb,const double3 *shiftpos
  ,word *code,double2 *movxy,double *movz)
{
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    if(floating)KerComputeShift<true>  <<<sgridf,bsfluid>>> (npf,npb,shiftpos,code,movxy,movz);
    else        KerComputeShift<false> <<<sgridf,bsfluid>>> (npf,npb,shiftpos,code,movxy,movz);
  }
}

template<TpKernel tker> __device__ void KerCalcShiftVelocity
  (const bool fluid,const unsigned &pini,const unsigned &pfin,const double2 *posxy,const double *posz,double3 posdp1
	,const double3 velp1,const double3 *velrhop,const double3 dwx,const double3 dwy,const double3 dwz,const float massp2,const float RhopZero,const word *code,const unsigned *idp
	,const float *divr,const float boundaryfs,double3 &gradvx,double3 &gradvy,double3 &gradvz,const double PistonPos,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure)
{
	float volumep2=massp2/RhopZero; //Volume of particle j
  for(int p2=pini;p2<pfin;p2++){
    double drx,dry,drz;
		for(int count=0;count<=5;count++){
			bool interact=false;
			double3 velp2;
			double pressp2;
			double NeumannDist=0;
			KerGetParticlesDr(count,p2,posxy,posz,interact,posdp1,velrhop[p2],pressure[p2],drx,dry,drz,velp2,pressp2,PistonPos,PistonVel,NeumannDist,RightWall,gravity);
			if(interact){
				double rr2=drx*drx+dry*dry+drz*drz;
				if(rr2<=CTE.fourh2 && rr2>=ALMOSTZERO){
					//-Wendland kernel.
						double frx,fry,frz;
						if(tker==KERNEL_Quintic) KerGetKernelQuintic(rr2,drx,dry,drz,frx,fry,frz);
						//else if(tker==KERNEL_Wendland) KerGetKernelWendland(rr2,drx,dry,drz,frx,fry,frz);

						const double temp_x=frx*dwx.x+fry*dwy.x+frz*dwz.x;
						const double temp_y=frx*dwx.y+fry*dwy.y+frz*dwz.y;
						const double temp_z=frx*dwx.z+fry*dwy.z+frz*dwz.z;

						double dvx=velp2.x-velp1.x, dvy=velp2.y-velp1.y, dvz=velp2.z-velp1.z;
						gradvx.x+=temp_x*dvx*volumep2; gradvx.y+=temp_y*dvx*volumep2; gradvx.z+=temp_z*dvx*volumep2;
						gradvy.x+=temp_x*dvy*volumep2; gradvy.y+=temp_y*dvy*volumep2; gradvy.z+=temp_z*dvy*volumep2;
						gradvz.x+=temp_x*dvz*volumep2; gradvz.y+=temp_y*dvz*volumep2; gradvz.z+=temp_z*dvz*volumep2;   
					}
			}
		}
  }
}

template<TpKernel tker> __global__ void KerFindShiftVelocity
  (unsigned n,unsigned pinit,int hdiv,uint4 nc,unsigned cellfluid,const int2 *begincell,int3 cellzero,const unsigned *dcell
  ,const double2 *posxy,const double *posz,double3 *velrhop,const double3 *dwxCorr,const double3 *dwyCorr,const double3 *dwzCorr,const float *divr,const word *code
  ,const unsigned *idp,const float boundaryfs,double3 *shiftpos,double3 *shiftvel,const double PistonPos,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
   
		//-Obtiene datos basicos de particula p1.
  	//-Obtains basic data of particle p1.
    double3 posdp1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
		double3 velp1=make_double3(velrhop[p1].x,velrhop[p1].y,velrhop[p1].z);
		double3 shift=make_double3(shiftpos[Correctp1].x,shiftpos[Correctp1].y,shiftpos[Correctp1].z);
		double3 gradvx=make_double3(0,0,0);
		double3 gradvy=make_double3(0,0,0);
		double3 gradvz=make_double3(0,0,0);
    //-Obtiene limites de interaccion
	  //-Obtains interaction limits
    int cxini,cxfin,yini,yfin,zini,zfin;
    KerGetInteractionCells(dcell[p1],hdiv,nc,cellzero,cxini,cxfin,yini,yfin,zini,zfin);
        
		for(int z=zini;z<zfin;z++){
			int zmod=(nc.w)*z+(cellfluid); //-Le suma donde empiezan las celdas de fluido. //-The sum showing where fluid cells start
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
					KerCalcShiftVelocity<tker> (1,pini,pfin,posxy,posz,posdp1,velp1,velrhop,dwxCorr[Correctp1],dwyCorr[Correctp1],dwzCorr[Correctp1],CTE.massf,CTE.rhopzero,code,idp,divr,boundaryfs,gradvx,gradvy,gradvz,PistonPos,PistonVel,RightWall,gravity,pressure);
				}
			}
		}

		shiftvel[Correctp1].x=velp1.x+gradvx.x*shift.x+gradvx.y*shift.y+gradvx.z*shift.z;
		shiftvel[Correctp1].y=velp1.y+gradvy.x*shift.x+gradvy.y*shift.y+gradvy.z*shift.z;
		shiftvel[Correctp1].z=velp1.z+gradvz.x*shift.x+gradvz.y*shift.y+gradvz.z*shift.z;
  }
}

template<bool wavegen> __global__ void KerCorrectShiftVelocity
  (unsigned n,unsigned pinit,double3 *velrhop,double3 *shiftvel,const double2 *posxy,const double dampingpoint,const double dampinglength)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Nº de la partícula //-NI of the particle
  if(p<n){
    unsigned p1=p+pinit;      //-Nº de particula. //-NI of particle
		const unsigned Correctp1=p;
		velrhop[p1].x=shiftvel[Correctp1].x;
		velrhop[p1].y=shiftvel[Correctp1].y;
		velrhop[p1].z=shiftvel[Correctp1].z;

		if(wavegen){
			const double xp1=posxy[p1].x;
			KerDampingZone(xp1,velrhop[p1],dampingpoint,dampinglength);
		}
  }
}

void CorrectShiftVelocity(const bool wavegen,TpKernel tkernel,TpCellMode cellmode,const unsigned bsbound,const unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const int2 *begincell,tuint3 cellmin
	,const unsigned *dcell,const double2 *posxy,const double *posz,double3 *velrhop,const double3 *dwxCorr,const double3 *dwyCorr,const double3 *dwzCorr
  ,const unsigned *idp,const float *divr,const word *code,const float boundaryfs,double3 *shiftpos,double3 *shiftvel,const double dampingpoint,const double dampinglength,const double PistonPos,const double PistonVel,const double RightWall,const tfloat3 gravity,const double *pressure){
  const unsigned npf=np-npb;
  const int hdiv=(cellmode==CELLMODE_H? 2: 1);
  const uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  const unsigned cellfluid=nc.w*nc.z+1;
  const int3 cellzero=make_int3(cellmin.x,cellmin.y,cellmin.z);
  //-Interaccion Fluid-Fluid & Fluid-Bound
  //-Interaction Fluid-Fluid & Fluid-Bound
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);

		if(tkernel==KERNEL_Quintic){    const TpKernel tker=KERNEL_Quintic;
			KerFindShiftVelocity<tker> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,boundaryfs,shiftpos,shiftvel,PistonPos,PistonVel,RightWall,gravity,pressure);
		}
		else if(tkernel==KERNEL_Wendland){    const TpKernel tker=KERNEL_Wendland;
			KerFindShiftVelocity<tker> <<<sgridf,bsfluid>>> (npf,npb,hdiv,nc,cellfluid,begincell,cellzero,dcell,posxy,posz,velrhop,dwxCorr,dwyCorr,dwzCorr,divr,code,idp,boundaryfs,shiftpos,shiftvel,PistonPos,PistonVel,RightWall,gravity,pressure);
		}

		if(wavegen) KerCorrectShiftVelocity<true> <<<sgridf,bsfluid>>> (npf,npb,velrhop,shiftvel,posxy,dampingpoint,dampinglength);
		else KerCorrectShiftVelocity<false> <<<sgridf,bsfluid>>> (npf,npb,velrhop,shiftvel,posxy,dampingpoint,dampinglength);
	}
}

}