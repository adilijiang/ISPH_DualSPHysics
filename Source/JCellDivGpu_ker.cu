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

#include "JCellDivGpu_ker.h"
#include "Types.h"
#include <float.h>
#include <cmath>
#include "JLog2.h"
//#include "JDgKerPrint.h"
//#include "JDgKerPrint_ker.h"

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace cudiv{

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para un warp de KerPosLimitsRedu.
/// Reduction of shared memory values for a warp of KerPosLimitsRedu.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerUintLimitsWarpRedu(volatile unsigned* sp1,volatile unsigned* sp2,const unsigned &tid){
  if(blockSize>=64){
    const unsigned tid2=tid+32;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=32){
    const unsigned tid2=tid+16;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=16){
    const unsigned tid2=tid+8;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=8){
    const unsigned tid2=tid+4;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=4){
    const unsigned tid2=tid+2;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=2){
    const unsigned tid2=tid+1;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para KerPosLimits.
/// Reduction of shared memory values for KerPosLimits
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerUintLimitsRedu(unsigned* sp1,unsigned* sp2,const unsigned &tid,unsigned* results){
  __syncthreads();
  if(blockSize>=512){ 
    if(tid<256){
      sp1[tid]=min(sp1[tid],sp1[tid+256]);
      sp2[tid]=max(sp2[tid],sp2[tid+256]);
    }
    __syncthreads(); 
  }
  if(blockSize>=256){ 
    if(tid<128){
      sp1[tid]=min(sp1[tid],sp1[tid+128]);
      sp2[tid]=max(sp2[tid],sp2[tid+128]);
    }
    __syncthreads(); 
  }
  if(blockSize>=128){ 
    if(tid<64){
      sp1[tid]=min(sp1[tid],sp1[tid+64]);
      sp2[tid]=max(sp2[tid],sp2[tid+64]);
    }
    __syncthreads(); 
  }
  if(tid<32)KerUintLimitsWarpRedu<blockSize>(sp1,sp2,tid);
  if(tid==0){
    const unsigned nblocks=gridDim.x*gridDim.y;
    unsigned cr=blockIdx.y*gridDim.x+blockIdx.x;
    results[cr]=sp1[0]; cr+=nblocks;
    results[cr]=sp2[0];
  }
}

//==============================================================================
/// Ordena valores usando RadixSort de thrust.
/// Reorders the values using RadixSort thrust
//==============================================================================
void Sort(unsigned* keys,unsigned* values,unsigned size,bool stable){
  if(size){
    thrust::device_ptr<unsigned> dev_keysg(keys);
    thrust::device_ptr<unsigned> dev_valuesg(values);
    if(stable)thrust::stable_sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
    else thrust::sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
  }
}

//==============================================================================
/// Devuelve tama�o de gridsize segun parametros.
/// Returns the dimensions of gridsize according to parameters
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize;//-Numero total de bloques a lanzar.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para un warp de KerPosLimitsRedu.
/// Reduction of values in shared memory for a warp of KerPosLimitsRedu.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerPosLimitsWarpRedu(volatile float* spx1,volatile float* spy1,volatile float* spz1,volatile float* spx2,volatile float* spy2,volatile float* spz2,const unsigned &tid){
  if(blockSize>=64){
    const unsigned tid2=tid+32;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=32){
    const unsigned tid2=tid+16;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=16){
    const unsigned tid2=tid+8;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=8){
    const unsigned tid2=tid+4;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=4){
    const unsigned tid2=tid+2;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=2){
    const unsigned tid2=tid+1;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para KerPosLimits.
/// Reduction of shared memory values for KerPosLimits.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerPosLimitsRedu(float* spx1,float* spy1,float* spz1,float* spx2,float* spy2,float* spz2,const unsigned &tid,float* results){
  __syncthreads();
  if(blockSize>=512){ 
    if(tid<256){
      spx1[tid]=min(spx1[tid],spx1[tid+256]); spy1[tid]=min(spy1[tid],spy1[tid+256]); spz1[tid]=min(spz1[tid],spz1[tid+256]);  
      spx2[tid]=max(spx2[tid],spx2[tid+256]); spy2[tid]=max(spy2[tid],spy2[tid+256]); spz2[tid]=max(spz2[tid],spz2[tid+256]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=256){ 
    if(tid<128){
      spx1[tid]=min(spx1[tid],spx1[tid+128]); spy1[tid]=min(spy1[tid],spy1[tid+128]); spz1[tid]=min(spz1[tid],spz1[tid+128]);  
      spx2[tid]=max(spx2[tid],spx2[tid+128]); spy2[tid]=max(spy2[tid],spy2[tid+128]); spz2[tid]=max(spz2[tid],spz2[tid+128]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=128){ 
    if(tid<64){
      spx1[tid]=min(spx1[tid],spx1[tid+64]); spy1[tid]=min(spy1[tid],spy1[tid+64]); spz1[tid]=min(spz1[tid],spz1[tid+64]);  
      spx2[tid]=max(spx2[tid],spx2[tid+64]); spy2[tid]=max(spy2[tid],spy2[tid+64]); spz2[tid]=max(spz2[tid],spz2[tid+64]);  
    }
    __syncthreads(); 
  }
  if(tid<32)KerPosLimitsWarpRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid);
  if(tid==0){
    const unsigned nblocks=gridDim.x*gridDim.y;
    unsigned cr=blockIdx.y*gridDim.x+blockIdx.x;
    results[cr]=spx1[0]; cr+=nblocks;
    results[cr]=spy1[0]; cr+=nblocks;
    results[cr]=spz1[0]; cr+=nblocks;
    results[cr]=spx2[0]; cr+=nblocks;
    results[cr]=spy2[0]; cr+=nblocks;
    results[cr]=spz2[0];
  }
}

//------------------------------------------------------------------------------
/// Calcula posicion minima y maxima a partir de los resultados de KerPosLimit.
/// Computes minimum and maximum position starting from the results of KerPosLimit.
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduPosLimits(unsigned n,float* data,float *results)
{
  extern __shared__ float spx1[];
  float *spy1=spx1+blockDim.x;
  float *spz1=spy1+blockDim.x;
  float *spx2=spz1+blockDim.x;
  float *spy2=spx2+blockDim.x;
  float *spz2=spy2+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de valor //-Value number
  //-Carga valores en memoria shared.
  //-Loads values in shared memory.
  unsigned p2=p;
  spx1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spy1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spz1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spx2[tid]=(p<n? data[p2]: -FLT_MAX); p2+=n;
  spy2[tid]=(p<n? data[p2]: -FLT_MAX); p2+=n;
  spz2[tid]=(p<n? data[p2]: -FLT_MAX);
  __syncthreads();
  //-Reduce valores de memoria shared.
  //-Reduction of values in shared memory.
  KerPosLimitsRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid,results);
}

//==============================================================================
/// ES:
/// Reduce los limites de posicion a partir de results[].
/// En results[] cada bloque graba xmin,ymin,zmin,xmax,ymax,zmax agrupando por
/// bloque.
/// - EN:
/// Reduction of position limits starting from results[].
/// In results[] each block stores xmin,ymin,zmin,xmax,ymax,zmax
/// grouped per block
//==============================================================================
void ReduPosLimits(unsigned nblocks,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(float)*6;
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  float *dat=aux;
  float *res=aux+(n_blocks*6);
  while(n>1){
    KerReduPosLimits<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    float* x=dat; dat=res; res=x;
  }
  float resf[6];
  cudaMemcpy(resf,dat,sizeof(float)*6,cudaMemcpyDeviceToHost);
  pmin=TFloat3(resf[0],resf[1],resf[2]);
  pmax=TFloat3(resf[3],resf[4],resf[5]);
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para un warp de KerLimitsCellRedu.
/// Reduction of shared memory values for a warp of KerLimitsCellRedu.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerLimitsCellWarpRedu(volatile unsigned* spx1,volatile unsigned* spy1,volatile unsigned* spz1,volatile unsigned* spx2,volatile unsigned* spy2,volatile unsigned* spz2,const unsigned &tid){
  if(blockSize>=64){
    const unsigned tid2=tid+32;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=32){
    const unsigned tid2=tid+16;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=16){
    const unsigned tid2=tid+8;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=8){
    const unsigned tid2=tid+4;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=4){
    const unsigned tid2=tid+2;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=2){
    const unsigned tid2=tid+1;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para KerLimitsCell.
/// Reduction of shared memory values for KerLimitsCell.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerLimitsCellRedu(unsigned cellcode,unsigned* spx1,unsigned* spy1,unsigned* spz1,unsigned* spx2,unsigned* spy2,unsigned* spz2,const unsigned &tid,unsigned* results){
  __syncthreads();
  if(blockSize>=512){ 
    if(tid<256){
      spx1[tid]=min(spx1[tid],spx1[tid+256]); spy1[tid]=min(spy1[tid],spy1[tid+256]); spz1[tid]=min(spz1[tid],spz1[tid+256]);  
      spx2[tid]=max(spx2[tid],spx2[tid+256]); spy2[tid]=max(spy2[tid],spy2[tid+256]); spz2[tid]=max(spz2[tid],spz2[tid+256]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=256){ 
    if(tid<128){
      spx1[tid]=min(spx1[tid],spx1[tid+128]); spy1[tid]=min(spy1[tid],spy1[tid+128]); spz1[tid]=min(spz1[tid],spz1[tid+128]);  
      spx2[tid]=max(spx2[tid],spx2[tid+128]); spy2[tid]=max(spy2[tid],spy2[tid+128]); spz2[tid]=max(spz2[tid],spz2[tid+128]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=128){ 
    if(tid<64){
      spx1[tid]=min(spx1[tid],spx1[tid+64]); spy1[tid]=min(spy1[tid],spy1[tid+64]); spz1[tid]=min(spz1[tid],spz1[tid+64]);  
      spx2[tid]=max(spx2[tid],spx2[tid+64]); spy2[tid]=max(spy2[tid],spy2[tid+64]); spz2[tid]=max(spz2[tid],spz2[tid+64]);  
    }
    __syncthreads(); 
  }
  if(tid<32)KerLimitsCellWarpRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid);
  if(tid==0){
    const unsigned nblocks=gridDim.x*gridDim.y;
    unsigned cr=blockIdx.y*gridDim.x+blockIdx.x;
    results[cr]=PC__Cell(cellcode,spx1[0],spy1[0],spz1[0]);  cr+=nblocks;
    results[cr]=PC__Cell(cellcode,spx2[0],spy2[0],spz2[0]);
  }
}

//------------------------------------------------------------------------------
/// Calcula posicion minima y maxima a partir de los resultados de KerPosLimit.
/// Computes minimum and maximum postion startibg from the results of KerPosLimit.
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerLimitsCellReduBase(unsigned cellcode,unsigned n,unsigned* data,unsigned *results)
{
  extern __shared__ unsigned scx1[];
  unsigned *scy1=scx1+blockDim.x;
  unsigned *scz1=scy1+blockDim.x;
  unsigned *scx2=scz1+blockDim.x;
  unsigned *scy2=scx2+blockDim.x;
  unsigned *scz2=scy2+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de valor //-Value number
  //-Carga valores en memoria shared.
  //-Loads values in shared memory.
  unsigned p2=p;
  const unsigned celmin=(p<n? data[p2]: UINT_MAX);  p2+=n;
  const unsigned celmax=(p<n? data[p2]: 0);
  scx1[tid]=PC__Cellx(cellcode,celmin);
  scy1[tid]=PC__Celly(cellcode,celmin);
  scz1[tid]=PC__Cellz(cellcode,celmin);
  scx2[tid]=PC__Cellx(cellcode,celmax);
  scy2[tid]=PC__Celly(cellcode,celmax);
  scz2[tid]=PC__Cellz(cellcode,celmax);
  __syncthreads();
  //-Reduce valores de memoria shared.
  //-Reduction of shared memory values.
  KerLimitsCellRedu<blockSize>(cellcode,scx1,scy1,scz1,scx2,scy2,scz2,tid,results);
}

//==============================================================================
/// ES:
/// Reduce los limites de celdas a partir de results[].
/// En results[] cada bloque graba cxmin,cymin,czmin,cxmax,cymax,czmax codificando
/// los valores como celdas en 2 unsigned y agrupando por bloque.
/// - EN:
/// Reduction of cell limits starting from results[].
/// In results[] each block stores cxmin,cymin,czmin,cxmax,cymax,czmax encodes
/// the values as cells in 2 unsigned and groups them per block.
//==============================================================================
void LimitsCellRedu(unsigned cellcode,unsigned nblocks,unsigned *aux,tuint3 &celmin,tuint3 &celmax,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(float)*6;
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  unsigned *dat=aux;
  unsigned *res=aux+(n_blocks*2); //value min y max. //minimum and maximum value
  while(n>1){
    KerLimitsCellReduBase<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(cellcode,n,dat,res);
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    unsigned* x=dat; dat=res; res=x;
  }
  unsigned resf[6];
  cudaMemcpy(resf,dat,sizeof(unsigned)*2,cudaMemcpyDeviceToHost);
  celmin=TUint3(PC__Cellx(cellcode,resf[0]),PC__Celly(cellcode,resf[0]),PC__Cellz(cellcode,resf[0]));
  celmax=TUint3(PC__Cellx(cellcode,resf[1]),PC__Celly(cellcode,resf[1]),PC__Cellz(cellcode,resf[1]));
}

//------------------------------------------------------------------------------
/// ES:
/// Calcula celda minima y maxima de las particulas validas.
/// Ignora las particulas con check[p]!=0 
/// Las particulas fuera del dominio ya estan marcadas con check[p]=CHECK_OUTPOS
/// Si rhop!=NULL y la particula esta fuera del rango permitido (rhopmin,rhopmax)
/// se marca check[p]=CHECK_OUTRHOP
/// En results[] cada bloque graba cxmin,cymin,czmin,cxmax,cymax,czmax codificando
/// los valores como celdas en 2 unsigned y agrupando por bloque.
/// En caso de no haber ninguna particula valida el minimo sera mayor que el maximo.
/// - EN:
/// Computes minimu and maximum cells for valid particles.
/// Ignores the particles with check[p]!=0.
/// The particles outside of the domain are marked with check[p]=CHECK_OUTPOS.
/// If rhop!=NULL and the particle is outside the allowed range (rhopmin,rhopmax),
/// it is marked with check[p]=CHECK_OUTRHOP.
/// In results[], each block stores cxmin,cymin,czmin,cxmax,cymax,czmax encodes
/// the values as cells in 2 unsigned and groups them per block.
/// In case of having no valid particles the minimum value igreater than the maximum.
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerLimitsCell(unsigned n,unsigned pini,unsigned cellcode,const unsigned *dcell,const word *code,unsigned *results)
{
  extern __shared__ unsigned scx1[];
  unsigned *scy1=scx1+blockDim.x;
  unsigned *scz1=scy1+blockDim.x;
  unsigned *scx2=scz1+blockDim.x;
  unsigned *scy2=scx2+blockDim.x;
  unsigned *scz2=scy2+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula //-Particle number
  //-Carga valores en memoria shared. 
  //-Loads shared memory values.
  if(p<n){
    const unsigned pp=p+pini;
    unsigned rcell=dcell[pp];
    const unsigned cx=PC__Cellx(cellcode,rcell);
    const unsigned cy=PC__Celly(cellcode,rcell);
    const unsigned cz=PC__Cellz(cellcode,rcell);
    if(CODE_GetSpecialValue(code[pp])<CODE_OUTIGNORE){ //-Particula no excluida. //-Excluded particles
      scx1[tid]=cx; scy1[tid]=cy; scz1[tid]=cz;
      scx2[tid]=cx; scy2[tid]=cy; scz2[tid]=cz;
    }
    else{
      scx1[tid]=UINT_MAX; scy1[tid]=UINT_MAX;  scz1[tid]=UINT_MAX;
      scx2[tid]=0;        scy2[tid]=0;         scz2[tid]=0;
    }
  }
  else{
    scx1[tid]=UINT_MAX; scy1[tid]=UINT_MAX;  scz1[tid]=UINT_MAX;
    scx2[tid]=0;        scy2[tid]=0;         scz2[tid]=0;
  }
  __syncthreads();
  //-Reduce valores de memoria shared.
  //-Reduction of shared memory values.
  KerLimitsCellRedu<blockSize>(cellcode,scx1,scy1,scz1,scx2,scy2,scz2,tid,results);
}

//==============================================================================
/// ES:
/// Calcula celda minima y maxima de las particulas validas.
/// Ignora las particulas excluidas con code[p].out!=CODE_OUT_OK 
/// En results[] cada bloque graba cxmin,cymin,czmin,cxmax,cymax,czmax codificando
/// los valores como celdas en 2 unsigned y agrupando por bloque.
/// En caso de no haber ninguna particula valida el minimo sera mayor que el maximo.
/// - EN:
/// Computes minimun and maximum cell for valid particles.
/// Ignores excluded particles with code[p].out!=CODE_OUT_OK
/// In results[], each block stores cxmin,cymin,czmin,cxmax,cymax,czmax encodes
/// the values as cells in 2 unsigned and groups them per block.
/// In case of having no valid particles the minimum value igreater than the maximum.
//==============================================================================
void LimitsCell(unsigned np,unsigned pini,unsigned cellcode,const unsigned *dcell,const word *code,unsigned *aux,tuint3 &celmin,tuint3 &celmax,JLog2 *log){
  if(!np){//-Si no hay particulas cancela proceso.
    celmin=TUint3(1);
    celmax=TUint3(0);
    return;
  }
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned)*6;
  dim3 sgrid=GetGridSize(np,DIVBSIZE);
  unsigned nblocks=sgrid.x*sgrid.y;
  KerLimitsCell<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,cellcode,dcell,code,aux);
  LimitsCellRedu(cellcode,nblocks,aux,celmin,celmax,log);
}

//------------------------------------------------------------------------------
/// Calcula particula inicial y final de cada celda.
/// Compute first and last particle for each cell.
//------------------------------------------------------------------------------
__global__ void KerCalcBeginEndCell(unsigned n,unsigned pini,const unsigned *cellpart,int2 *begcell)
{
  extern __shared__ unsigned scell[];    // [blockDim.x+1}
  const unsigned pt=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula //-Particle number
  const unsigned p=pt+pini;
  unsigned cel;
  if(pt<n){
    cel=cellpart[p];
    scell[threadIdx.x+1]=cel;
    if(pt&&!threadIdx.x)scell[0]=cellpart[p-1];
  }
  __syncthreads();
  if(pt<n){
    if(!pt||cel!=scell[threadIdx.x]){
      begcell[cel].x=p;
      if(pt)begcell[scell[threadIdx.x]].y=p;
    }
    if(pt==n-1)begcell[cel].y=p+1;
  }
}

//==============================================================================
/// Calcula particula inicial y final de cada celda.
/// Compute first and last particle for each cell.
//==============================================================================
void CalcBeginEndCell(bool full,unsigned np,unsigned npb,unsigned sizebegcell,unsigned cellfluid,const unsigned *cellpart,int2 *begcell){
  if(full)cudaMemset(begcell,0,sizeof(int2)*sizebegcell);
  else cudaMemset(begcell+cellfluid,0,sizeof(int2)*(sizebegcell-cellfluid));
  const unsigned pini=(full? 0: npb);
  const unsigned n=np-pini;
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerCalcBeginEndCell <<<sgrid,DIVBSIZE,sizeof(unsigned)*(DIVBSIZE+1)>>> (n,pini,cellpart,begcell);
  }
}

//------------------------------------------------------------------------------
/// Reordena datos de particulas segun idsort[]
/// Reorders particle data according to idsort[]
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const unsigned *idp,const word *code,const unsigned *dcell,const double2 *posxy,const double *posz,const float4 *velrhop,unsigned *idp2,word *code2,unsigned *dcell2,double2 *posxy2,double *posz2,float4 *velrhop2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    idp2[p]=idp[oldpos];
    code2[p]=code[oldpos];
    dcell2[p]=dcell[oldpos];
    posxy2[p]=posxy[oldpos];
    posz2[p]=posz[oldpos];
    velrhop2[p]=velrhop[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float4 *a,float4 *a2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float *a,const float *b,float *a2,float *b2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
    b2[p]=b[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const double2 *a,const double *b,const float4 *c,double2 *a2,double *b2,float4 *c2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
    b2[p]=b[oldpos];
    c2[p]=c[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de la part�cula
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
  }
}

//==============================================================================
/// Reordena datos de particulas segun sortpart.
/// Reorders particle data according to sortpart.
//==============================================================================
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const unsigned *idp,const word *code,const unsigned *dcell,const double2 *posxy,const double *posz,const float4 *velrhop,unsigned *idp2,word *code2,unsigned *dcell2,double2 *posxy2,double *posz2,float4 *velrhop2){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(np,pini,sortpart,idp,code,dcell,posxy,posz,velrhop,idp2,code2,dcell2,posxy2,posz2,velrhop2);
  }
}
//==============================================================================
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float4 *a,float4 *a2){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(np,pini,sortpart,a,a2);
  }
}
//==============================================================================
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float *a,const float *b,float *a2,float *b2){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(np,pini,sortpart,a,b,a2,b2);
  }
}
//==============================================================================
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const double2 *a,const double *b,const float4 *c,double2 *a2,double *b2,float4 *c2){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(np,pini,sortpart,a,b,c,a2,b2,c2);
  }
}
//==============================================================================
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2){
  if(np){
    dim3 sgrid=GetGridSize(np,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(np,pini,sortpart,a,a2);
  }
}

//------------------------------------------------------------------------------
/// Calcula valores minimo y maximo a partir de data[].
/// Compute minimum and maximum values starting from data[].
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduUintLimits(unsigned n,unsigned* data,unsigned *results)
{
  extern __shared__ unsigned sp1[];
  unsigned *sp2=sp1+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de valor
  //-Carga valores en memoria shared.
  //-Loads variables to shared memory.
  unsigned p2=p;
  sp1[tid]=(p<n? data[p2]: UINT_MAX);  p2+=n;
  sp2[tid]=(p<n? data[p2]: 0);
  __syncthreads();
  //-Reduce valores de memoria shared.
  //-Reduction of shared memory values.
  KerUintLimitsRedu<blockSize>(sp1,sp2,tid,results);
}

//==============================================================================
/// ES:
/// Reduce los limites de valores unsigned a partir de results[].
/// En results[] cada bloque graba vmin,vmax agrupando por bloque.
/// EN:
/// Reduce the limits of unsigned values from results[].
/// In results[] each block stores vmin,vamx gropued per block.
//==============================================================================
void ReduUintLimits(unsigned nblocks,unsigned *aux,unsigned &vmin,unsigned &vmax,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned)*2;
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  unsigned *dat=aux;
  unsigned *res=aux+(n_blocks*2);
  while(n>1){
    KerReduUintLimits<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    unsigned* x=dat; dat=res; res=x;
  }
  unsigned resf[2];
  cudaMemcpy(resf,dat,sizeof(unsigned)*2,cudaMemcpyDeviceToHost);
  vmin=resf[0];
  vmax=resf[1];
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para un warp de KerReduUintSum.
/// Reduction of shared memory values for a warp of KerReduUintSum.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerUintSumWarpRedu(volatile unsigned* sp1,const unsigned &tid){
  if(blockSize>=64)sp1[tid]+=sp1[tid+32];
  if(blockSize>=32)sp1[tid]+=sp1[tid+16];
  if(blockSize>=16)sp1[tid]+=sp1[tid+ 8];
  if(blockSize>= 8)sp1[tid]+=sp1[tid+ 4];
  if(blockSize>= 4)sp1[tid]+=sp1[tid+ 2];
  if(blockSize>= 2)sp1[tid]+=sp1[tid+ 1];
}

//------------------------------------------------------------------------------
/// Reduccion de valores en memoria shared para KerReduUintSum.
/// Reduction of shared memory values for KerReduUintSum.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ void KerUintSumRedu(unsigned* sp1,const unsigned &tid,unsigned* results){
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sp1[tid]+=sp1[tid+256]; __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sp1[tid]+=sp1[tid+128]; __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sp1[tid]+=sp1[tid+64];  __syncthreads(); }
  if(tid<32)KerUintSumWarpRedu<blockSize>(sp1,tid);
  if(tid==0)results[blockIdx.y*gridDim.x+blockIdx.x]=sp1[0];
}

//------------------------------------------------------------------------------
/// Devuelve la suma de los valores contenidos en data[].
/// Returns the sum of the values contained in data[].
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduUintSum(unsigned n,unsigned* data,unsigned *results)
{
  extern __shared__ unsigned sp1[];
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-N� de valor //-Value number
  //-Carga valores en memoria shared.
  //-Loads values in shared memory.
  sp1[tid]=(p<n? data[p]: 0);
  __syncthreads();
  //-Reduce valores de memoria shared.
  //-Reduce shared memory values.
  KerUintSumRedu<blockSize>(sp1,tid,results);
}

//==============================================================================
/// Devuelve la suma de los valores contenidos en aux[].
/// Returns the sum of the values contained in aux[].
//==============================================================================
unsigned ReduUintSum(unsigned nblocks,unsigned *aux,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned);
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  unsigned *dat=aux;
  unsigned *res=aux+(n_blocks);
  while(n>1){
    KerReduUintSum<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    unsigned* x=dat; dat=res; res=x;
  }
  unsigned resf;
  cudaMemcpy(&resf,dat,sizeof(unsigned),cudaMemcpyDeviceToHost);
  return(resf);
}

}



