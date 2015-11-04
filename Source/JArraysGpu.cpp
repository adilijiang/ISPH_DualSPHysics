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

#include "JArraysGpu.h"
#include "Functions.h"
#include <cstdio>

using namespace std;

//##############################################################################
//# JArraysGpuSize
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JArraysGpuSize::JArraysGpuSize(unsigned elementsize):ElementSize(elementsize){
  ClassName="JArraysGpuSize";
  for(unsigned c=0;c<MAXPOINTERS;c++)Pointers[c]=NULL;
  Count=0;
  CountMax=CountUsedMax=0;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JArraysGpuSize::~JArraysGpuSize(){
  Reset();
}
 
//==============================================================================
/// Initialization of variables.
//==============================================================================
void JArraysGpuSize::Reset(){
  FreeMemory();
  ArraySize=0;
}

//==============================================================================
/// Libera memoria reservada.
/// Frees allocated memory.
//==============================================================================
void JArraysGpuSize::FreeMemory(){
  for(unsigned c=0;c<Count;c++)if(Pointers[c]){ cudaFree(Pointers[c]); Pointers[c]=NULL; }
  CountUsed=Count=0;
}

//==============================================================================
/// ES:
/// Cambia el numero de arrays almacenados. Asignando nuevos arrays o liberando
/// los de los actuales sin uso. 
/// Si count es inferior al numero de los que estan en uso lanza una excepcion.
/// - EN:
/// Changes the number of arrays stored. Assigns or releases new arrays if
/// the current are unused.
/// If the count is less than the number of those in use raises an exception.
//==============================================================================
void JArraysGpuSize::SetArrayCount(unsigned count){
  const char met[]="SetArrayCount";
  if(count>MAXPOINTERS)RunException(met,"El numero de arrays solicitados supera el maximo.");
  if(count<CountUsed)RunException(met,"No se pude liberar arrays en uso.");
  if(ArraySize){
    if(Count<count){//-Genera nuevos arrays. //-Generates new arrays
      for(unsigned c=Count;c<count;c++)cudaMalloc((void**)(Pointers+c),ElementSize*ArraySize);
      CheckCudaError(met,"Failed GPU memory allocation.");
    }
    if(Count>count){//-Libera arrays. //-Frees arrays
      for(unsigned c=count;c<Count;c++){ cudaFree(Pointers[c]); Pointers[c]=NULL; }
    }
  }
  Count=count;
  CountMax=max(CountMax,Count);
}

//==============================================================================
/// ES:
/// Cambia el numero de elementos de los arrays.
/// Si hay algun array en uso lanza una excepcion.
/// - EN:
/// Changes the number of elements in the arrays.
/// If there is any array in use raises an exception.
//==============================================================================
void JArraysGpuSize::SetArraySize(unsigned size){
  if(CountUsed)RunException("SetArraySize","No se puede cambiar la dimension de los arrays porque hay alguno en uso.");
  if(ArraySize!=size){
    ArraySize=size;
    unsigned count=Count;
    FreeMemory();
    if(count)SetArrayCount(count);
  }
}

//==============================================================================
/// Solicita la reserva de un array.
/// Requests allocating an array.
//==============================================================================
void* JArraysGpuSize::Reserve(){
  if(CountUsed==Count||!ArraySize)RunException("Reserve",fun::PrintStr("No hay arrays disponibles de %u bytes.",ElementSize));
  CountUsed++;
  CountUsedMax=max(CountUsedMax,CountUsed);
  return(Pointers[CountUsed-1]);
}

//==============================================================================
/// Devuelve la posicion del puntero indicado. Si no existe devuelve MAXPOINTERS.
/// Returns the position of indicated pointer. If it doesn't exist returns MAXPOINTERS.
//==============================================================================
unsigned JArraysGpuSize::FindPointerUsed(void *pointer)const{
  unsigned pos=0;
  for(;pos<CountUsed&&Pointers[pos]!=pointer;pos++);
  return(pos>=CountUsed? MAXPOINTERS: pos);
}

//==============================================================================
/// Libera la reserva de un array.
/// Frees an allocated array.
//==============================================================================
void JArraysGpuSize::Free(void *pointer){
  if(pointer){
    unsigned pos=FindPointerUsed(pointer);
    if(pos==MAXPOINTERS)RunException("Free","El puntero indicado no estaba reservado.");
    if(pos+1<CountUsed){
      void *aux=Pointers[CountUsed-1]; Pointers[CountUsed-1]=Pointers[pos]; Pointers[pos]=aux;
    }
    CountUsed--;
  }
}  


//##############################################################################
//# JArraysGpu
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JArraysGpu::JArraysGpu(){
  ClassName="JArraysGpu";
  Arrays1b=new JArraysGpuSize(1);
  Arrays2b=new JArraysGpuSize(2);
  Arrays4b=new JArraysGpuSize(4);
  Arrays8b=new JArraysGpuSize(8);
  Arrays12b=new JArraysGpuSize(12);
  Arrays16b=new JArraysGpuSize(16);
  Arrays24b=new JArraysGpuSize(24);
  Arrays32b=new JArraysGpuSize(32);
}

//==============================================================================
/// Destructor.
//==============================================================================
JArraysGpu::~JArraysGpu(){
  delete Arrays1b;
  delete Arrays2b;
  delete Arrays4b;
  delete Arrays8b;
  delete Arrays12b;
  delete Arrays16b;
  delete Arrays24b;
  delete Arrays32b;
}
 
//==============================================================================
/// Initialization of variables.
//==============================================================================
void JArraysGpu::Reset(){
  Arrays1b->Reset(); 
  Arrays2b->Reset(); 
  Arrays4b->Reset(); 
  Arrays8b->Reset(); 
  Arrays12b->Reset();
  Arrays16b->Reset();
  Arrays24b->Reset();
  Arrays32b->Reset();
}
 
//==============================================================================
/// Devuelve la cantidad de memoria reservada.
/// Returns amount of allocated memory.
//==============================================================================
llong JArraysGpu::GetAllocMemoryGpu()const{ 
  llong m=Arrays1b->GetAllocMemoryGpu();
  m+=Arrays2b->GetAllocMemoryGpu();
  m+=Arrays4b->GetAllocMemoryGpu();
  m+=Arrays8b->GetAllocMemoryGpu();
  m+=Arrays12b->GetAllocMemoryGpu();
  m+=Arrays16b->GetAllocMemoryGpu();
  m+=Arrays24b->GetAllocMemoryGpu();
  m+=Arrays32b->GetAllocMemoryGpu();
  return(m);
}

//==============================================================================
/// ES:
/// Cambia el numero de elementos de los arrays.
/// Si hay algun array en uso lanza una excepcion.
/// - EN:
/// Changes the number of elements in the arrays.
/// If there is any array in use raises an exception.
//==============================================================================
void JArraysGpu::SetArraySize(unsigned size){ 
  //-Frees memory.
  Arrays1b->SetArraySize(0); 
  Arrays2b->SetArraySize(0); 
  Arrays4b->SetArraySize(0); 
  Arrays8b->SetArraySize(0); 
  Arrays12b->SetArraySize(0);
  Arrays16b->SetArraySize(0);
  Arrays24b->SetArraySize(0);
  Arrays32b->SetArraySize(0);
  //-Allocates memory.
  Arrays1b->SetArraySize(size); 
  Arrays2b->SetArraySize(size); 
  Arrays4b->SetArraySize(size); 
  Arrays8b->SetArraySize(size); 
  Arrays12b->SetArraySize(size);
  Arrays16b->SetArraySize(size);
  Arrays24b->SetArraySize(size);
  Arrays32b->SetArraySize(size);
}


