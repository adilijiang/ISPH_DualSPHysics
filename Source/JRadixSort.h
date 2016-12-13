/*
 <DUALSPHYSICS>  Copyright (c) 2016, Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JRadixSort.h \brief Declares the class  \ref JRadixSort.

#ifndef _JRadixSort_
#define _JRadixSort_


#include "JObject.h"
#include "TypesDef.h"


//##############################################################################
//# JRadixSort
//##############################################################################
/// \brief Implements the algorithm RadixSort.

class JRadixSort : protected JObject
{
private:
  static const int OMPSTRIDE=200;
  static const int OMPSIZE=1024;

  const bool UseOmp;

  bool Type32;
  unsigned *InitData32;
  ullong *InitData64;

  unsigned *Index;
  unsigned *PrevIndex;
  
  static const int KEYSBITS=8;
  static const int KEYSRANGE=256;
  static const int KEYSMASK=0xff;

  unsigned Size;
  unsigned Nbits;
  unsigned Nkeys;

  unsigned *Data32;
  ullong *Data64;
  unsigned *PrevData32;
  ullong *PrevData64;

  unsigned *BeginKeys;

  void AllocMemory(unsigned s);
  template<class T> void LoadBeginKeys(const T* data);

  template<class T> unsigned TBitsSize(T v,unsigned smax)const;
  template<class T> unsigned TCalcNbits(unsigned size,const T *data)const;
  template<class T> void SortStep(unsigned ck,const T* data,T* data2);
  template<class T> void SortStepIndex(unsigned ck,const T* data,T* data2,const unsigned *index,unsigned *index2);

  template<class T> void TSortData(unsigned size,const T *data,T *result);

  void IndexCreate();

public:
  JRadixSort(bool useomp);
  ~JRadixSort();
  void Reset();

  void Sort(bool makeindex,unsigned size,unsigned *data,unsigned nbits);
  void Sort(bool makeindex,unsigned size,ullong *data,unsigned nbits);

  void Sort(bool makeindex,unsigned size,unsigned *data){ Sort(makeindex,size,data,CalcNbits(size,data)); }
  void Sort(bool makeindex,unsigned size,ullong *data){ Sort(makeindex,size,data,CalcNbits(size,data)); }

  void MakeIndex(unsigned size,const unsigned *data){ MakeIndex(size,data,CalcNbits(size,data)); }
  void MakeIndex(unsigned size,const ullong *data){ MakeIndex(size,data,CalcNbits(size,data)); }

  void MakeIndex(unsigned size,const unsigned *data,unsigned nbits);
  void MakeIndex(unsigned size,const ullong *data,unsigned nbits);


  unsigned BitsSize(unsigned v)const;
  unsigned BitsSize(ullong v)const;

  unsigned CalcNbits(unsigned size,const unsigned *data)const;
  unsigned CalcNbits(unsigned size,const ullong *data)const;

  void SortData(unsigned size,const byte *data,byte *result);
  void SortData(unsigned size,const word *data,word *result);
  void SortData(unsigned size,const unsigned *data,unsigned *result);
  void SortData(unsigned size,const int *data,int *result);
  void SortData(unsigned size,const float *data,float *result);
  void SortData(unsigned size,const double *data,double *result);
  void SortData(unsigned size,const tfloat2 *data,tfloat2 *result);
  void SortData(unsigned size,const tfloat3 *data,tfloat3 *result);
  void SortData(unsigned size,const tfloat4 *data,tfloat4 *result);
  void SortData(unsigned size,const tdouble2 *data,tdouble2 *result);
  void SortData(unsigned size,const tdouble3 *data,tdouble3 *result);

  void DgCheckResult32()const;
  void DgCheckResult64()const;
};

#endif

