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

/// \file JMeanValues.h \brief Declares the class \ref JMeanValue and class \ref JMeanMoving.

#ifndef _JMeanValues_
#define _JMeanValues_


#include "JObject.h"
#include "TypesDef.h"

//##############################################################################
//# JMeanValue
//##############################################################################
/// \brief Calculates the average value of a sequence of values.

class JMeanValue
{
public:
  double Mean;
  ullong Values;

public:
  JMeanValue():Mean(0),Values(0){ }
  void Reset(){ Mean=0; Values=0; }
  void AddValue(double v){ Mean=(Mean*Values+v)/(Values+1); Values++; }
  double GetMean()const{ return(Mean); }
  ullong GetValues()const{ return(Values); }
};


//##############################################################################
//# JMeanMoving
//##############################################################################
/// \brief Calculates the mobile and weighted average value of a sequence of values.

class JMeanMoving : protected JObject
{
public:
protected:
  unsigned SizeValues;
  double *Values;
  double *Weights;
  unsigned NextValue;
  bool ValuesFull;      ///< Array of values is full.

  void Init(unsigned size,bool weighted);

public:
  JMeanMoving(unsigned size=10);
  ~JMeanMoving();
  void Reset();
  void InitSimple(unsigned size);
  void InitWeightedLinear(unsigned size);
  void InitWeightedExponential(unsigned size,float fac=1);

  void AddValue(double v);
  double GetSimpleMean()const;
  double GetWeightedMean()const;

};


#endif



