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


/// \file JSphMotion.h \brief Declares the class \ref JSphMotion.

#ifndef _JSphMotion_
#define _JSphMotion_

#include "TypesDef.h"
#include <string>

class JMotion;
class JXml;


//##############################################################################
//# JSphMotion
//##############################################################################
/// \brief Provides the displacement of moving objects during a time interval.  

class JSphMotion
{
private:
  JMotion *Mot;

public:

  //==============================================================================
  /// Constructor.
  //==============================================================================
  JSphMotion();

  //==============================================================================
  /// Destructor.
  //==============================================================================
  ~JSphMotion();

  //==============================================================================
  /// Initialisation of variables.
  //==============================================================================
  void Reset();

  //==============================================================================
  /// Initialisation of configuration and returns number of moving objects.
  //==============================================================================
  unsigned Init(JXml *jxml,const std::string &path,const std::string &dirdata);

  //==============================================================================
  /// Processes next time interval and returns true if there are active motions.
  //==============================================================================
  bool ProcesTime(float timestep,float dt){ return(ProcesTime(double(timestep),double(dt))); };
  bool ProcesTime(double timestep,double dt);

  //==============================================================================
  /// Returns the number of performed movements.
  //==============================================================================
  unsigned GetMovCount()const;

  //==============================================================================
  /// Returns data of the motion of an object.
  //==============================================================================
  bool GetMov(unsigned mov,unsigned &ref,tfloat3 &mvsimple,tmatrix4f &mvmatrix)const;
  bool GetMov(unsigned mov,unsigned &ref,tdouble3 &mvsimple,tmatrix4d &mvmatrix)const;

  //==============================================================================
  /// Returns the number of finished movements.
  //==============================================================================
  unsigned GetStopCount()const;

  //==============================================================================
  /// Returns the reference of the stopped object.
  //==============================================================================
  unsigned GetStopRef(unsigned mov)const;
};

#endif







