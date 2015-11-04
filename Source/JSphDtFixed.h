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

#ifndef _JSphDtFixed_
#define _JSphDtFixed_

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Gestiona el uso de un dt fijo (en milisegundos) a partir de los valores 
//#   para determinados instantes en segundos, interpolando los valores 
//#   intermedios. (10/11/2012)
//# - Los datos float pasaron a double. (28/11/2013) 
//# - EN:
//# Changes:
//# =========
//# - Manage the use of a fixed dt (in milliseconds) from the values
//# for certain moments in seconds, by interpolating intermediate values (10/11/2012)
//# - The float data became double. (11.28.2013)
//#############################################################################

#include "JObject.h"
#include "Types.h"
#include "JTimer.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

//##############################################################################
//# JSphDtFixed
//##############################################################################
/// \brief Manages the use of prefixed values of DT loaded from an input file.

class JSphDtFixed : protected JObject
{
protected:
  static const unsigned SIZEMAX=100000;
  static const unsigned SIZEINITIAL=500;

  std::string File;
  unsigned Size;
  unsigned Count;
  unsigned Position;
  double *Times;
  double *Values;
  double DtError; //- max(DtFixed-DtVariable)

  void Resize(unsigned size);

public:
  JSphDtFixed();
  ~JSphDtFixed();
  void Reset();
  unsigned GetAllocMemory()const;
  void LoadFile(std::string file);
  double GetDt(double timestep,double dtvar);
  double GetDtError(bool reset);
  std::string GetFile()const{ return(File); };
};

#endif


