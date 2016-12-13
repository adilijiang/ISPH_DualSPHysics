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

/// \file JSphVisco.h \brief Declares the class \ref JSphVisco.

#ifndef _JSphVisco_
#define _JSphVisco_

#include "JObject.h"
#include "Types.h"
#include "JTimer.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

//##############################################################################
//# JSphVisco
//##############################################################################
/// \brief Manages the use of viscosity values from an input file.

class JSphVisco : protected JObject
{
protected:
  static const unsigned FILESIZEMAX=104857600; ///<Maximum file size (100mb).

  std::string File;
  unsigned Size;
  unsigned Count;
  unsigned Position;
  float *Times;
  float *Values;

  void Resize(unsigned size);

public:
  JSphVisco();
  ~JSphVisco();
  void Reset();
  unsigned GetAllocMemory()const;
  void LoadFile(std::string file);
  float GetVisco(float timestep);
  std::string GetFile()const{ return(File); };
};

#endif


