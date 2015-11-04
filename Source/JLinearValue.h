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

#ifndef _JLinearValue_
#define _JLinearValue_

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Clase obtener valores interpolados linealmente de una lista. (05/01/2014)
//# - Nuevo metodo LoadFile() para cargar valores de un fichero. (20/01/2014)
//# - Se eliminaron includes innecesarios. (01/05/2014)
//# EN:
//# Changes:
//# =========
//# - Class obtains linearly interpolated values from a list. (05.01.2014)
//# - New method LoadFile() to load values of a file. (20.01.2014)
//# - Removed unnecessary inclusions. (05.01.2014)
//#############################################################################

#include "JObject.h"
#include "TypesDef.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>


//==============================================================================
//##############################################################################
//==============================================================================
class JLinearValue : protected JObject
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

public:
  JLinearValue();
  ~JLinearValue();
  void Reset();
  unsigned GetAllocMemory()const;

  void SetSize(unsigned size);
  unsigned GetSize()const{ return(Size); }

  void AddTimeValue(double time,double value);
  unsigned GetCount()const{ return(Count); }

  double GetValue(double timestep);

  void LoadFile(std::string file);
  std::string GetFile()const{ return(File); };
};

#endif


