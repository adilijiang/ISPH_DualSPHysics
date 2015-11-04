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

/// \file JObjectGpu.h \brief Declares the class \ref JObjectGpu.

#ifndef _JObjectGpu_
#define _JObjectGpu_

#include "JObject.h"
#include <cuda_runtime_api.h>
#include <string>

//##############################################################################
//# JObject
//##############################################################################
/// \brief Defines objects with methods that throws exceptions for interaction with gpu.
class JObjectGpu : protected JObject
{
protected:
  void RunExceptionCuda(const std::string &method,const std::string &msg,cudaError_t error)const;
  void CheckCudaError(const std::string &method,const std::string &msg)const;
public:  
  JObjectGpu(){ ClassName="JObjectGpu"; } ///<Constructor.
};

#endif


