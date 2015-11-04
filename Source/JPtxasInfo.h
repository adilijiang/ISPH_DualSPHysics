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

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Ahora tambien gestiona el SM_12, ademas de SM_10 y SM_20. (05/05/2011)
//# - Permite el uso de namespace. (08/01/2012)
//# - Compatibilidad con Cuda 4.0. (06/02/2012)
//# - Traduccion de comentarios al ingles. (10/02/2012)
//# - Se cambio el SM_12 por el SM_13. (28/04/2013)
//# - Ahora graba tambien Stack frame (memoria local). (19/08/2014)
//# - Solo graba los SM que tengan datos. (19/08/2014)
//# - Nuevo metodo GetKernelIdx(). (20/08/2014)
//# - Se anadio el SM_35. (01/09/2014)
//# - EN:
//# Changes:
//# =========
// # - Now also manages the SM_12, plus SM_10 and SM_20. (05/05/2011)
// # - Allows the use of namespace. (08/01/2012)
// # - Supports Cuda 4.0. (06/02/2012)
// # - Comment on English translation. (10/02/2012)
// # - Changed the SM_12 for SM_13. (28/04/2013)
// # - Now also records Stack frame (local memory). (19.08.2014)
// # - Only records SMs that have data. (19.08.2014)
// # - New method GetKernelIdx (). (20.08.2014)
// # - The SM_35 is added. (09.01.2014)
//#############################################################################

/// \file JPtxasInfo.h \brief Declares the class \ref JPtxasInfo.

#ifndef _JPtxasInfo_
#define _JPtxasInfo_

#include "JObject.h"
#include "TypesDef.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

class JPtxasInfoKer : protected JObject
{
public:
  typedef struct{
    std::string type;
    std::string value;
  }StTemplateArg; 

protected:
  std::string Code,CodeMin;
  std::string NameSpace;
  std::string Name;
  std::string Args;

  std::vector<StTemplateArg*> TemplateArgs;  

  unsigned Regs_sm10; ///<Number of registers according to compute capability 1.0 (0:No info).
  unsigned Regs_sm13; ///<Number of registers according to compute capability 1.3 (0:No info).
  unsigned Regs_sm20; ///<Number of registers according to compute capability 2.0 (0:No info).
  unsigned Regs_sm30; ///<Number of registers according to compute capability 3.0 (0:No info).
  unsigned Regs_sm35; ///<Number of registers according to compute capability 3.5 (0:No info).

  unsigned StackFrame_sm10; ///<Number of bytes from local memory according to compute capability 1.0 (0:No info).
  unsigned StackFrame_sm13; ///<Number of bytes from local memory according to compute capability 1.3 (0:No info).
  unsigned StackFrame_sm20; ///<Number of bytes from local memory according to compute capability 2.0 (0:No info).
  unsigned StackFrame_sm30; ///<Number of bytes from local memory according to compute capability 3.0 (0:No info).
  unsigned StackFrame_sm35; ///<Number of bytes from local memory according to compute capability 3.5 (0:No info).

  void UpdateCode();

public:

  JPtxasInfoKer();
  JPtxasInfoKer(const JPtxasInfoKer &ker);
  ~JPtxasInfoKer(){ Reset(); }
  JPtxasInfoKer & operator=(const JPtxasInfoKer &ker);

  void Reset();
  void SetName(const std::string &name){ Name=name; UpdateCode(); }
  void SetNameSpace(const std::string &namesp){ NameSpace=namesp; UpdateCode(); }
  void SetArgs(const std::string &args){ Args=args; UpdateCode(); }
  void AddTemplateArg(const std::string &type,const std::string &value);
  void SetRegs(unsigned sm,unsigned regs);
  void SetStackFrame(unsigned sm,unsigned mem);
  
  std::string GetCode()const{ return(Code); };
  std::string GetCodeMin()const{ return(CodeMin); };
  std::string GetName()const{ return(Name); };
  std::string GetNameSpace()const{ return(NameSpace); };
  std::string GetArgs()const{ return(Args); };
  unsigned GetRegs(unsigned sm)const;
  unsigned GetStackFrame(unsigned sm)const;
  unsigned CountTemplateArgs()const{ return(unsigned(TemplateArgs.size())); }
  std::string GetTemplateArgsType(unsigned pos)const{ return(pos<CountTemplateArgs()? TemplateArgs[pos]->type: std::string("")); }
  std::string GetTemplateArgsValue(unsigned pos)const{ return(pos<CountTemplateArgs()? TemplateArgs[pos]->value: std::string("")); }

  void Print()const;
};

//##############################################################################
//# JPtxasInfo
//##############################################################################
/// \brief Returns the number of registers of each CUDA kernel. 

class JPtxasInfo : protected JObject
{
protected:
  static const unsigned SM_COUNT=5;
  unsigned SmValues[SM_COUNT];
  std::vector<JPtxasInfoKer*> Kernels;  

public:

  JPtxasInfo();
  void Reset();
  void LoadFile(const std::string &file);
  void AddKernel(const JPtxasInfoKer &ker);
  int GetIndexKernel(const std::string &code)const;

  unsigned Count()const{ return(unsigned(Kernels.size())); }
  const JPtxasInfoKer* GetKernel(unsigned pos)const;
  void SaveCsv(const std::string &file)const;

  bool CheckSm(unsigned sm)const;

  int GetKernelIdx(const std::string &kername)const;
  int GetKernelIdx(const std::string &kername,unsigned v1)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6)const;
  int GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6,unsigned v7)const;

  unsigned GetRegs(int keridx,unsigned sm)const;
  unsigned GetStackFrame(int keridx,unsigned sm)const;

  tuint2 GetData(const std::string &kername,unsigned sm)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6)const;
  tuint2 GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6,unsigned v7)const;

  void Sort();

  void Print()const;
};

#endif







