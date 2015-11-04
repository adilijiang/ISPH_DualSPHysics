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
//# - Soporte para MPI. (28/10/2011)
//# - Nuevo metodo PrintDbg() que por defecto hace un fflush(stdout). (11/01/2012)
//# - Traduccion de comentarios al ingles. (10/02/2012)
//# - Nuevo metodo Printf() y PrintfDbg() usando argumentos variables. (10/03/2014)
//# - Nuevo metodo GetDirOut() para obtener el directiorio de salida.. (07/05/2014)
//# - EN:
//# Changes:
//# =========
//# - Support for MPI. (28/10/2011)
//# - New method PrintDbg() which by default makes fflush(stdout). (11/01/2012)
//# - Comment on English translation. (10/02/2012)
//# - New method Printf() and PrintfDbg() using variable arguments. (03.10.2014)
//# - New method GetDirOut() to get the output directory.. (07.05.2014)
//#############################################################################

/// \file JLog2.h \brief Declares the class \ref JLog2.

#ifndef _JLog2_
#define _JLog2_


#include "JObject.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

//##############################################################################
//# JLog2
//##############################################################################
/// \brief Manages the output of information in the file Run.out and on screen.

class JLog2 : protected JObject
{
public:
  typedef enum{ Out_ScrFile=3,Out_File=2,Out_Screen=1,Out_None=0 }TpMode_Out;
protected:
  std::string FileName;
  std::ofstream *Pf;
  bool Ok;
  bool MpiRun;
  int MpiRank,MpiLaunch;
public:
  JLog2();
  ~JLog2();
  void Reset();
  void Init(std::string fname,bool mpirun=false,int mpirank=0,int mpilaunch=0);
  void Print(const std::string &tx,TpMode_Out mode=Out_ScrFile,bool flush=false);
  void PrintDbg(const std::string &tx,TpMode_Out mode=Out_ScrFile,bool flush=true){ Print(tx,mode,flush); }
  bool IsOk()const{ return(Ok); }
  int GetMpiRank()const{ return(MpiRun? MpiRank: -1); }
  std::string GetDirOut()const;
  void Printf(const char *format,...);
  void PrintfDbg(const char *format,...);
};

#endif


