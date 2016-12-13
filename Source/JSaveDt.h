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


/// \file JSaveDt.h \brief Declares the class \ref JSaveDt.

#ifndef _JSaveDt_
#define _JSaveDt_

#include <string>
#include <vector>
#include "JObject.h"
#include "Types.h"

class JXml;
class TiXmlElement;
class JLog2;

//##############################################################################
//# XML format.
//##############################################################################
//<special>
//  <savedt>
//    <start value="0" comment="Initial time (def=0)" />
//    <finish value="1000" comment="End time (def=0, no limit)" />
//    <interval value="0.01" comment="Time between output data (def=TimeOut)" />
//    <fullinfo value="1" comment="Saves AceMax, ViscDtMax and VelMax (def=0)" />
//    <alldt value="1" comment="Saves file with all dt values (def=0)" />
//  </savedt>
//</special>

//##############################################################################
//# JSaveDt
//##############################################################################
/// \brief Manages the info of dt.

class JSaveDt : protected JObject
{
public:

/// Structure with dt information.
  typedef struct {
    double tini;
    unsigned num;
    double vmean;
    double vmin;
    double vmax;
  }StValue;

private:
  JLog2* Log;
  double TimeStart;    //-Instante a partir del cual se empieza a recopilar informacion del dt.
  double TimeFinish;   //-Instante a partir del cual se deja de recopilar informacion del dt.
  double TimeInterval; //-Cada cuanto se guarda info del dt.
  bool FullInfo;       //-Saves AceMax, ViscDtMax and VelMax.
  bool AllDt;
  unsigned SizeValuesSave;

  StValue ValueNull;

  unsigned Count;                       //-Numero de intervalos almacenados.
  static const unsigned SizeValues=100; //-Numero maximo de intervalos a almacenar en buffer.
  StValue DtFinal[SizeValues];          //-Dt minimo resultante [SizeValues].
  StValue Dt1[SizeValues];              //-Dt1 [SizeValues].
  StValue Dt2[SizeValues];              //-Dt2 [SizeValues].
  StValue AceMax[SizeValues];           //-AceMax [SizeValues].
  StValue ViscDtMax[SizeValues];        //-ViscDtMax [SizeValues].
  StValue VelMax[SizeValues];           //-VelMax [SizeValues].

  unsigned GetSizeValues()const{ return(SizeValues); }

  static const unsigned SizeAllDts=1000;
  tdouble2 AllDts[SizeAllDts];           
  unsigned CountAllDts;

  unsigned LastInterval;
  StValue LastDtf,LastDt1,LastDt2;
  StValue LastAceMax,LastViscDtMax,LastVelMax;

  void ReadXml(JXml *sxml,TiXmlElement* ele);
  void LoadXml(JXml *sxml,const std::string &place);
  void SaveFileValues();
  void SaveFileValuesEnd();
  void SaveFileAllDts();
  void AddValueData(double timestep,double dt,StValue &value);
  void AddLastValues();
public:
  JSaveDt(JLog2* log);
  ~JSaveDt();
  void Reset();
  void Config(JXml *sxml,const std::string &place,double timemax,double timeout);
  void VisuConfig(std::string txhead,std::string txfoot);
  void AddValues(double timestep,double dtfinal,double dt1,double dt2,double acemax,double viscdtmax,double velmax);
  bool GetFullInfo()const{ return(FullInfo); }
};


#endif




