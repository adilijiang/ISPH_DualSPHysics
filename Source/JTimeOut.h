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

/// \file JTimeOut.h \brief Declares the class \ref JTimeOut.

#ifndef _JTimeOut_
#define _JTimeOut_

#include "JObject.h"
#include <string>
#include <vector>

class JXml;
class TiXmlElement;
class JLog2;

//##############################################################################
//# XML format.
//##############################################################################
//<special>
//  <timeout>
//    <tout time="0" timeout="0.01" />
//    <tout time="0.1" timeout="0.5" />
//    <tout time="0.2" timeout="0.001" />
//    <tout time="0.3" timeout="0.01" />
//  </timeout>
//</special>


//##############################################################################
//# JTimeOut
//##############################################################################
/// \brief Manage the use of variable output time to save PARTs.

class JTimeOut : protected JObject
{
protected:
  ///Structure used to store information about timeout.
  typedef struct {
    double time;
    double tout;
  }StTimeOut;

  std::vector<StTimeOut> Times;  ///<List values for timeout.
  unsigned TimeBase;

  bool SpecialConfig; ///<Configuration loaded from XML file in special section.

  void ReadXml(JXml *sxml,TiXmlElement* ele);
  void LoadXml(JXml *sxml,const std::string &place);
  unsigned GetCount()const{ return(unsigned(Times.size())); }
  bool AddTimeOut(double t,double tout);

public:
  JTimeOut();
  ~JTimeOut();
  void Reset();
  void Config(double timeoutdef);
  void Config(std::string filexml,const std::string &place,double timeoutdef);
  bool UseSpecialConfig()const{ return(SpecialConfig); }
  void VisuConfig(JLog2 *log,std::string txhead,std::string txfoot);
  double GetNextTime(double t);
};

#endif


