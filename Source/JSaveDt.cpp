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

/// \file JSaveDt.cpp \brief Implements the class \ref JSaveDt.

#include "JSaveDt.h"
#include "JLog2.h"
#include "JXml.h"
#include "Functions.h"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

//##############################################################################
//# JSaveDt
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JSaveDt::JSaveDt(JLog2* log):Log(log){
  ClassName="JSaveDt";
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSaveDt::~JSaveDt(){
  SaveFileValuesEnd();
  if(AllDt)SaveFileAllDts();
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSaveDt::Reset(){
  TimeStart=TimeFinish=TimeInterval=0;
  FullInfo=AllDt=false;
  Count=0;
  LastInterval=0;
  memset(&ValueNull,0,sizeof(StValue));
  LastDtf=LastDt1=LastDt2=ValueNull;
  LastAceMax=LastViscDtMax=LastVelMax=ValueNull;
  CountAllDts=0;
}

//==============================================================================
/// Configures object.
//==============================================================================
void JSaveDt::Config(JXml *sxml,const std::string &place,double timemax,double timeout){
  Reset();
  LoadXml(sxml,place);
  if(TimeFinish<0)TimeFinish=timemax;
  if(TimeInterval<0)TimeInterval=timeout;
  SizeValuesSave=max(1u,min(GetSizeValues(),unsigned(timeout/TimeInterval)));
}


//==============================================================================
/// Loads initial conditions of XML object.
//==============================================================================
void JSaveDt::LoadXml(JXml *sxml,const std::string &place){
  TiXmlNode* node=sxml->GetNode(place,false);
  if(!node)RunException("LoadXml",std::string("Cannot find the element \'")+place+"\'.");
  ReadXml(sxml,node->ToElement());
}

//==============================================================================
/// Reads list of initial conditions in the XML node.
//==============================================================================
void JSaveDt::ReadXml(JXml *sxml,TiXmlElement* ele){
  const char met[]="ReadXml";
  TimeStart=sxml->ReadElementFloat(ele,"start","value",true);
  TimeFinish=sxml->ReadElementFloat(ele,"finish","value",true,-1);
  TimeInterval=sxml->ReadElementFloat(ele,"interval","value",true,-1);
  AllDt=sxml->ReadElementBool(ele,"alldt","value",true,false);
  FullInfo=sxml->ReadElementBool(ele,"fullinfo","value",true,false);
}

//==============================================================================
/// Shows object configuration using Log.
//==============================================================================
void JSaveDt::VisuConfig(std::string txhead,std::string txfoot){
  if(!txhead.empty())Log->Print(txhead);
  Log->Printf("  Time    : (%f - %f)",TimeStart,TimeFinish);
  Log->Printf("  Interval: %f (group:%u)",TimeInterval,SizeValuesSave);
  if(!txfoot.empty())Log->Print(txfoot);
}

//==============================================================================
/// Graba valores de buffer en fichero.
/// Stores file buffer values.
//==============================================================================
void JSaveDt::SaveFileValues(){
  const char met[]="SaveFileValues";
  string file=Log->GetDirOut()+"DtInfo.csv";
  const bool fexists=fun::FileExists(file);
  std::fstream pf;
  if(fexists)pf.open(file.c_str(),ios::binary|ios::out|ios::in|ios::app);
  else pf.open(file.c_str(),ios::binary|ios::out);
  if(!pf)RunException(met,"File could not be opened.",file);
  if(fexists)pf.seekp(0,pf.end);
  else{
    pf << "Time;Values";
    string tex;
    tex="Dtf";       pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
    tex="Dt1";       pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
    tex="Dt2";       pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
    if(FullInfo){
      tex="AceMax";    pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
      tex="ViscDtMax"; pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
      tex="VelMax";    pf << fun::PrintStr(";%s_mean;%s_min;%s_max",tex.c_str(),tex.c_str(),tex.c_str());
    }
    pf << endl;
    if(pf.fail())RunException(met,"File writing failure.",file);
  }
  for(unsigned c=0;c<Count;c++){
    StValue v;
    v=DtFinal[c];    pf << fun::PrintStr("%20.12E;%u;%20.12E;%20.12E;%20.12E",v.tini,v.num,v.vmean,v.vmin,v.vmax);
    v=Dt1[c];        pf << fun::PrintStr(";%20.12E;%20.12E;%20.12E",v.vmean,v.vmin,v.vmax);
    v=Dt2[c];        pf << fun::PrintStr(";%20.12E;%20.12E;%20.12E",v.vmean,v.vmin,v.vmax);
    if(FullInfo){
      v=AceMax[c];     pf << fun::PrintStr(";%20.12E;%20.12E;%20.12E",v.vmean,v.vmin,v.vmax);
      v=ViscDtMax[c];  pf << fun::PrintStr(";%20.12E;%20.12E;%20.12E",v.vmean,v.vmin,v.vmax);
      v=VelMax[c];     pf << fun::PrintStr(";%20.12E;%20.12E;%20.12E",v.vmean,v.vmin,v.vmax);
    }
    pf << endl;
  }
  if(pf.fail())RunException(met,"File writing failure.",file);
  pf.close();
  Count=0;
}

//==============================================================================
/// Graba valores de buffer.
/// Stores buffer values.
//==============================================================================
void JSaveDt::SaveFileValuesEnd(){
  if(LastDtf.num)AddLastValues();
  if(Count)SaveFileValues();
}

//==============================================================================
/// Graba valores de buffer en fichero.
/// Stores file buffer values.
//==============================================================================
void JSaveDt::SaveFileAllDts(){
  const char met[]="SaveFileAllDts";
  string file=Log->GetDirOut()+"DtAllInfo.csv";
  const bool fexists=fun::FileExists(file);
  std::fstream pf;
  if(fexists)pf.open(file.c_str(),ios::binary|ios::out|ios::in|ios::app);
  else pf.open(file.c_str(),ios::binary|ios::out);
  if(!pf)RunException(met,"File could not be opened.",file);
  if(fexists)pf.seekp(0,pf.end);
  else{
    pf << "Time;Dtf" << endl;
    if(pf.fail())RunException(met,"File writing failure.",file);
  }
  for(unsigned c=0;c<CountAllDts;c++){
    pf << fun::PrintStr("%20.12E;%20.12E",AllDts[c].x,AllDts[c].y) << endl;
  }
  if(pf.fail())RunException(met,"File writing failure.",file);
  pf.close();
  CountAllDts=0;
}

//==============================================================================
/// Guarda info del dt inicado. Si coincide timestep lo sobre.
/// Saves indicated info for dt. If it matches with timestep.
//==============================================================================
void JSaveDt::AddValueData(double timestep,double dt,StValue &value){
  if(!value.num){
    value.tini=timestep;
    value.vmean=value.vmin=value.vmax=dt;
    value.num=1;
  }
  else{
    if(value.vmin>dt)value.vmin=dt;
    if(value.vmax<dt)value.vmax=dt;
    value.vmean=(value.vmean*value.num+dt)/(value.num+1);
    value.num++;
  }
}

//==============================================================================
/// Guarda info en buffer.
/// Saves buffer info.
//==============================================================================
void JSaveDt::AddLastValues(){
  if(Count>=GetSizeValues())SaveFileValues();
  DtFinal[Count]=LastDtf;
  Dt1[Count]=LastDt1;
  Dt2[Count]=LastDt2;
  LastDtf=LastDt1=LastDt2=ValueNull;
  if(FullInfo){
    AceMax[Count]=LastAceMax;
    ViscDtMax[Count]=LastViscDtMax;
    VelMax[Count]=LastVelMax;
    LastAceMax=LastViscDtMax=LastVelMax=ValueNull;
  }
  Count++;
}

//==============================================================================
/// Guarda info del dt inicado. Si coincide timestep lo sobre
/// Saves indicated info for dt. If it matches with timestep.
//==============================================================================
void JSaveDt::AddValues(double timestep,double dtfinal,double dt1,double dt2,double acemax,double viscdtmax,double velmax){
  if(TimeStart<=timestep && timestep<=TimeFinish){
    unsigned interval=unsigned((timestep-TimeStart)/TimeInterval);
    if(LastInterval!=interval && LastDtf.num){
      AddLastValues();
      if(Count>=SizeValuesSave)SaveFileValues();
    }
    LastInterval=interval;
    AddValueData(timestep,dtfinal,LastDtf);
    AddValueData(timestep,dt1,LastDt1);
    AddValueData(timestep,dt2,LastDt2);
    if(FullInfo){
      AddValueData(timestep,acemax,LastAceMax);
      AddValueData(timestep,viscdtmax,LastViscDtMax);
      AddValueData(timestep,velmax,LastVelMax);
    }
    //-Gestion de AllDt.
    //-Management of AllDt.
    if(AllDt){
      if(CountAllDts>=SizeAllDts)SaveFileAllDts();
      AllDts[CountAllDts]=TDouble2(timestep,dtfinal);
      CountAllDts++;
    }
  }
  else if(timestep>TimeFinish && Count)SaveFileValuesEnd();
}


