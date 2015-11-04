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

/// \file JSpaceCtes.cpp \brief Implements the class \ref JSpaceCtes.

#include "JSpaceCtes.h"
#include "JXml.h"

//==============================================================================
/// Constructor.
//==============================================================================
JSpaceCtes::JSpaceCtes(){
  ClassName="JSpaceCtes";
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSpaceCtes::Reset(){
  SetLatticeBound(true);
  SetLatticeFluid(true);
  Gravity=TDouble3(0);
  CFLnumber=0; 
  HSwlAuto=true; HSwl=0;
  SpeedSystemAuto=true; SpeedSystem=0;
  CoefSound=0; 
  SpeedSoundAuto=true; SpeedSound=0;
  CoefH=CoefHdp=0; Gamma=0; Rhop0=0;
  Eps=0; EpsDefined=false;
  HAuto=BAuto=MassBoundAuto=MassFluidAuto=true;
  H=B=MassBound=MassFluid=0;
  Dp=0;
}

//==============================================================================
/// Loads values by default.
//==============================================================================
void JSpaceCtes::LoadDefault(){
  Reset();
  SetLatticeBound(true);
  SetLatticeFluid(true);
  SetGravity(TDouble3(0,0,-9.81));
  SetCFLnumber(0.2);
  SetHSwlAuto(true);  SetHSwl(0);
  SetSpeedSystemAuto(true);  SetSpeedSystem(0);
  SetCoefSound(10);
  SetSpeedSoundAuto(true);  SetSpeedSound(0);
  SetCoefH(0.866025);
  SetGamma(7);
  SetRhop0(1000);
  SetEps(0);
  SetHAuto(true);  SetH(0);
  SetBAuto(true);  SetB(0);
  SetMassBoundAuto(true);  SetMassBound(0);
  SetMassFluidAuto(true);  SetMassFluid(0);
}

//==============================================================================
/// Reads constants auto for definition of the case of xml node.
//==============================================================================
void JSpaceCtes::ReadXmlElementAuto(JXml *sxml,TiXmlElement* node,bool optional,std::string name,double &value,bool &valueauto){
  TiXmlElement* xele=sxml->GetFirstElement(node,name,optional);
  if(xele){
    value=sxml->GetAttributeDouble(xele,"value");
    valueauto=sxml->GetAttributeBool(xele,"auto");
  }
}

//==============================================================================
/// Reads constants for definition of the case of xml node.
//==============================================================================
void JSpaceCtes::ReadXmlDef(JXml *sxml,TiXmlElement* node){
  TiXmlElement* lattice=sxml->GetFirstElement(node,"lattice");
  SetLatticeBound(sxml->GetAttributeInt(lattice,"bound")==1);
  SetLatticeFluid(sxml->GetAttributeInt(lattice,"fluid")==1);
  SetGravity(sxml->ReadElementDouble3(node,"gravity"));
  SetCFLnumber(sxml->ReadElementDouble(node,"cflnumber","value"));
  ReadXmlElementAuto(sxml,node,false,"hswl",HSwl,HSwlAuto);
  ReadXmlElementAuto(sxml,node,true,"speedsystem",SpeedSystem,SpeedSystemAuto);
  SetCoefSound(sxml->ReadElementDouble(node,"coefsound","value"));
  ReadXmlElementAuto(sxml,node,true,"speedsound",SpeedSound,SpeedSoundAuto);
  double ch=sxml->ReadElementDouble(node,"coefh","value",true,0);
  if(!ch)ch=sxml->ReadElementDouble(node,"coefficient","value",true,0);
  double chdp=sxml->ReadElementDouble(node,"coefh","hdp",true,0);
  if(!ch && !chdp)ch=sxml->ReadElementDouble(node,"coefh","value");
  SetCoefH(ch); SetCoefHdp(chdp);
  SetGamma(sxml->ReadElementDouble(node,"gamma","value"));
  SetRhop0(sxml->ReadElementDouble(node,"rhop0","value"));
  EpsDefined=sxml->ExistsElement(node,"eps");
  SetEps(sxml->ReadElementDouble(node,"eps","value",true,0));
  ReadXmlElementAuto(sxml,node,true,"h",H,HAuto);
  ReadXmlElementAuto(sxml,node,true,"b",B,BAuto);
  ReadXmlElementAuto(sxml,node,true,"massbound",MassBound,MassBoundAuto);
  ReadXmlElementAuto(sxml,node,true,"massfluid",MassFluid,MassFluidAuto);
}

//==============================================================================
/// Writes constants auto for definition of the case of xml node.
//==============================================================================
void JSpaceCtes::WriteXmlElementAuto(JXml *sxml,TiXmlElement* node,std::string name,double value,bool valueauto,std::string comment)const{
  TiXmlElement xele(name.c_str());
  JXml::AddAttribute(&xele,"value",value); 
  JXml::AddAttribute(&xele,"auto",valueauto);
  if(!comment.empty())JXml::AddAttribute(&xele,"comment",comment);
  node->InsertEndChild(xele);
}

//==============================================================================
/// Writes constants for definition of the case of xml node.
//==============================================================================
void JSpaceCtes::WriteXmlDef(JXml *sxml,TiXmlElement* node)const{
  TiXmlElement lattice("lattice");
  JXml::AddAttribute(&lattice,"bound",GetLatticeBound());
  JXml::AddAttribute(&lattice,"fluid",GetLatticeFluid());
  node->InsertEndChild(lattice);
  JXml::AddAttribute(JXml::AddElementDouble3(node,"gravity",GetGravity()),"comment","Gravitational acceleration");
  JXml::AddAttribute(JXml::AddElementAttrib(node,"cflnumber","value",GetCFLnumber()),"comment","Coefficient to multiply Dt");
  WriteXmlElementAuto(sxml,node,"hswl",GetHSwl(),GetHSwlAuto(),"Maximum still water level to calculate speedofsound using coefsound");
  WriteXmlElementAuto(sxml,node,"speedsystem",GetSpeedSystem(),GetSpeedSystemAuto(),"Maximum system speed (by default the dam-break propagation is used)");
  JXml::AddAttribute(JXml::AddElementAttrib(node,"coefsound","value",GetCoefSound()),"comment","Coefficient to multiply speedsystem");
  WriteXmlElementAuto(sxml,node,"speedsound",GetSpeedSound(),GetSpeedSoundAuto(),"Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)");
  if(!GetCoefH()&&GetCoefHdp())JXml::AddAttribute(JXml::AddElementAttrib(node,"coefh","hdp",GetCoefHdp()),"comment","Coefficient to calculate the smoothing length (H=coefficient*sqrt(3*dp^2) in 3D)");
  else JXml::AddAttribute(JXml::AddElementAttrib(node,"coefh","value",GetCoefH()),"comment","Coefficient to calculate the smoothing length (H=coefficient*sqrt(3*dp^2) in 3D)");
  JXml::AddAttribute(JXml::AddElementAttrib(node,"gamma","value",GetGamma()),"comment","Politropic constant for water used in the state equation");
  JXml::AddAttribute(JXml::AddElementAttrib(node,"rhop0","value",GetRhop0()),"comment","Reference density of the fluid");
  WriteXmlElementAuto(sxml,node,"h",GetH(),GetHAuto());
  WriteXmlElementAuto(sxml,node,"b",GetB(),GetBAuto());
  WriteXmlElementAuto(sxml,node,"massbound",GetMassBound(),GetMassBoundAuto());
  WriteXmlElementAuto(sxml,node,"massfluid",GetMassFluid(),GetMassFluidAuto());
}

//==============================================================================
/// Reads constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::ReadXmlRun(JXml *sxml,TiXmlElement* node){
  SetGravity(sxml->ReadElementDouble3(node,"gravity"));
  SetCFLnumber(sxml->ReadElementDouble(node,"cflnumber","value"));
  SetGamma(sxml->ReadElementDouble(node,"gamma","value"));
  SetRhop0(sxml->ReadElementDouble(node,"rhop0","value"));
  SetEps(sxml->ReadElementDouble(node,"eps","value",true,0));
  SetDp(sxml->ReadElementDouble(node,"dp","value"));
  SetH(sxml->ReadElementDouble(node,"h","value"));
  SetB(sxml->ReadElementDouble(node,"b","value"));
  SetMassBound(sxml->ReadElementDouble(node,"massbound","value"));
  SetMassFluid(sxml->ReadElementDouble(node,"massfluid","value"));
}

//==============================================================================
/// Writes constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::WriteXmlRun(JXml *sxml,TiXmlElement* node)const{
  JXml::AddElementDouble3(node,"gravity",GetGravity());
  JXml::AddElementAttrib(node,"cflnumber","value",GetCFLnumber());
  JXml::AddElementAttrib(node,"gamma","value",GetGamma());
  JXml::AddElementAttrib(node,"rhop0","value",GetRhop0());
  if(EpsDefined)JXml::AddElementAttrib(node,"eps","value",GetEps());
  JXml::AddElementAttrib(node,"dp","value",GetDp());
  JXml::AddElementAttrib(node,"h","value",GetH(),"%.10E");
  JXml::AddElementAttrib(node,"b","value",GetB(),"%.10E");
  JXml::AddElementAttrib(node,"massbound","value",GetMassBound(),"%.10E");
  JXml::AddElementAttrib(node,"massfluid","value",GetMassFluid(),"%.10E");
}

//==============================================================================
/// Loads constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::LoadXmlDef(JXml *sxml,const std::string &place){
  Reset();
  TiXmlNode* node=sxml->GetNode(place,false);
  if(!node)RunException("LoadXmlDef",std::string("The item is not found \'")+place+"\'.");
  ReadXmlDef(sxml,node->ToElement());
}

//==============================================================================
/// Stores constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::SaveXmlDef(JXml *sxml,const std::string &place)const{
  WriteXmlDef(sxml,sxml->GetNode(place,true)->ToElement());
}

//==============================================================================
/// Loads constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::LoadXmlRun(JXml *sxml,const std::string &place){
  Reset();
  TiXmlNode* node=sxml->GetNode(place,false);
  if(!node)RunException("LoadXmlRun",std::string("The item is not found \'")+place+"\'.");
  ReadXmlRun(sxml,node->ToElement());
}

//==============================================================================
/// Stores constants for execution of the case of xml node.
//==============================================================================
void JSpaceCtes::SaveXmlRun(JXml *sxml,const std::string &place)const{
  WriteXmlRun(sxml,sxml->GetNode(place,true)->ToElement());
}




