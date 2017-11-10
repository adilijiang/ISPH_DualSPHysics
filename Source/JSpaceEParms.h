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


/// \file JSpaceEParms.h \brief Declares the class \ref JSpaceEParms.

#ifndef _JSpaceEParms_
#define _JSpaceEParms_

#include <string>
#include <vector>
#include "JObject.h"
#include "TypesDef.h"

class JXml;
class TiXmlElement;

//##############################################################################
//# JSpaceEParms
//##############################################################################
/// \brief Manages the info of execution parameters from the input XML file.

class JSpaceEParms : protected JObject
{
public:
  /// Structure used to store information about each parameter.
  typedef struct{
    std::string key;
    std::string value;
    std::string comment;
    std::string unitscomment;
  }JSpaceEParmsItem;
private:
  typedef std::vector<JSpaceEParmsItem> VecList;
  typedef std::vector<JSpaceEParmsItem>::iterator VecListIt;

  VecList List;

  JSpaceEParmsItem* GetItemPointer(const std::string &key);
  std::string GetValueNum(const std::string &key,int num);
  std::string GetValue(const std::string &key);
  void ReadXml(JXml *sxml,TiXmlElement* lis);
  void WriteXml(JXml *sxml,TiXmlElement* lis)const;
public:
  JSpaceEParms();
  ~JSpaceEParms();
  void Reset();
  void Add(const std::string &key,const std::string &value,const std::string &comment,const std::string &unitscomment="");
  void SetValue(const std::string &key,const std::string &value);
  void SetComment(const std::string &key,const std::string &comment);
  bool Exists(const std::string &key){ return(GetItemPointer(key)!=NULL); }

  int GetValueNumInt(const std::string &key,int num,bool optional=false,int valdef=0);
  double GetValueNumDouble(const std::string &key,int num,bool optional=false,double valdef=0);
  float GetValueNumFloat(const std::string &key,int num,bool optional=false,float valdef=0){ return(float(GetValueNumDouble(key,num,optional,valdef))); }
  std::string GetValueNumStr(const std::string &key,int num,bool optional=false,std::string valdef="");
  
  int GetValueInt(const std::string &key,bool optional=false,int valdef=0){ return(GetValueNumInt(key,0,optional,valdef)); }
  double GetValueDouble(const std::string &key,bool optional=false,double valdef=0){ return(GetValueNumDouble(key,0,optional,valdef)); }
  float GetValueFloat(const std::string &key,bool optional=false,float valdef=0){ return(GetValueNumFloat(key,0,optional,valdef)); }
  std::string GetValueStr(const std::string &key,bool optional=false,std::string valdef=""){ return(GetValueNumStr(key,0,optional,valdef)); }
  tdouble3 GetValueDouble3(const std::string &key,bool optional=false,tdouble3 valdef=TDouble3(0)){ return(TDouble3(GetValueNumDouble(key,0,optional,valdef.x),GetValueNumDouble(key,1,optional,valdef.y),GetValueNumDouble(key,2,optional,valdef.z))); }

  unsigned Count()const{ return(unsigned(List.size())); }
  std::string ToString(unsigned pos)const;
  JSpaceEParmsItem GetParm(unsigned pos)const;
  void LoadFileXml(const std::string &file,const std::string &path);
  void SaveFileXml(const std::string &file,const std::string &path,bool newfile=true)const;
  void LoadXml(JXml *sxml,const std::string &place);
  void SaveXml(JXml *sxml,const std::string &place)const;
};

#endif




