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


/// \file JSpaceProperties.h \brief Declares the classes \ref JSpaceProperties.

#ifndef _JSpaceProperties_
#define _JSpaceProperties_

#include <string>
#include <vector>
#include "JObject.h"
#include "TypesDef.h"

class JXml;
class TiXmlElement;

//##############################################################################
//# JSpacePropValue
//##############################################################################
/// \brief Manages the info of each value of properties.

class JSpacePropValue : public JObject
{
private:
  std::string Name;                    ///<Value name.
  bool Simple;                         ///<Indicates that value is in line, not item. 
  std::string Value;                   ///<Value when it is simple. 
  std::vector<std::string> SubNames;   ///<Name of each subvalue when it is not simple. 
  std::vector<std::string> SubValues;  ///<Value of each subvalue when it is not simple. 

public:
  JSpacePropValue(std::string name);
  ~JSpacePropValue();
  void Clear();
  std::string GetName()const{ return(Name); };
  bool GetSimple()const{ return(Simple); };

  void SetValue(std::string v);
  void AddSubValue(std::string subname,std::string v);

  int GetIndexSubName(std::string subname)const;
  bool ExistsSubName(std::string subname)const{ return(GetIndexSubName(subname)!=-1); }

  std::string GetValue()const;

  std::string GetSubValue(std::string subname)const;
  unsigned GetSubValuesCount()const{ return(Simple? 0: unsigned(SubValues.size())); }
  std::string GetSubValueName(unsigned idx)const;
  std::string GetSubValue(unsigned idx)const;

  std::string ToStr()const;
};

//##############################################################################
//# JSpacePropProperty
//##############################################################################
/// \brief Manages the info of each property.

class JSpacePropProperty : public JObject
{
private:
  std::string Name;                    ///<Property name.
  std::vector<JSpacePropValue*> Values;///<List of values.

public:
  JSpacePropProperty(std::string name);
  ~JSpacePropProperty();
  void Clear();
  void CopyFrom(const JSpacePropProperty* pro);

  std::string GetName()const{ return(Name); }

  int GetIndexValue(std::string name)const;
  bool ExistsNameValue(std::string name)const{ return(GetIndexValue(name)!=-1); }

  void AddValue(std::string name,std::string v);
  void AddSubValue(std::string name,std::string subname,std::string v);

  unsigned GetValuesCount()const{ return(unsigned(Values.size())); }
  std::string GetValueName(unsigned idx)const;
  bool GetValueSimple(unsigned idx)const;
  std::string GetValue(unsigned idx)const;
  const JSpacePropValue* GetValuePtr(unsigned idx)const;

  unsigned GetSubValuesCount(unsigned idx)const;
  std::string GetSubValueName(unsigned idx,unsigned subidx)const;
  std::string GetSubValue(unsigned idx,unsigned subidx)const;

  std::string ToStr()const;
};

//##############################################################################
//# JSpacePropLink
//##############################################################################
/// \brief Manages the info of each link.

class JSpacePropLink
{
public:
  typedef enum{ LINK_Mk=1,LINK_MkBound=2,LINK_MkFluid=3 }TpLink;     ///<Types of link.
private:
  TpLink Type;                         ///<Type of link.
  std::string Mks;                     ///<MK or list of MK values.
  std::string Props;                   ///<Property or list of properties.

public:
  JSpacePropLink(TpLink type,std::string mks,std::string props):Type(type),Mks(mks),Props(props){}

  TpLink GetType()const{ return(Type); }
  std::string GetMks()const{ return(Mks); }
  std::string GetProps()const{ return(Props); }
};

//##############################################################################
//# JSpacePropLinks
//##############################################################################
/// \brief Manages the links defined in properties.

class JSpacePropLinks : public JObject
{
private:
  std::vector<JSpacePropLink*> Links;  ///<List of links.

  static std::string GetPropsSort(std::string props);

public:
  JSpacePropLinks();
  ~JSpacePropLinks();
  void Reset();
  void CopyFrom(const JSpacePropLinks* links);

  unsigned GetCount()const{ return(unsigned(Links.size())); }
  
  void AddLink(JSpacePropLink::TpLink type,std::string mks,std::string props);
  void ReadXml(JXml *sxml,TiXmlElement* eprops);
  void WriteXml(JXml *sxml,TiXmlElement* eprops)const;

  std::string GetProps(word mk,word mkboundfirst,word mkfluidfirst)const;
  std::string GetProps(word mk)const;
  std::string GetAllProps()const;

};

//##############################################################################
//# JSpaceProperties
//##############################################################################
/// \brief Manages the properties assigned to the particles in the XML file.

class JSpaceProperties  : protected JObject
{
private:
  ///Structure used to store information about properties in external files.
  typedef struct {
    std::string file;
    std::string path;
  }StPropertyFile;
  std::vector<StPropertyFile> PropsFile;    ///<List of external files with properties (used only to write XML file).

  std::vector<JSpacePropProperty*> Props;   ///<List of properties.
  JSpacePropLinks* Links;                   ///<Object to manage the links.

  int GetIndexProperty(std::string name)const;
  const JSpacePropProperty* GetProperty(std::string name)const;

  const JSpacePropValue* GetValuePtr(std::string props,unsigned idx)const;
  const JSpacePropValue* GetValuePtr(std::string props,std::string name)const;

  const JSpacePropValue* GetValue(std::string props,unsigned idx)const;
  const JSpacePropValue* GetValue(std::string props,std::string name)const;
  void ReadXmlProperty(JXml *sxml,TiXmlElement* eprop);
  void ReadXmlPropertyFile(JXml *sxml,TiXmlElement* epropfile);
  void WriteXmlProperty(JXml *sxml,TiXmlElement* eprops,const JSpacePropProperty* prop)const;
  void WriteXmlPropertyFile(JXml *sxml,TiXmlElement* eprops,StPropertyFile propfile)const;
  void CheckLinks();

public:
  JSpaceProperties();
  ~JSpaceProperties();
  void Reset();
  void CopyFrom(const JSpaceProperties* props);

  void LoadFileXml(const std::string &file,const std::string &path);
  void SaveFileXml(const std::string &file,const std::string &path,bool newfile=true)const;
  void LoadXml(JXml *sxml,const std::string &place);
  void SaveXml(JXml *sxml,const std::string &place)const;
  void ReadXml(JXml *sxml,TiXmlElement* eprops);
  void WriteXml(JXml *sxml,TiXmlElement* eprops)const;

  void FilterMk(word mkboundfirst,word mkfluidfirst,std::string mkselect);

  //-Returns information.
  unsigned GetLinksCount()const{ return(Links->GetCount()); }
  unsigned GetPropertyCount()const{ return(unsigned(Props.size())); }
  std::string GetPropertyMk(word mk)const;

  //-Returns values of properties.
  unsigned GetValuesCount(std::string props)const;
  std::string GetValueName(std::string props,unsigned idx)const;
  std::string GetValueStr(std::string props,unsigned idx)const;
  bool ExistsValue(std::string props,std::string name)const;
  std::string GetValueStr(std::string props,std::string name)const;
  unsigned GetSubValuesCount(std::string props,unsigned idx)const;
  std::string GetSubValueName(std::string props,unsigned idx,unsigned subidx)const;
  std::string GetSubValueStr(std::string props,unsigned idx,unsigned subidx)const;
  bool ExistsSubValue(std::string props,std::string name,std::string subname)const;
  std::string GetSubValueStr(std::string props,std::string name,std::string subname)const;

  //-Add links and properties.
  void AddLink(std::string mks,std::string props);
  void AddLinkBound(std::string mks,std::string props);
  void AddLinkFluid(std::string mks,std::string props);
  JSpacePropProperty* AddProperty(std::string name);
  void AddPropertyFile(std::string file,std::string path);
  void ClearPropertyFile();

};



#endif




