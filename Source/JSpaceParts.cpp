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

/// \file JSpaceParts.cpp \brief Implements the class \ref JSpaceParts.

#include "JSpaceParts.h"
#include "JSpaceProperties.h"
#include "Functions.h"
#include "JXml.h"


//##############################################################################
//# JSpacePartBlock
//##############################################################################
//==============================================================================
/// Updates attribute Props.
//==============================================================================
void JSpacePartBlock::UpdateProperty(){
  Props=Properties->GetPropertyMk(Mk);
}

//==============================================================================
/// Returns name for XML file.
//==============================================================================
std::string JSpacePartBlock::GetNameXml()const{
  std::string ret;
  for(unsigned c=16;c<ClassName.length();c++)ret=ret+char(tolower(ClassName[c]));
  return(ret);
}

//==============================================================================
/// Reads particles information in xml format.
//==============================================================================
void JSpacePartBlock::ReadXml(JXml *sxml,TiXmlElement* ele){
  MkType=(word)sxml->GetAttributeUnsigned(ele,(Bound? "mkbound": "mkfluid")); 
  Begin=sxml->GetAttributeUnsigned(ele,"begin");
  Count=sxml->GetAttributeUnsigned(ele,"count");
  Props=sxml->GetAttributeStr(ele,"property",true);
}

//==============================================================================
/// Writes particles information in xml format.
//==============================================================================
TiXmlElement* JSpacePartBlock::WriteXml(JXml *sxml,TiXmlElement* ele)const{
  TiXmlElement item(GetNameXml().c_str());
  JXml::AddAttribute(&item,(Bound? "mkbound": "mkfluid"),MkType);
  JXml::AddAttribute(&item,"mk",Mk);
  JXml::AddAttribute(&item,"begin",Begin);
  JXml::AddAttribute(&item,"count",Count);
  if(!GetProperty().empty())JXml::AddAttribute(&item,"property",GetProperty());
  return(ele->InsertEndChild(item)->ToElement());
}

//==============================================================================
/// Returns number of values.
//==============================================================================
unsigned JSpacePartBlock::GetValuesCount()const{
  return(Properties->GetValuesCount(Props));
}

//==============================================================================
/// Returns name of value indicated.
//==============================================================================
std::string JSpacePartBlock::GetValueName(unsigned idx)const{
  return(Properties->GetValueName(Props,idx));
}

//==============================================================================
/// Returns value as string of value indicated.
//==============================================================================
std::string JSpacePartBlock::GetValueStr(unsigned idx)const{
  return(Properties->GetValueStr(Props,idx));
}

//==============================================================================
/// returns if exists the value indicated.
//==============================================================================
bool JSpacePartBlock::ExistsValue(std::string name)const{
  return(Properties->ExistsValue(Props,name));
}

//==============================================================================
/// Returns value as string of value indicated.
//==============================================================================
std::string JSpacePartBlock::GetValueStr(std::string name)const{
  return(Properties->GetValueStr(Props,name));
}

//==============================================================================
/// Returns number of subvalues.
//==============================================================================
unsigned JSpacePartBlock::GetSubValuesCount(unsigned idx)const{
  return(Properties->GetSubValuesCount(Props,idx));
}

//==============================================================================
/// Returns name of subvalue indicated.
//==============================================================================
std::string JSpacePartBlock::GetSubValueName(unsigned idx,unsigned subidx)const{
  return(Properties->GetSubValueName(Props,idx,subidx));
}

//==============================================================================
/// Returns value as string of subvalue indicated.
//==============================================================================
std::string JSpacePartBlock::GetSubValueStr(unsigned idx,unsigned subidx)const{
  return(Properties->GetSubValueStr(Props,idx,subidx));
}

//==============================================================================
/// returns if exists the subvalue indicated.
//==============================================================================
bool JSpacePartBlock::ExistsSubValue(std::string name,std::string subname)const{
  return(Properties->ExistsSubValue(Props,name,subname));
}

//==============================================================================
/// Returns value as string of subvalue indicated.
//==============================================================================
std::string JSpacePartBlock::GetSubValueStr(std::string name,std::string subname,bool optional,std::string valdef)const{
  if(optional && !Properties->ExistsSubValue(Props,name,subname))return(valdef);
  else return(Properties->GetSubValueStr(Props,name,subname));
}

//==============================================================================
/// Returns value as int of subvalue indicated.
//==============================================================================
int JSpacePartBlock::GetSubValueInt(std::string name,std::string subname,bool optional,int valdef)const{   
  if(optional && !Properties->ExistsSubValue(Props,name,subname))return(valdef);
  else return(atoi(GetSubValueStr(name,subname).c_str()));
}

//==============================================================================
/// Returns value as double of subvalue indicated.
//==============================================================================
double JSpacePartBlock::GetSubValueDouble(std::string name,std::string subname,bool optional,double valdef)const{   
  if(optional && !Properties->ExistsSubValue(Props,name,subname))return(valdef);
  else return(atof(GetSubValueStr(name,subname).c_str()));
}


//##############################################################################
//# JSpacePartBlock_Moving
//##############################################################################
//==============================================================================
/// Reads particles information in xml format.
//==============================================================================
void JSpacePartBlock_Moving::ReadXml(JXml *sxml,TiXmlElement* ele){
  JSpacePartBlock::ReadXml(sxml,ele);
  RefMotion=sxml->GetAttributeUnsigned(ele,"refmotion"); 
}

//==============================================================================
/// Writes particles information in xml format.
//==============================================================================
TiXmlElement* JSpacePartBlock_Moving::WriteXml(JXml *sxml,TiXmlElement* ele)const{
  ele=JSpacePartBlock::WriteXml(sxml,ele);
  JXml::AddAttribute(ele,"refmotion",RefMotion);
  return(ele);
}


//##############################################################################
//# JSpacePartBlock_Floating
//##############################################################################
//==============================================================================
/// Reads particles information in xml format.
//==============================================================================
void JSpacePartBlock_Floating::ReadXml(JXml *sxml,TiXmlElement* ele){
  JSpacePartBlock::ReadXml(sxml,ele);
  Massbody=sxml->ReadElementDouble(ele,"massbody","value");
  Center=sxml->ReadElementDouble3(ele,"center");
  Inertia=sxml->ReadElementDouble3(ele,"inertia");
  Velini=(sxml->GetFirstElement(ele,"velini",true)!=NULL? sxml->ReadElementDouble3(ele,"velini"): TDouble3(0));
  Omegaini=(sxml->GetFirstElement(ele,"omegaini",true)!=NULL? sxml->ReadElementDouble3(ele,"omegaini"): TDouble3(0));
}

//==============================================================================
/// Writes particles information in xml format.
//==============================================================================
TiXmlElement* JSpacePartBlock_Floating::WriteXml(JXml *sxml,TiXmlElement* ele)const{
  ele=JSpacePartBlock::WriteXml(sxml,ele);
  sxml->AddAttribute(sxml->AddElementAttrib(ele,"massbody","value",Massbody),"units_comment","kg");
  sxml->AddAttribute(sxml->AddElementAttrib(ele,"masspart","value",Massbody/GetCount()),"units_comment","kg");
  sxml->AddAttribute(sxml->AddElementDouble3(ele,"center",Center),"units_comment","metres (m)");
  sxml->AddAttribute(sxml->AddElementDouble3(ele,"inertia",Inertia),"units_comment","kg*m^2");
  if(Velini!=TDouble3(0))sxml->AddAttribute(sxml->AddElementDouble3(ele,"velini",Velini),"units_comment","m/s");
  if(Omegaini!=TDouble3(0))sxml->AddAttribute(sxml->AddElementDouble3(ele,"omegaini",Omegaini),"units_comment","radians/s");
  return(ele);
}


//##############################################################################
//# JSpaceParts
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JSpaceParts::JSpaceParts(){
  ClassName="JSpaceParts";
  Properties=new JSpaceProperties;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSpaceParts::~JSpaceParts(){
  Reset();
  delete Properties; Properties=NULL;
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSpaceParts::Reset(){
  for(unsigned c=0;c<Blocks.size();c++)delete Blocks[c];
  Blocks.clear();
  Begin=0;
  LastType=PT_Fixed;
  SetMkFirst(0,0);
  Properties->Reset();
}

//==============================================================================
/// Returns the number of particles of a given type.
//==============================================================================
unsigned JSpaceParts::Count(TpParticles type)const{
  unsigned n=0;
  for(unsigned c=0;c<Blocks.size();c++)if(Blocks[c]->Type==type)n+=Blocks[c]->GetCount();
  return(n);
}

//==============================================================================
/// Returns the total number of particles.
//==============================================================================
unsigned JSpaceParts::Count()const{
  unsigned n=0;
  for(unsigned c=0;c<Blocks.size();c++)n+=Blocks[c]->GetCount();
  return(n);
}

//==============================================================================
/// Returns the number of particles of the requested type.
//==============================================================================
unsigned JSpaceParts::CountBlocks(TpParticles type)const{
  unsigned n=0;
  for(unsigned c=0;c<Blocks.size();c++)if(Blocks[c]->Type==type)n++;
  return(n);
}

//==============================================================================
/// Returns the requested block.
//==============================================================================
const JSpacePartBlock& JSpaceParts::GetBlock(unsigned pos)const{
  if(pos>=CountBlocks())RunException("GetBlock","The requested particles block is missing.");
  return(*(Blocks[pos]));
}

//==============================================================================
/// Returns the block with a given MK.
//==============================================================================
JSpacePartBlock* JSpaceParts::GetByMkType(bool bound,word mktype)const{
  JSpacePartBlock* bk=NULL;
  for(unsigned c=0;c<Blocks.size()&&!bk;c++)if((Blocks[c]->Type!=PT_Fluid)==bound && Blocks[c]->GetMkType()==mktype)bk=Blocks[c];
  return(bk);
}

//==============================================================================
/// Checks and adds a new block of particles.
//==============================================================================
void JSpaceParts::Add(JSpacePartBlock* block){
  if(GetByMkType(block->Bound,block->GetMkType()))RunException("Add","Cannot add a block with a existing mk.");
  if(block->Type<LastType)RunException("Add","The block type is invalid after the last type added.");
  block->ConfigMk(block->Type==PT_Fluid? MkFluidFirst: MkBoundFirst);
  Blocks.push_back(block);
  Begin+=block->GetCount();
  LastType=block->Type;
}

//==============================================================================
/// Loads data in XML format from a file.
//==============================================================================
void JSpaceParts::LoadFileXml(const std::string &file,const std::string &path){
  JXml jxml;
  jxml.LoadFile(file);
  LoadXml(&jxml,path);
}

//==============================================================================
/// Stores data in XML format into a file.
//==============================================================================
void JSpaceParts::SaveFileXml(const std::string &file,const std::string &path,bool newfile)const{
  JXml jxml;
  if(!newfile)jxml.LoadFile(file);
  SaveXml(&jxml,path);
  jxml.SaveFile(file);
}

//==============================================================================
/// Loads particles information from the object XML.
//==============================================================================
void JSpaceParts::LoadXml(JXml *sxml,const std::string &place){
  Reset();
  TiXmlNode* node=sxml->GetNode(place,false);
  if(!node)RunException("LoadXml",std::string("Cannot find the element \'")+place+"\'.");
  ReadXml(sxml,node->ToElement());
}

//==============================================================================
/// Stores particles information in the object XML.
//==============================================================================
void JSpaceParts::SaveXml(JXml *sxml,const std::string &place)const{
  WriteXml(sxml,sxml->GetNode(place,true)->ToElement());
}

//==============================================================================
/// Reads particles information in XML format.
//==============================================================================
void JSpaceParts::ReadXml(JXml *sxml,TiXmlElement* lis){
  const char met[]="ReadXml";
  unsigned np=sxml->GetAttributeUnsigned(lis,"np");
  unsigned nb=sxml->GetAttributeUnsigned(lis,"nb");
  unsigned nbf=sxml->GetAttributeUnsigned(lis,"nbf");
  word mkboundfirst=(word)sxml->GetAttributeUnsigned(lis,"mkboundfirst");
  word mkfluidfirst=(word)sxml->GetAttributeUnsigned(lis,"mkfluidfirst");
  SetMkFirst(mkboundfirst,mkfluidfirst);
  //-Loads properties information.
  Properties->Reset();
  TiXmlElement* eprops=lis->FirstChildElement("properties"); 
  if(eprops)Properties->ReadXml(sxml,eprops);
  //-Loads particles information.
  TiXmlElement* ele=lis->FirstChildElement(); 
  while(ele){
    std::string cmd=ele->Value();
    if(cmd.length()&&cmd[0]!='_'){
      if(cmd=="fixed")          Add(new JSpacePartBlock_Fixed(Properties,sxml,ele));
      else if(cmd=="moving")    Add(new JSpacePartBlock_Moving(Properties,sxml,ele));
      else if(cmd=="floating")  Add(new JSpacePartBlock_Floating(Properties,sxml,ele));
      else if(cmd=="fluid")     Add(new JSpacePartBlock_Fluid(Properties,sxml,ele));
      else if(cmd!="properties")sxml->ErrReadElement(ele,cmd,false);
    }
    ele=ele->NextSiblingElement();
  }
  Begin=Count();
  if(np!=Count()||nb!=np-Count(PT_Fluid)||nbf!=Count(PT_Fixed))RunException(met,"The amount of particles does not match the header.");
  //-Checks property info in blocks.
  for(unsigned c=0;c<Blocks.size();c++){
    if(Blocks[c]->GetProperty()!=Properties->GetPropertyMk(Blocks[c]->GetMk())){
      RunException(met,std::string("Property information of mk=")+fun::IntStr(Blocks[c]->GetMk())+" does not correspond to Links configuration.");
    }
  }
}

//==============================================================================
/// Writes particles information in XML format.
//==============================================================================
void JSpaceParts::WriteXml(JXml *sxml,TiXmlElement* lis)const{
  unsigned np=Count();
  lis->Clear();
  JXml::AddAttribute(lis,"np",np);
  JXml::AddAttribute(lis,"nb",np-Count(PT_Fluid));
  JXml::AddAttribute(lis,"nbf",Count(PT_Fixed));
  JXml::AddAttribute(lis,"mkboundfirst",GetMkBoundFirst());
  JXml::AddAttribute(lis,"mkfluidfirst",GetMkFluidFirst());
  for(unsigned c=0;c<Blocks.size();c++)Blocks[c]->WriteXml(sxml,lis);
  if(Properties->GetPropertyCount()){
    TiXmlElement* eprops=JXml::AddElement(lis,"properties");
    Properties->WriteXml(sxml,eprops);
  }
}

//==============================================================================
/// Adjusts the value of Mk according to boundfirst and fluidfirst.
//==============================================================================
void JSpaceParts::SetMkFirst(word boundfirst,word fluidfirst){
  MkBoundFirst=boundfirst; MkFluidFirst=fluidfirst;
  for(unsigned c=0;c<Blocks.size();c++)Blocks[c]->ConfigMk(Blocks[c]->Type==PT_Fluid? MkFluidFirst: MkBoundFirst);
}

//==============================================================================
/// Changes particle number of block.
//==============================================================================
void JSpaceParts::SetBlockSize(unsigned pos,unsigned np){
  if(pos>=CountBlocks())RunException("SetBlockSize","The requested particles block is missing.");
  JSpacePartBlock* bk=Blocks[pos];
  if(bk->GetCount()!=np){
    const unsigned nsum=np-bk->GetCount();
    bk->SetCount(np);
    //-Cambia el begin del resto de bloques.
    for(unsigned c=pos+1;c<CountBlocks();c++)Blocks[c]->SetBegin(Blocks[c]->GetBegin()+nsum);
  }
}

//==============================================================================
/// Load information of properties and filter data.
//==============================================================================
void JSpaceParts::LoadProperties(const JSpaceProperties *props){
  //-Load data of properties.
  Properties->CopyFrom(props);
  if(Properties->GetPropertyCount()){
    //-Get list of used mk.
    std::string mks;
    for(unsigned c=0;c<CountBlocks();c++){
      const JSpacePartBlock &block=GetBlock(c);
      mks=mks+fun::IntStr(block.GetMk())+",";
    }
    //-Filter data of mks.
    Properties->FilterMk(MkBoundFirst,MkFluidFirst,mks);
  }
  //-Update property info in blocks.
  for(unsigned c=0;c<Blocks.size();c++)Blocks[c]->UpdateProperty();
}


//##############################################################################
//# JSpacePartsGetMk
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JSpacePartsGetMk::JSpacePartsGetMk(const JSpaceParts *sparts,bool splitting):Splitting(splitting){
  ClassName="JSpacePartsGetMk";
  MkRange=NULL; MkValue=NULL;
  Reset();
  Config(sparts);
}

//==============================================================================
/// Destructor.
//==============================================================================
JSpacePartsGetMk::~JSpacePartsGetMk(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSpacePartsGetMk::Reset(){
  delete[] MkRange; MkRange=NULL;
  delete[] MkValue; MkValue=NULL;
  MkCount=0;
  MkSplitting=0;
}

//==============================================================================
/// Config object.
//==============================================================================
void JSpacePartsGetMk::Config(const JSpaceParts *sparts){
  MkCount=sparts->CountBlocks();
  MkRange=new unsigned[MkCount];
  MkValue=new word[MkCount];
  for(unsigned c=0;c<sparts->CountBlocks();c++){
    const JSpacePartBlock &block=sparts->GetBlock(c);
    if(Splitting && block.Type==PT_Fluid)MkSplitting=block.GetMk();
    MkValue[c]=block.GetMk();
    MkRange[c]=block.GetBegin()+block.GetCount();
  }
}



