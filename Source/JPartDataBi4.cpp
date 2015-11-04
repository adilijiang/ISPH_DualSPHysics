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

/// \file JPartDataBi4.cpp \brief Implements the class \ref JPartDataBi4

#include "JPartDataBi4.h"
//#include "JBinaryData.h"
#include "Functions.h"

#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

using namespace std;

//##############################################################################
//# JPartDataBi4
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JPartDataBi4::JPartDataBi4(){
  ClassName="JPartDataBi4";
  Data=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JPartDataBi4::~JPartDataBi4(){
  delete Data; Data=NULL;
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPartDataBi4::Reset(){
  ResetData();
  Dir="";
  Piece=0;
  Npiece=1;
}

//==============================================================================
/// Elimina informacion de Data.
/// Deletes information from data.
//==============================================================================
void JPartDataBi4::ResetData(){
  delete Data; 
  Data=new JBinaryData(ClassName);
  Part=Data->CreateItem("Part");
  Cpart=0;
}

//==============================================================================
/// Elimina informacion de PARTs.
/// Deletes information from PARTs.
//==============================================================================
void JPartDataBi4::ResetPart(){
  Part->Clear();
}

//==============================================================================
/// Devuelve la memoria reservada.
/// Returns allocated memory
//==============================================================================
long long JPartDataBi4::GetAllocMemory()const{  
  long long s=0;
  s+=Data->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Devuelve nombre de fichero PART segun los parametros indicados.
/// Returns the filename PART according to the specified parameters.
//==============================================================================
std::string JPartDataBi4::GetFileNamePart(unsigned cpart,unsigned piece,unsigned npiece){
  string fname="Part";
  char cad[32];
  if(npiece>1){
    sprintf(cad,"_p%02d",piece);
    fname=fname+cad;
  }
  sprintf(cad,"_%04u.bi4",cpart);
  return(fname+cad);
}

//==============================================================================
/// Devuelve nombre de fichero de caso segun los parametros indicados.
/// Returns filename's case according to the specified parameters.
//==============================================================================
std::string JPartDataBi4::GetFileNameCase(const std::string &casename,unsigned piece,unsigned npiece){
  string fname=casename;
  if(npiece>1){
    char cad[32];
    sprintf(cad,"_p%02d",piece);
    fname=fname+cad;
  }
  return(fname+".bi4");
}

//==============================================================================
/// Devuelve nombre de fichero info de caso segun los parametros indicados.
/// Returns filename's case info according to the specified parameters.
//==============================================================================
std::string JPartDataBi4::GetFileNameInfo(unsigned piece,unsigned npiece){
  string fname="PartInfo";
  if(npiece>1){
    char cad[32];
    sprintf(cad,"_p%02d",piece);
    fname=fname+cad;
  }
  return(fname+".ibi4");
}

//==============================================================================
/// Devuelve nombre de fichero encontrado segun configuracion e indica si esta o 
/// no dividido en varias piezas (0:No se encontro, 1:Una pieza, 2:Varias piezas).
/// Returns file name found depending on configuration and indicates 
/// whether or not it is divided into several parts
/// (0:not found, 1:a piece, 2:Several parts).
//==============================================================================
std::string JPartDataBi4::GetFileData(std::string casename,std::string dirname,unsigned cpart,byte &npiece){
  byte npie=0;
  string file;
  if(casename.empty()){
    dirname=fun::GetDirWithSlash(dirname);
    if(fun::FileExists(dirname+JPartDataBi4::GetFileNamePart(cpart,0,1))){
      file=dirname+JPartDataBi4::GetFileNamePart(cpart,0,1);
      npie=1;
    }
    else if(fun::FileExists(dirname+JPartDataBi4::GetFileNamePart(cpart,0,2))){
      file=dirname+JPartDataBi4::GetFileNamePart(cpart,0,2);
      npie=2;
    }
  }
  else{
    if(fun::FileExists(JPartDataBi4::GetFileNameCase(casename,0,1))){
      file=JPartDataBi4::GetFileNameCase(casename,0,1);
      npie=1;
    }
    else if(fun::FileExists(JPartDataBi4::GetFileNameCase(casename,0,2))){
      file=JPartDataBi4::GetFileNameCase(casename,0,2);
      npie=2;
    }
  }
  npiece=npie;
  return(file);
}




//==============================================================================
/// Configuracion de variables basicas.
/// Configuration of basic variables.
//==============================================================================
void JPartDataBi4::ConfigBasic(unsigned piece,unsigned npiece,std::string runcode,std::string appname,bool data2d,const std::string &dir){
  ResetData();
  Piece=piece; Npiece=npiece;
  Dir=fun::GetDirWithSlash(dir);
  Data->SetvUint("Piece",Piece);
  Data->SetvUint("Npiece",Npiece);
  Data->SetvText("RunCode",runcode);
  Data->SetvText("Date",fun::GetDateTime());
  Data->SetvText("AppName",appname);
  Data->SetvBool("Data2d",data2d);
  ConfigSimMap(TDouble3(0),TDouble3(0));
  ConfigSimPeri(PERI_Unknown,TDouble3(0),TDouble3(0),TDouble3(0));
  ConfigSimDiv(DIV_Unknown);
}

//==============================================================================
/// Configuracion de numero de particulas y dominio del caso.
/// Setting number of particles and domain of the case.
//==============================================================================
void JPartDataBi4::ConfigParticles(ullong casenp,ullong casenfixed,ullong casenmoving,ullong casenfloat,ullong casenfluid,tdouble3 caseposmin,tdouble3 caseposmax,bool npdynamic,bool reuseids){
  if(casenp!=casenfixed+casenmoving+casenfloat+casenfluid)RunException("ConfigParticles","Error in the number of particles.");
  Data->SetvUllong("CaseNp",casenp);
  Data->SetvUllong("CaseNfixed",casenfixed);
  Data->SetvUllong("CaseNmoving",casenmoving);
  Data->SetvUllong("CaseNfloat",casenfloat);
  Data->SetvUllong("CaseNfluid",casenfluid);
  Data->SetvDouble3("CasePosMin",caseposmin);
  Data->SetvDouble3("CasePosMax",caseposmax);
  Data->SetvBool("NpDynamic",npdynamic);
  Data->SetvBool("ReuseIds",reuseids);
}

//==============================================================================
/// Configuracion de constantes.
/// Configuration of constants.
//==============================================================================
void JPartDataBi4::ConfigCtes(double dp,double h,double b,double rhop0,double gamma,double massbound,double massfluid){
  Data->SetvDouble("Dp",dp);
  Data->SetvDouble("H",h);
  Data->SetvDouble("B",b);
  Data->SetvDouble("Rhop0",rhop0);
  Data->SetvDouble("Gamma",gamma);
  Data->SetvDouble("MassBound",massbound);
  Data->SetvDouble("MassFluid",massfluid);
}

//==============================================================================
/// Configuracion de variables de simulacion: map limits.
/// Configuration of variables of simulation: map limits.
//==============================================================================
void JPartDataBi4::ConfigSimMap(tdouble3 mapposmin,tdouble3 mapposmax){
  Data->SetvDouble3("MapPosMin",mapposmin);
  Data->SetvDouble3("MapPosMax",mapposmax);
}

//==============================================================================
/// Configuracion de variables de simulacion: map limits.
/// Configuration of variables of simulation: map limits.
//==============================================================================
void JPartDataBi4::ConfigSimPeri(TpPeri periactive,tdouble3 perixinc,tdouble3 periyinc,tdouble3 perizinc){
  Data->SetvInt("PeriActive",int(periactive));
  Data->SetvDouble3("PeriXinc",perixinc);
  Data->SetvDouble3("PeriYinc",periyinc);
  Data->SetvDouble3("PeriZinc",perizinc);
}

//==============================================================================
/// Configuracion de variables de simulacion: axis division.
/// Configuration of variables of simulation: axis division.
//==============================================================================
void JPartDataBi4::ConfigSimDiv(TpAxisDiv axisdiv){
  Data->SetvInt("AxisDiv",int(axisdiv));
}

//==============================================================================
/// Configuracion uso de Splitting.
/// Configuration used for Splitting.
//==============================================================================
void JPartDataBi4::ConfigSplitting(bool splitting){
  Data->SetvBool("Splitting",splitting);
}

//==============================================================================
/// Devuelve nombre de part segun su numero.
/// Returns name of part according to their number.
//==============================================================================
std::string JPartDataBi4::GetNamePart(unsigned cpart){
  char cad[64];
  sprintf(cad,"PART_%04u",cpart);
  return(cad);
}

//==============================================================================
/// A�ade informacion de nuevo part.
// Add information to new part.
//==============================================================================
JBinaryData* JPartDataBi4::AddPartInfo(unsigned cpart,double timestep,unsigned npok,unsigned nout,unsigned step,double runtime,tdouble3 domainmin,tdouble3 domainmax,ullong nptotal,ullong idmax){
  Part->Clear();
  Cpart=cpart;
  Part->SetName(GetNamePart(cpart));
  Part->SetvUint("Cpart",cpart);
  Part->SetvDouble("TimeStep",timestep);
  Part->SetvUint("Npok",npok);
  Part->SetvUint("Nout",nout);
  Part->SetvUint("Step",step);
  Part->SetvDouble("RunTime",runtime);
  Part->SetvDouble3("DomainMin",domainmin);
  Part->SetvDouble3("DomainMax",domainmax);
  if(nptotal)Part->SetvUllong("NpTotal",nptotal);
  if(idmax)Part->SetvUllong("IdMax",idmax);
  return(Part);
}


//==============================================================================
/// A�ade datos de particulas de de nuevo part.
/// Adds data of particles to new part.
//==============================================================================
void JPartDataBi4::AddPartData(unsigned npok,const unsigned *idp,const ullong *idpd,const tfloat3 *pos,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){
  const char met[]="AddPartData";
  if(!idp&&!idpd)RunException(met,"The id of particles is invalid.");
  if(!pos&&!posd)RunException(met,"The position of particles is invalid.");
  //-Comprueba valor de npok. Checks value of npok.
  if(Part->GetvUint("Npok")!=npok)RunException(met,"Part information is invalid.");
  //-Crea array con particulas validas. Creates valid particles array.
  if(idpd)Part->CreateArray("Idpd",JBinaryDataDef::DatUllong,npok,idpd,true);
  else    Part->CreateArray("Idp" ,JBinaryDataDef::DatUint,npok,idp,true);
  if(posd)Part->CreateArray("Posd",JBinaryDataDef::DatDouble3,npok,posd,true);
  else    Part->CreateArray("Pos" ,JBinaryDataDef::DatFloat3,npok,pos,true);
  Part->CreateArray("Vel",JBinaryDataDef::DatFloat3,npok,vel,true);
  Part->CreateArray("Rhop",JBinaryDataDef::DatFloat,npok,rhop,true);
}

//==============================================================================
/// A�ade datos Splitting de particulas de de nuevo part.
/// Add data Splitting of particles to new part.
//==============================================================================
void JPartDataBi4::AddPartDataSplitting(unsigned npok,const float *splitmass,const float *splithvar){
  const char met[]="AddPartDataSplitting";
  if(!splitmass || !splithvar)RunException(met,"The pointer data is invalid.");
  //-Comprueba valor de npok. Checks value of npok.
  if(Part->GetvUint("Npok")!=npok)RunException(met,"Part information is invalid.");
  if(!Data->GetvBool("Splitting"))RunException(met,"Splitting is not configured.");
  //-Crea array con particulas validas. Creates valid particles array.
  Part->CreateArray("SplitMass",JBinaryDataDef::DatFloat,npok,splitmass,true);
  Part->CreateArray("SplitHvar",JBinaryDataDef::DatFloat,npok,splithvar,true);
}

//==============================================================================
/// Graba le fichero BI4 indicado.
/// Writes indicated BI4 file.
//==============================================================================
void JPartDataBi4::SaveFileData(std::string fname){
  const char met[]="SaveFileData";
  //-Comprueba que Part tenga algun array de datos. Check that Part has array with data.
  if(!Part->GetArraysCount())RunException(met,"There is not array of particles data.");
  //-Graba fichero. Record file.
  Data->SaveFile(Dir+fname,false,true);
  Part->RemoveArrays();
  //Data->SaveFileXml(Dir+fun::GetWithoutExtension(fname)+"__.xml");
}

//==============================================================================
/// Graba fichero BI4 con el nombre da caso indicado.
/// Writes file BI4 with the case name indicated.
//==============================================================================
void JPartDataBi4::SaveFileCase(std::string casename){
  SaveFileData(GetFileNameCase(casename,Piece,Npiece));
}

//==============================================================================
/// Graba fichero PART con datos de particulas.
/// Writes file PART with data of particles.
//==============================================================================
void JPartDataBi4::SaveFilePart(){
  SaveFileData(GetFileNamePart(Cpart,Piece,Npiece));
}

//==============================================================================
/// Graba info de PART.
/// Writes info PART.
//==============================================================================
void JPartDataBi4::SaveFileInfo(){
  Data->SetHideItems(true,false);
  Part->SetHideArrays(true,false);
  Part->SaveFileListApp(Dir+GetFileNameInfo(Piece,Npiece),ClassName+"_Info",true,false);
  Data->SetHideItems(false,false);
}

//==============================================================================
/// Devuelve el numero de piezas del fichero indicado.
/// Returns the number of parts from the indicated file.
//==============================================================================
unsigned JPartDataBi4::GetPiecesFile(std::string file)const{
  unsigned npieces=0;
  if(fun::FileExists(file)){
    JBinaryData dat(ClassName);
    dat.OpenFileStructure(file,ClassName);
    npieces=dat.GetvUint("Npiece");
  }
  return(npieces);
}

//==============================================================================
/// Devuelve el numero de piezas del caso indicado.
/// Returns the number of parts of the case indicated.
//==============================================================================
unsigned JPartDataBi4::GetPiecesFileCase(std::string dir,std::string casename)const{
  unsigned npieces=0;
  if(fun::FileExists(dir+GetFileNameCase(casename,0,1)))npieces=1;
  else npieces=GetPiecesFile(dir+GetFileNameCase(casename,0,2));
  return(npieces);
}

//==============================================================================
/// Devuelve el numero de piezas del caso indicado.
/// Returns the number of parts of the case indicated.
//==============================================================================
unsigned JPartDataBi4::GetPiecesFilePart(std::string dir,unsigned cpart)const{
  unsigned npieces=0;
  if(fun::FileExists(Dir+GetFileNamePart(cpart,0,1)))npieces=1;
  else npieces=GetPiecesFile(dir+GetFileNamePart(cpart,0,2));
  return(npieces);
}

//==============================================================================
/// Graba fichero BI4 con el nombre da caso indicado.
/// Writes file BI4 with the case name indicated.
//==============================================================================
void JPartDataBi4::LoadFileData(std::string file,unsigned cpart,unsigned piece,unsigned npiece){
  const char met[]="LoadFileData";
  ResetData();
  Cpart=cpart; Piece=piece; Npiece=npiece;
  Data->OpenFileStructure(file,ClassName);
  if(Piece!=Data->GetvUint("Piece")||Npiece!=Data->GetvUint("Npiece"))RunException(met,"PART configuration is invalid.");
  Part=Data->GetItem(GetNamePart(Cpart));
  if(!Part)RunException(met,"PART data is invalid.");
  Cpart=Part->GetvUint("Cpart");
}

//==============================================================================
/// Carga fichero BI4 con el nombre da caso indicado.
/// Load file BI4 with the case name indicated.
//==============================================================================
void JPartDataBi4::LoadFileCase(std::string dir,std::string casename,unsigned piece,unsigned npiece){
  LoadFileData(fun::GetDirWithSlash(dir)+GetFileNameCase(casename,piece,npiece),0,piece,npiece);
}

//==============================================================================
/// Carga fichero PART con datos de particulas.
/// Load file PART with data of particles.
//==============================================================================
void JPartDataBi4::LoadFilePart(std::string dir,unsigned cpart,unsigned piece,unsigned npiece){
  LoadFileData(fun::GetDirWithSlash(dir)+GetFileNamePart(cpart,piece,npiece),cpart,piece,npiece);
}

//==============================================================================
/// Devuelve el puntero a Part con los datos del PART.
/// Returns a pointer to Part with the data of the PART.
//==============================================================================
JBinaryData* JPartDataBi4::GetData()const{
  if(!Data)RunException("GetData","The data object is not available.");
  return(Data);
}

//==============================================================================
/// Devuelve el puntero a Part con los datos del PART.
/// Returns a pointer to Part with the data of the PART.
//==============================================================================
JBinaryData* JPartDataBi4::GetPart()const{
  if(!Part)RunException("GetPart","PART data is not available.");
  return(Part);
}

//==============================================================================
/// Devuelve el puntero a Part con los datos del PART.
/// Returns a pointer to Part with the data of the PART.
//==============================================================================
JBinaryDataArray* JPartDataBi4::GetArray(std::string name)const{
  JBinaryDataArray* ar=GetPart()->GetArray(name);
  if(!ar)RunException("GetArray","Array is not available.");
  return(ar);
}

//==============================================================================
/// Devuelve true si existe el array indicado.
/// Returns true if the indicated array exists.
//==============================================================================
bool JPartDataBi4::ArrayExists(std::string name)const{
  return(GetPart()->GetArray(name)!=NULL);
}




