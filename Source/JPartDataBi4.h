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


/// \file JPartDataBi4.h \brief Declares the class \ref JPartDataBi4.

#ifndef _JPartDataBi4_
#define _JPartDataBi4_

#include "JObject.h"
#include "TypesDef.h"
#include "JBinaryData.h"
#include <string>
#include <vector>
#include <fstream>


//##############################################################################
//# JPartDataBi4
//##############################################################################
/// \brief Allows reading/writing files with data of particles in format bi4.

class JPartDataBi4 : protected JObject
{
 public:
  typedef enum{ DIV_None=0,DIV_X=1,DIV_Y=2,DIV_Z=3,DIV_Unknown=99 }TpAxisDiv; 
  typedef enum{ PERI_None=0,PERI_X=1,PERI_Y=2,PERI_Z=4,PERI_XY=3,PERI_XZ=5,PERI_YZ=6,PERI_Unknown=96 }TpPeri; 

 private:
  JBinaryData *Data;      ///<Almacena la informacion general de los datos (constante para cada PART). Stores general information of data (constant for each PART).
  JBinaryData *Part;      ///<Pertenece a Data y almacena informacion de un part (incluyendo datos de particulas). It belongs to Data, and stores information about a part (including data of particles).

  //-Variables de gestion.
  static const unsigned FormatVerDef=130825;    ///<Version de formato by default. Version of format by default.
  unsigned FormatVer;        ///<Version de formato. Version of format.

  std::string Dir;   ///<Directorio de datos. Data Directory.
  unsigned Piece;    ///<Numero de parte. Part number.
  unsigned Npiece;   ///<Numero total de partes. Number of total parts.
  unsigned Cpart;    ///<Numero de PART. PART number.

  static std::string GetNamePart(unsigned cpart);
  void AddPartData(unsigned npok,const unsigned *idp,const ullong *idpd,const tfloat3 *pos,const tdouble3 *posd,const tfloat3 *vel,const float *rhop);
  void AddPartDataVar(const std::string &name,JBinaryDataDef::TpData type,unsigned npok,const void *v);

  void SaveFileData(std::string fname);
  unsigned GetPiecesFile(std::string file)const;
  void LoadFileData(std::string file,unsigned cpart,unsigned piece,unsigned npiece);

 public:
  JPartDataBi4();
  ~JPartDataBi4();
  void Reset();
  void ResetData();
  void ResetPart();

  long long GetAllocMemory()const;
  static std::string GetFileNamePart(unsigned cpart,unsigned piece=0,unsigned npiece=1);
  static std::string GetFileNameCase(const std::string &casename,unsigned piece=0,unsigned npiece=1);
  static std::string GetFileNameInfo(unsigned piece=0,unsigned npiece=1);
  static std::string GetFileData(std::string casename,std::string dirname,unsigned cpart,byte &npiece);

  //Grabacion de datos:
  //Recording of data
  //====================
  //-Configuracion de objeto. Object Configuration
  void ConfigBasic(unsigned piece,unsigned npiece,std::string runcode,std::string appname,std::string casename,bool data2d,const std::string &dir);
  void ConfigParticles(ullong casenp,ullong casenfixed,ullong casenmoving,ullong casenfloat,ullong casenfluid,tdouble3 caseposmin,tdouble3 caseposmax,bool npdynamic=false,bool reuseids=false);
  void ConfigCtes(double dp,double h,double b,double rhop0,double gamma,double massbound,double massfluid);
  void ConfigSimMap(tdouble3 mapposmin,tdouble3 mapposmax);
  void ConfigSimPeri(TpPeri periactive,tdouble3 perixinc,tdouble3 periyinc,tdouble3 perizinc);
  void ConfigSimDiv(TpAxisDiv axisdiv);
  void ConfigSplitting(bool splitting);

  //-Configuracion de parts. Configuration of parts.
  JBinaryData* AddPartInfo(unsigned cpart,double timestep,unsigned npok,unsigned nout,unsigned step,double runtime,tdouble3 domainmin,tdouble3 domainmax,ullong nptotal=0,ullong idmax=0);
  void AddPartData(unsigned npok,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){   AddPartData(npok,idp,NULL,pos,NULL,vel,rhop);   }
  void AddPartData(unsigned npok,const unsigned *idp,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){ AddPartData(npok,idp,NULL,NULL,posd,vel,rhop);  }
  void AddPartData(unsigned npok,const ullong *idpd,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){    AddPartData(npok,NULL,idpd,pos,NULL,vel,rhop);  }
  void AddPartData(unsigned npok,const ullong *idpd,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){  AddPartData(npok,NULL,idpd,NULL,posd,vel,rhop); }
  void AddPartDataSplitting(unsigned npok,const float *mass,const float *hvar);

  void AddPartData(const std::string &name,unsigned npok,const float    *v){  AddPartDataVar(name,JBinaryDataDef::DatFloat  ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const double   *v){  AddPartDataVar(name,JBinaryDataDef::DatDouble ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const int      *v){  AddPartDataVar(name,JBinaryDataDef::DatInt    ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const unsigned *v){  AddPartDataVar(name,JBinaryDataDef::DatUint   ,npok,(const void *)v);  }

  void AddPartData(const std::string &name,unsigned npok,const tfloat3  *v){  AddPartDataVar(name,JBinaryDataDef::DatFloat3 ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const tdouble3 *v){  AddPartDataVar(name,JBinaryDataDef::DatDouble3,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const tint3    *v){  AddPartDataVar(name,JBinaryDataDef::DatInt3   ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const tuint3   *v){  AddPartDataVar(name,JBinaryDataDef::DatUint3  ,npok,(const void *)v);  }

  void AddPartData(const std::string &name,unsigned npok,const llong    *v){  AddPartDataVar(name,JBinaryDataDef::DatLlong  ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const ullong   *v){  AddPartDataVar(name,JBinaryDataDef::DatUllong ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const short    *v){  AddPartDataVar(name,JBinaryDataDef::DatShort  ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const word     *v){  AddPartDataVar(name,JBinaryDataDef::DatUshort ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const char     *v){  AddPartDataVar(name,JBinaryDataDef::DatChar   ,npok,(const void *)v);  }
  void AddPartData(const std::string &name,unsigned npok,const byte     *v){  AddPartDataVar(name,JBinaryDataDef::DatUchar  ,npok,(const void *)v);  }

  //-Grabacion de fichero. File recording.
  void SaveFileCase(std::string casename);
  void SaveFilePart();
  void SaveFileInfo();

  //Carga de datos:
  //Loading data:
  //================
  //-Carga de fichero. File loaded.
  unsigned GetPiecesFileCase(std::string dir,std::string casename)const;
  unsigned GetPiecesFilePart(std::string dir,unsigned cpart)const;
  void LoadFileCase(std::string dir,std::string casename,unsigned piece=0,unsigned npiece=1);
  void LoadFilePart(std::string dir,unsigned cpart,unsigned piece=0,unsigned npiece=1);

  //Obtencion de datos basicos:
  //Obtaining basic data:
  //============================
  JBinaryData* GetData()const;
  unsigned GetPiece()const{ return(Piece); } 
  unsigned GetNpiece()const{ return(Npiece); } 
  std::string Get_RunCode()const{ return(GetData()->GetvText("RunCode")); } 
  std::string Get_Date()const{    return(GetData()->GetvText("Date"));    } 
  std::string Get_AppName()const{ return(GetData()->GetvText("AppName")); } 
  std::string Get_CaseName()const{return(GetData()->GetvText("CaseName",true,"")); } 
  bool Get_Data2d()const{         return(GetData()->GetvBool("Data2d"));  } 
  bool Get_Splitting()const{      return(GetData()->GetvBool("Splitting",true,false));  } 

  ullong Get_CaseNp()const{       return(GetData()->GetvUllong("CaseNp"));      } 
  ullong Get_CaseNfixed()const{   return(GetData()->GetvUllong("CaseNfixed"));  } 
  ullong Get_CaseNmoving()const{  return(GetData()->GetvUllong("CaseNmoving")); } 
  ullong Get_CaseNfloat()const{   return(GetData()->GetvUllong("CaseNfloat"));  } 
  ullong Get_CaseNfluid()const{   return(GetData()->GetvUllong("CaseNfluid"));  } 
  tdouble3 Get_CasePosMin()const{ return(GetData()->GetvDouble3("CasePosMin")); }
  tdouble3 Get_CasePosMax()const{ return(GetData()->GetvDouble3("CasePosMax")); }
  bool Get_NpDynamic()const{      return(GetData()->GetvBool("NpDynamic",true,false));  } 
  bool Get_ReuseIds()const{       return(GetData()->GetvBool("ReuseIds",true,false));   } 

  double Get_Dp()const{           return(GetData()->GetvDouble("Dp"));         }
  double Get_H()const{            return(GetData()->GetvDouble("H"));          }
  double Get_B()const{            return(GetData()->GetvDouble("B"));          }
  double Get_Rhop0()const{        return(GetData()->GetvDouble("Rhop0"));      }
  double Get_Gamma()const{        return(GetData()->GetvDouble("Gamma"));      }
  double Get_MassBound()const{    return(GetData()->GetvDouble("MassBound"));  }
  double Get_MassFluid()const{    return(GetData()->GetvDouble("MassFluid"));  }

  tdouble3 Get_MapPosMin()const{  return(GetData()->GetvDouble3("MapPosMin")); }
  tdouble3 Get_MapPosMax()const{  return(GetData()->GetvDouble3("MapPosMax")); }

  TpPeri Get_PeriActive()const{   return((TpPeri)GetData()->GetvInt("PeriActive")); }
  tdouble3 Get_PeriXinc()const{   return(GetData()->GetvDouble3("PeriXinc"));       }
  tdouble3 Get_PeriYinc()const{   return(GetData()->GetvDouble3("PeriYinc"));       }
  tdouble3 Get_PeriZinc()const{   return(GetData()->GetvDouble3("PeriZinc"));       }

  TpAxisDiv Get_AxisDiv()const{   return((TpAxisDiv)GetData()->GetvInt("AxisDiv")); }

  //Obtencion de datos del PART:
  //Obtaining the data of PART:
  //=============================
  JBinaryData* GetPart()const;
  double Get_TimeStep()const{     return(GetPart()->GetvDouble("TimeStep"));   }
  unsigned Get_Npok()const{       return(GetPart()->GetvUint("Npok"));         }
  unsigned Get_Nout()const{       return(GetPart()->GetvUint("Nout"));         }
  unsigned Get_Step()const{       return(GetPart()->GetvUint("Step"));         }
  double Get_RunTime()const{      return(GetPart()->GetvDouble("RunTime"));    }
  tdouble3 Get_DomainMin()const{  return(GetPart()->GetvDouble3("DomainMin")); }
  tdouble3 Get_DomainMax()const{  return(GetPart()->GetvDouble3("DomainMax")); }
  ullong Get_NpTotal()const{      return(GetPart()->GetvUllong("NpTotal"));    }
  ullong Get_IdMax()const{        return(GetPart()->GetvUllong("IdMax"));      }

  //Obtencion de arrays del PART:
  //Obtaining the arrays of PART:
  //==============================
  unsigned ArraysCount()const;
  std::string ArrayName(unsigned num)const;
  bool ArrayTriple(unsigned num)const;
  bool ArrayExists(std::string name)const;
  JBinaryDataArray* GetArray(std::string name)const;
  JBinaryDataArray* GetArray(std::string name,JBinaryDataDef::TpData type)const;
  unsigned Get_ArrayCount(std::string name)const{ return(GetArray(name)->GetCount()); }
  bool Get_IdpSimple()const{ return(ArrayExists("Idp")); }
  bool Get_PosSimple()const{ return(ArrayExists("Pos")); }
  unsigned Get_Idp  (unsigned size,unsigned *data)const{ return(GetArray("Idp" ,JBinaryDataDef::DatUint   )->GetDataCopy(size,data)); }
  unsigned Get_Idpd (unsigned size,ullong   *data)const{ return(GetArray("Idpd",JBinaryDataDef::DatUllong )->GetDataCopy(size,data)); }
  unsigned Get_Pos  (unsigned size,tfloat3  *data)const{ return(GetArray("Pos" ,JBinaryDataDef::DatFloat3 )->GetDataCopy(size,data)); }
  unsigned Get_Posd (unsigned size,tdouble3 *data)const{ return(GetArray("Posd",JBinaryDataDef::DatDouble3)->GetDataCopy(size,data)); }
  unsigned Get_Vel  (unsigned size,tfloat3  *data)const{ return(GetArray("Vel" ,JBinaryDataDef::DatFloat3 )->GetDataCopy(size,data)); }
  unsigned Get_Rhop (unsigned size,float    *data)const{ return(GetArray("Rhop",JBinaryDataDef::DatFloat  )->GetDataCopy(size,data)); }
  unsigned Get_Mass (unsigned size,float    *data)const{ return(GetArray("Mass",JBinaryDataDef::DatFloat  )->GetDataCopy(size,data)); }
  unsigned Get_Hvar (unsigned size,float    *data)const{ return(GetArray("Hvar",JBinaryDataDef::DatFloat  )->GetDataCopy(size,data)); }
};


#endif

/*
// - La clase JPartDataBi4 solo gestiona las particulas validas. Las particulas excluidas
//   se gestionan en JPartDataBi4Out.
// - El objeto Data

// - The class JPartDataBi4 only manages the valid particles. 
// - The particles excluded are managed in JPartDataBi4Out.
//   The Data object
*/



