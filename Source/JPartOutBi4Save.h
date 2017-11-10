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


/// \file JPartOutBi4Save.h \brief Declares the class \ref JPartOutBi4Save.

#ifndef _JPartOutBi4Save_
#define _JPartOutBi4Save_

#include "JObject.h"
#include "TypesDef.h"
#include "JBinaryData.h"
#include <string>
#include <vector>
#include <fstream>


//##############################################################################
//# JPartOutBi4Save
//##############################################################################
/// \brief Allows writing information of excluded particles during simulation.

class JPartOutBi4Save : protected JObject
{
 private:
  JBinaryData *Data;      ///<Almacena la informacion general de los datos (constante para cada PART). Stores general information of data (constant for each PART).
  JBinaryData *Part;      ///<Pertenece a Data y almacena informacion de un part (incluyendo datos de particulas). Belongs to data and stores information of a part (including data for particles).

  //-Variables de gestion. Management of variables.
  static const unsigned FormatVerDef=131122;    ///<Version de formato by default. Version of format by default.
  unsigned FormatVer;        ///<Version de formato. Format version.

  std::string Dir;   ///<Directorio de datos. Data directory.
  unsigned Block;    ///<Numero de bloque. Block number.
  unsigned Piece;    ///<Numero de parte. Part number.
  unsigned Npiece;   ///<Numero total de partes. Number total of parts.

  bool InitialSaved;     ///<Indica si se grabo la informacion de cabecera. Indicates if header information is recorded.
  unsigned BlockNout;    ///<Numero de particulas exclidas ya grabadas en el bloque actual. Number of excluded particles already recorded in the current block.

  static const unsigned BLOCKNOUTMIN=500000;      //-Valor por defecto para BlockNoutMin. 0.5M (18-24 MB). Default value for BlockNoutMin. 0.5 M (18-24 MB).
  static const unsigned BLOCKNOUTMAX=2500000;  //-Valor por defecto para BlockNoutMax. 2.5M (90-120 MB). Default value for BlockNoutMin.  2.5M (90-120 MB).
  unsigned BlockNoutMin; ///<Numero minimo de particulas que se meten por bloque. Minimum number of particles that are are getting in a block.
  unsigned BlockNoutMax; ///<Maximum number of particles that should geting in a block.

  unsigned Cpart;    ///<Numero de PART. PART number.


  static std::string GetNamePart(unsigned cpart);
  JBinaryData* AddPartOut(unsigned cpart,double timestep,unsigned nout,const unsigned *idp,const ullong *idpd,const tfloat3 *pos,const tdouble3 *posd,const tfloat3 *vel,const float *rhop);

 public:
  JPartOutBi4Save();
  ~JPartOutBi4Save();
  void Reset();
  void ResetData();
  void ResetPart();

  long long GetAllocMemory()const;
  static std::string GetFileNamePart(unsigned block,unsigned piece=0,unsigned npiece=1);

  //Grabacion de datos:
  //Data recording:
  //====================
  //-Configuracion de objeto. Object Configuration.
  void ConfigBasic(unsigned piece,unsigned npiece,std::string runcode,std::string appname,bool data2d,const std::string &dir);
  void ConfigParticles(ullong casenp,ullong casenfixed,ullong casenmoving,ullong casenfloat,ullong casenfluid);
  void ConfigLimits(const tdouble3 &mapposmin,const tdouble3 &mapposmax,float rhopmin,float rhopmax);
  void SaveInitial();

  //-Configuracion de parts. Configuration of parts.
  JBinaryData* AddPartOut(unsigned cpart,double timestep,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){   return(AddPartOut(cpart,timestep,nout,idp,NULL,pos,NULL,vel,rhop));   }
  JBinaryData* AddPartOut(unsigned cpart,double timestep,unsigned nout,const unsigned *idp,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){ return(AddPartOut(cpart,timestep,nout,idp,NULL,NULL,posd,vel,rhop));  }
  JBinaryData* AddPartOut(unsigned cpart,double timestep,unsigned nout,const ullong *idpd,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){    return(AddPartOut(cpart,timestep,nout,NULL,idpd,pos,NULL,vel,rhop));  }
  JBinaryData* AddPartOut(unsigned cpart,double timestep,unsigned nout,const ullong *idpd,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){  return(AddPartOut(cpart,timestep,nout,NULL,idpd,NULL,posd,vel,rhop)); }

  //-Grabacion de fichero. File recording.
  void SavePartOut();
  void SavePartOut(unsigned cpart,double timestep,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){   AddPartOut(cpart,timestep,nout,idp,NULL,pos,NULL,vel,rhop); SavePartOut();   }
  void SavePartOut(unsigned cpart,double timestep,unsigned nout,const unsigned *idp,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){ AddPartOut(cpart,timestep,nout,idp,NULL,NULL,posd,vel,rhop); SavePartOut();  }
  void SavePartOut(unsigned cpart,double timestep,unsigned nout,const ullong *idpd,const tfloat3 *pos,const tfloat3 *vel,const float *rhop){    AddPartOut(cpart,timestep,nout,NULL,idpd,pos,NULL,vel,rhop); SavePartOut();  }
  void SavePartOut(unsigned cpart,double timestep,unsigned nout,const ullong *idpd,const tdouble3 *posd,const tfloat3 *vel,const float *rhop){  AddPartOut(cpart,timestep,nout,NULL,idpd,NULL,posd,vel,rhop); SavePartOut(); }

  unsigned GetBlockNoutMin()const{ return(BlockNoutMin); }
  unsigned GetBlockNoutMax()const{ return(BlockNoutMin); }
  void SetBlockNoutMin(unsigned v){ BlockNoutMin=(v<50000u? 50000u: v); } 
  void SetBlockNoutMax(unsigned v){ BlockNoutMax=(v<250000u? 250000u: v); } 

};


#endif

