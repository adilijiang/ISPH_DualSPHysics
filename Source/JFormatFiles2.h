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


/// \file JFormatFiles2.h \brief Declares the class \ref JFormatFiles2.

#ifndef _JFormatFiles2_
#define _JFormatFiles2_

#include <string>
#include <cstring>
#include "TypesDef.h"

//##############################################################################
//# JFormatFiles2
//##############################################################################
/// \brief Provides functions to store particle data in formats VTK, CSV, ASCII.

class JFormatFiles2
{
public:

  typedef enum{ UChar8,Char8,UShort16,Short16,UInt32,Int32,Float32,Double64,TpDataNull }TpData;

  /// Structure with the information of an array of particle data to be stored in CSV or VTK format.
  typedef struct {
    std::string name;
    TpData type;
    unsigned comp;
    const void *pointer;
  }StScalarData;

  /// Strucutre with the information of an array to calculate and save statistic information.
  typedef struct {
    //-Data of arrays.
    std::string name;
    TpData type;
    unsigned comp;
    const void *pointer;
    //-Selection of results.
    bool selmin,selmax,selmean;
    bool seltotal,selcomponent;
    //-Results.
    ullong num;
    double min,max,mean;
    double min1,max1,mean1;
    double min2,max2,mean2;
    double min3,max3,mean3;
  }StStatistics;

  //==============================================================================
  /// Throws a simple exception.
  //==============================================================================
  static void RunException(std::string method,std::string msg);
  
  //==============================================================================
  /// Throws an exception related to a file.
  //==============================================================================  
  static void RunException(std::string method,std::string msg,std::string file);

  //==============================================================================
  /// Returns the definition of fields.
  //==============================================================================
  static StScalarData DefineField(const std::string &name,TpData type,unsigned comp,const void *pointer=NULL){
    StScalarData f; f.name=name; f.type=type; f.comp=comp; f.pointer=pointer;
    return(f);
  }

  //==============================================================================
  /// Checks the definition of fields.
  //==============================================================================
  static void CheckFields(unsigned nfields,const StScalarData* fields);

  //==============================================================================
  /// Stores data in VTK format.
  //============================================================================== 
  static void SaveVtk(std::string fname,unsigned np,const tfloat3* pos,unsigned nfields,const StScalarData* fields);

  //==============================================================================
  /// Stores data in VTK format.
  //============================================================================== 
  static void SaveVtk(std::string fname,unsigned np,const tdouble3* pos,unsigned nfields,const StScalarData* fields);
  
  //==============================================================================
  /// Stores data in CSV format.
  //============================================================================== 
  static void SaveCsv(std::string fname,unsigned np,const tfloat3* pos,unsigned nfields,const StScalarData* fields,std::string head="");
  
  //==============================================================================
  /// Stores data in CSV format.
  //============================================================================== 
  static void SaveCsv(std::string fname,unsigned np,const tdouble3* pos,unsigned nfields,const StScalarData* fields,std::string head="");
  
  //==============================================================================
  /// Stores data in ASCII format.
  //============================================================================== 
  static void SaveAscii(std::string fname,unsigned np,const tfloat3* pos,const tdouble3* posd,unsigned nfields,const StScalarData* fields,std::string head="");
  
  //==============================================================================
  /// Returns the definition of statistics fields.
  //==============================================================================
  static StStatistics DefineStatsField(const std::string &name,TpData type,unsigned comp,const void *pointer,bool selmin=true,bool selmax=true,bool selmean=true,bool seltotal=true,bool selcomponent=true);

  //==============================================================================
  /// Checks the definition of fields.
  //==============================================================================
  static void CheckStats(unsigned nfields,const StStatistics* fields);

  //==============================================================================
  /// Calculates statistic information of arrays.
  //============================================================================== 
  static void CalculateStats(unsigned np,unsigned nfields,StStatistics* fields);

  //==============================================================================
  /// Calculates and save statistic information of arrays.
  //============================================================================== 
  static void SaveStats(std::string fname,bool firstdata,unsigned part,double timestep,unsigned np,unsigned nfields,StStatistics* fields,std::string head);

  //==============================================================================
  /// Stores data of basic particles in VTK format (example code).
  //============================================================================== 
  static void SaveVtkBasic(std::string fname,unsigned np,const tfloat3* pos,const unsigned* idp,const tfloat3* vel,const float* rhop){
    StScalarData fields[3];
    unsigned nfields=0;
    if(idp){  fields[nfields]=DefineField("Idp" ,UInt32 ,1,idp);  nfields++; }
    if(vel){  fields[nfields]=DefineField("Vel" ,Float32,3,vel);  nfields++; }
    if(rhop){ fields[nfields]=DefineField("Rhop",Float32,1,rhop); nfields++; }
    SaveVtk(fname,np,pos,nfields,fields);
  }

  //==============================================================================
  /// Stores data in CSV format.
  //==============================================================================
  static void ParticlesToCsv(std::string fname,unsigned np,unsigned nfixed,unsigned nmoving,unsigned nfloat,unsigned nfluid,unsigned nfluidout,float timestep,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *ace,const tfloat3 *vor);
  
  //==============================================================================
  /// Stores data in ASCII format.
  //==============================================================================  
  static void ParticlesToAscii(std::string fname,unsigned np,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *ace,const tfloat3 *vor);
  
  //==============================================================================
  /// Stores data in CSV format (splits positive and negative part of Ace).
  //==============================================================================  
  static void ParticlesToCsv2(std::string fname,unsigned np,unsigned nfixed,unsigned nmoving,unsigned nfloat,unsigned nfluid,unsigned nfluidout,float timestep,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *acepos,const tfloat3 *aceneg,const tfloat3 *vor);

  //==============================================================================
  /// Stores data in ASCII format (splits positive and negative part of Ace).
  //==============================================================================
  static void ParticlesToAscii2(std::string fname,unsigned np,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *acepos,const tfloat3 *aceneg,const tfloat3 *vor);

  //==============================================================================
  /// Stores data in VTK format.
  //============================================================================== 
  static void ParticlesToVtk(std::string fname,unsigned np,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *ace,const tfloat3 *vor,int domain=0);

  //==============================================================================
  /// Stores data in VTK format (position is double).
  //============================================================================== 
  static void ParticlesToVtk(std::string fname,unsigned np,const tdouble3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const unsigned *id,const byte *type,const byte *mk,const tfloat3 *ace,const tfloat3 *vor,int domain=0);

  //==============================================================================
  /// Stores data in VTK format for variables of type float.
  //==============================================================================
  static void ParticlesToVtkFloat(std::string fname,unsigned np,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float *mass,const float *id,const float *type,const float *mk,const tfloat3 *ace,const tfloat3 *vor);

  //==============================================================================
  /// Stores information of points in  VTK format.
  //==============================================================================
  static void SaveVtkPointsVar(std::string fname,unsigned np,const tfloat3 *pos,const std::string &varname,const float *var);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float.
  //==============================================================================
  static void SaveCsvPointsVar(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tfloat3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tfloat3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float.
  //==============================================================================
  static void SaveCsvPointsVar(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tdouble3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type double.
  //==============================================================================
  static void SaveCsvPointsVar(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tdouble3* pos,const double* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tdouble3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type double3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,const std::string &dataname,int part,double timestep,unsigned np,const tdouble3* pos,const tdouble3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type float.
  //==============================================================================  
  static void SaveAscPointsVar(const std::string &fname,double timestep,unsigned np,const tfloat3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type float3.
  //============================================================================== 
  static void SaveAscPointsVar3(const std::string &fname,double timestep,unsigned np,const tfloat3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type float.
  //==============================================================================  
  static void SaveAscPointsVar(const std::string &fname,double timestep,unsigned np,const tdouble3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type double.
  //==============================================================================  
  static void SaveAscPointsVar(const std::string &fname,double timestep,unsigned np,const tdouble3* pos,const double* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type float3.
  //============================================================================== 
  static void SaveAscPointsVar3(const std::string &fname,double timestep,unsigned np,const tdouble3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in ASCII file for variables of type double3.
  //============================================================================== 
  static void SaveAscPointsVar3(const std::string &fname,double timestep,unsigned np,const tdouble3* pos,const tdouble3* data,bool first=false);

  //==============================================================================
  /// Stores time and position for predefined motion.
  //==============================================================================
  static void SaveMotionPredef(const std::string &fname,unsigned np,const float *time,const tfloat3 *pos);

  //==============================================================================
  /// Stores time and position for predefined motion.
  //==============================================================================
  static void SaveMotionPredef(const std::string &fname,unsigned np,const float *time,const float *pos);

  //==============================================================================
  /// Generates a VTK file with map cells.
  //==============================================================================
  static void SaveVtkCells(const std::string &fname,const tfloat3 &posmin,const tuint3 &cells,float scell);

  //==============================================================================
  /// Generates a VTK file with boxes.
  //==============================================================================
  static void SaveVtkBoxes(const std::string &fname,unsigned nbox,const tfloat3 *vbox,float sizemin=0);
};


#endif


