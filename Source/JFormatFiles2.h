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

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Clase creada para sustituir a JFormatFiles. (15/12/2010)
//# - Nuevas funciones para generar ficheros con datos de puntos en formato
//#   CSV y ASCII:  SaveCsvPointsVar(), SaveCsvPointsVar3(), SaveAscPointsVar(), 
//#   SaveAscPointsVar3() (22/12/2010)
//# - Nueva variable ACE en formatos VTK, CSV y ASCII. (16/06/2011)
//# - Nueva variable Vorticity en formatos VTK, CSV y ASCII. (13/07/2011)
//# - Se separa la variable ACE en parte positiva y negativa en formatos CSV y 
//#   ASCII (ParticlesToCsv2() y ParticlesToAscii()). (28/09/2011)
//# - En los VTK de puntos, cada punto se guarda como una celda. Esto permite
//#   el uso de algunas operaciones como Threshold en Paraview. Sin embargo,
//#   aumenta el tamaño en 3 bytes por partícula (entorno al 11%). (03/12/2011)
//# - Nueva funcion SaveVtkCells() para generar ficheros VTK con una malla de 
//#   celdas segun los parametros especificados. (13/12/2011)
//# - Nueva funcion PointsToVtk() para generar ficheros VTK de puntos a partir 
//#   de datos codificados en un JBuffer. (30/12/2011)
//# - Nueva funcion CellsToVtk() para generar ficheros VTK de celdas sin datos
//#   a partir de datos codificados en un JBuffer. (09/01/2012)
//# - Nueva funcion SaveVtkDomain() para generar ficheros VTK con una serie
//#   de dominios. (09/01/2012)
//# - En PointsToVtk() si el numero de puntos es cero, se crea uno para evitar
//#   un error al generar el fichero VTK. (18/01/2012)
//# - Traduccion de comentarios al ingles. (10/02/2012)
//# - Nuevas funciones (SaveVtk,SaveCsv) para crear ficheros VTK mas 
//#   generales. (17/05/2012)
//# - Nueva funcion SaveVtkBasic como ejemplo de uso de SaveVtk. (06/06/2012)
//# - Funcion SaveVtkBox() para generar ficheros VTK con una serie de 
//#   cajas. (06/06/2012)
//# - Los metodos PointsToVtk() y CellsToVtk() se eleminaron de esta clase y
//#   pasaron a la clase JBufferToVtk. (06/06/2012)
//# - Error corregido en SaveVtkBasic() cuando se usaban arrays nulos. (06/10/2012)
//# - Se elimino un posible problema de precision en el calculo de los vertices
//#   de una caja. (15-05-2013)
//# - Permite añadir una cabecera de fichero usando SaveCsv(...,head). (20-11-2013)
//# - Nuevos metodos SaveCsvPointsVar(), SaveCsvPointsVar3(), SaveAscPointsVar()
//#   y SaveAscPointsVar3() para grabar la posicion como tdouble3. (30-12-2013)
//# - Nuevo metodo SaveAscii() mediente StScalarData* fields. (16-01-2014)
//# - Nuevos metodos XXXStats() que permiten calcular y grabar ficheros CSV
//#   con valores de minimo, maximo y media. (16-01-2014)
//# - Error corregido en funcion DefineStatsField(). (18-02-2015)
//# - Funciones SaveCsv() y SaveCsv() para POS en doble precision. (24-03-2015)
//# - EN:
//# Changes:
//# ========
//# - Class created to replace JFormatFiles. (15/12/2010)
//# - New features to generate files with point data in 
//#   CSV and ASCII format: SaveCsvPointsVar(), SaveCsvPointsVar3(), SaveAscPointsVar(),
//#   SaveAscPointsVar3() (22/12/2010)
//# - New variable ACE in VTK, CSV and ASCII formats. (16/06/2011)
//# - New variable Vorticity in VTK, CSV and ASCII formats. (13/07/2011)
//# - ACE variable is separated into positive and negative parts in CSV and
//#   ASCII formats(ParticlesToCsv2() and ParticlesToAscii()). (28/09/2011)
//# - In the VTK of the points, each point is stored as a cell. This allows
//#   using some operations such as Threshold in Paraview. Nevertheless,
//#   the size increases in 3 bytes per particle (around 11%). (03/12/2011)
//# - New function SaveVtkCells() to generate VTK files with a cell mesh 
//#   according to specified settings. (13/12/2011)
//# - New function PointsToVtk() to generate VTK files from data points
//#   encoded in a JBuffer. (30/12/2011)
//# - New function CellsToVtk() to generate VTK files with cells without data
//#   using data encoded in a JBuffer. (09/01/2012)
//# - New function SaveVtkDomain() to generate VTK files for a
//#   number of domains. (09/01/2012)
//# - In PointsToVtk() if the number of points is zero, one is created to avoid
//#   generating an error VTK file. (18/01/2012)
//# - Comment on English translation. (10/02/2012)
//# - New Features (SaveVtk, SaveCsv) to create more general VTK files (17/05/2012)
//# - New function SaveVtkBasic shows an example of SaveVtk use. (06/06/2012)
//# - Function SaveVtkBox()  to generate VTK files for a series of
//#   boxes. (06/06/2012)
//# - The PointsToVtk () and CellsToVtk () are removed from this class and
//#   sent to class JBufferToVtk. (06/06/2012)
//# - Error fixed in SaveVtkBasic() when null arrays were used. (06/10/2012)
//# - A potential precisionproblem in calculating the vertices
//#   of a box was eliminated. (15-05-2013)
//# - Adds a header file using SaveCsv(..., head). (20-11-2013)
//# - New functions SaveCsvPointsVar(), SaveCsvPointsVar3(), SaveAscPointsVar()
//#   and SaveAscPointsVar3() to store the position as tdouble3. (30-12-2013)
//# - New function SaveAscii() using StScalarData* fields. (16-01-2014)
//# - New methods XXXStats() that calculate and store 
//#   minimum, maximum and average values for CSV files. (16-01-2014)
//# - Corrected error in DefineStatsField(). (18-02-2015)
//# - Functions SaveCsv() and SaveCsv() to store POS in double precision. (24-03-2015)
//#############################################################################

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
    if(idp){  fields[nfields]=DefineField("Id",UInt32,1,idp);     nfields++; }
    if(vel){  fields[nfields]=DefineField("Vel",Float32,3,vel);   nfields++; }
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
  static void SaveCsvPointsVar(const std::string &fname,int part,double timestep,unsigned np,const tfloat3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,int part,double timestep,unsigned np,const tfloat3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float.
  //==============================================================================
  static void SaveCsvPointsVar(const std::string &fname,int part,double timestep,unsigned np,const tdouble3* pos,const float* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type double.
  //==============================================================================
  static void SaveCsvPointsVar(const std::string &fname,int part,double timestep,unsigned np,const tdouble3* pos,const double* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type float3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,int part,double timestep,unsigned np,const tdouble3* pos,const tfloat3* data,bool first=false);

  //==============================================================================
  /// Stores information of points in CSV file for variables of type double3.
  //==============================================================================
  static void SaveCsvPointsVar3(const std::string &fname,int part,double timestep,unsigned np,const tdouble3* pos,const tdouble3* data,bool first=false);

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


