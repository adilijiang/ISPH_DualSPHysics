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
//# - Nuevas funciones para gestion de nombres de ficheros: GetFile(), 
//#   GetFileNameSplit(). (10/08/2010)
//# - Nuevas funciones para pasar de valores numericos a texto: UintStr(),
//#   IntStr(). (17/12/2010)
//# - Nueva funcion para ruats de ficheros: GetWithoutExtension(). (22/12/2010)
//# - Funciones para convertir datos entre BigEndian y LittleEndian. (09/03/2011)
//# - Agrupa funciones en namespace fun. (09/03/2011)
//# - Nuevas funciones FileExists() y DirExists(). (10/03/2011)
//# - Correccion en GetExtension() y GetWithoutExtension(), ahora busca la 
//#   extension apartir del punto del ultimo fichero o directorio. (08/05/2011)
//# - Nuevas funciones VarStr() para vectores de datos. (02/11/2011)
//# - Funcion StrSplit() para extraer partes de un texto. (27/01/2012)
//# - Traduccion de comentarios al ingles. (10/02/2012)
//# - Error corregido en ReverseByteOrder(). (21/03/2012)
//# - Nuevas funciones ResizeAlloc para redimensionar la cantidad de memoria
//#   reservada conservando los datos. (22/03/2012)
//# - Nuevas funciones GetDateTimeFormat() y GetTextRandomCode(). (29/09/2012)
//# - Nueva funcion LongStr(). (05/04/2013)
//# - Mejora de funcion StrSplit(). (06/06/2013)
//# - Nuevas funciones StrSplitCount() y StrSplitValue(). (06/06/2013)
//# - Algunas funciones nuevas para tipos double. (14/11/2013)
//# - Nueva funcion StrWithoutChar(). (13/12/2013)
//# - Nueva funcion ResizeAlloc para tfloat4 y tdouble2. (21/12/2013)
//# - Nueva funcion PrintStr usando argumentos como el printf(). (10/03/2014)
//# - Nuevos metodos VarStr para arrays de unsigned y word. (18/03/2014)
//# - Nuevas funcion StrTrimRepeated(). (08/05/2014)
//# - Nuevas funcion StrRepeat(). (03/10/2014)
//# - Nuevas funciones GetFirstValueXXX(). (15/12/2014)
//# - Remplaza long long por llong. (01-10-2015)
//# - EN:
//# Changes:
//# ========
//# - New management functions for file names: GetFile()
//#   GetFileName Split(). (10/08/2010)
//# - New functions to move numeric values ​​to text: UintStr()
//#   IntStr(). (17/12/2010)
//# - New function for file routes: GetWithoutExtension(). (22/12/2010)
//# - Functions for converting data between BigEndian and LittleEndian. (09/03/2011)
//# - Groups functions in namespace fun. (09/03/2011)
//# - New functions FileExists() and DirExists(). (10/03/2011)
//# - Correction in GetExtension() and GetWithoutExtension(), they now seek the
//#   extension starting from the last file or directory. (08/05/2011)
//# - New function varstr() for data vectors. (02/11/2011)
//# - Function StrSplit() to extract text parts. (27/01/2012)
//# - Comment English translation. (10/02/2012)
//# - Error fixed in ReverseByteOrder(). (21/03/2012)
//# - New function ResizeAlloc to resize the memory amount
//# - allocated retaining the data. (22/03/2012)
//# - New functions GetDateTimeFormat() and GetTextRandomCode(). (29/09/2012)
//# - New function LongStr(). (05.04.2013)
//# - Improved StrSplit function(). (06.06.2013)
//# - New functions StrSplitCount() and StrSplitValue(). (06.06.2013)
//# - Some new features for double types. (14.11.2013)
//# - New function StrWithoutChar(). (12.13.2013)
//# - New function for tfloat4 and tdouble2 ResizeAlloc. (21.12.2013)
//# - New function PrintStr using arguments such as printf(). (03.10.2014)
//# - New methods in VarStr for unsigned and word arrays. (18.03.2014)
//# - New function StrTrimRepeated(). (05.08.2014)
//# - New function StrRepeat(). (03.10.2014)
//# - New function GetFirstValueXXX(). (12.15.2014)
//# - Replace long long by llong. (01-10-2015)
//#############################################################################

/// \file Functions.h \brief Declares basic/general functions for the entire application.

#ifndef _Functions_
#define _Functions_

#include <ctime>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "TypesDef.h"

/// Implements a set of basic/general functions.
namespace fun{

std::string GetDateTimeFormat(const char* format,int nseg=0);
inline std::string GetDateTime(){ return(GetDateTimeFormat("%d-%m-%Y %H:%M:%S",0)); }
inline std::string GetDateTimeAfter(int nseg){ return(GetDateTimeFormat("%d-%m-%Y %H:%M:%S",nseg)); }
std::string GetHoursOfSeconds(double s);

std::string GetTextRandomCode(unsigned length);

std::string PrintStr(const char *format,...);

std::string IntStrFill(int v,int vmax);
std::string LongStr(llong v);
std::string UlongStr(ullong v);
std::string UintStr(unsigned v,const char* fmt="%u");
std::string IntStr(int v);
std::string Int3Str(const tint3 &v);
std::string Uint3Str(const tuint3 &v);
/// Converts range of tuint3 values to string.  
inline std::string Uint3RangeStr(const tuint3 &v,const tuint3 &v2){ return(std::string("(")+Uint3Str(v)+")-("+Uint3Str(v2)+")"); }
std::string FloatStr(float v,const char* fmt="%f");
std::string Float3Str(const tfloat3 &v,const char* fmt="%f,%f,%f");
/// Converts real value to string with format g.
inline std::string Float3gStr(const tfloat3 &v){ return(Float3Str(v,"%g,%g,%g")); }
/// Converts range of tfloat3 values to string.  
inline std::string Float3gRangeStr(const tfloat3 &v,const tfloat3 &v2){ return(std::string("(")+Float3gStr(v)+")-("+Float3gStr(v2)+")"); }
std::string DoubleStr(double v,const char* fmt="%g");
std::string Double3Str(const tdouble3 &v,const char* fmt="%f,%f,%f");
inline std::string Double3gStr(const tdouble3 &v){ return(Double3Str(v,"%g,%g,%g")); }
inline std::string Double3gRangeStr(const tdouble3 &v,const tdouble3 &v2){ return(std::string("(")+Double3gStr(v)+")-("+Double3gStr(v2)+")"); }

std::string Double4Str(const tdouble4 &v,const char* fmt="%f,%f,%f");
inline std::string Double4gStr(const tdouble4 &v){ return(Double4Str(v,"%g,%g,%g,%g")); }


std::string StrUpper(const std::string &cad);
std::string StrLower(const std::string &cad);
std::string StrTrim(const std::string &cad);
std::string StrTrimRepeated(const std::string &cad);
std::string StrWithoutChar(const std::string &cad,char let);
std::string StrRepeat(const std::string &cad,unsigned count);
std::string StrSplit(const std::string mark,std::string &text);
unsigned StrSplitCount(const std::string mark,std::string text);
std::string StrSplitValue(const std::string mark,std::string text,unsigned value);
unsigned VectorSplitInt(const std::string mark,const std::string &text,std::vector<int> &vec);

double GetFirstValueDouble(std::string tex,std::string pretex="");
double GetFirstValueDouble(std::string tex,std::string &endtex,std::string pretex);
int GetFirstValueInt(std::string tex,std::string pretex="");
int GetFirstValueInt(std::string tex,std::string &endtex,std::string pretex);

std::string VarStr(const std::string &name,const char *value);
std::string VarStr(const std::string &name,const std::string &value);
std::string VarStr(const std::string &name,float value);
std::string VarStr(const std::string &name,tfloat3 value);
std::string VarStr(const std::string &name,double value);
std::string VarStr(const std::string &name,tdouble3 value);
std::string VarStr(const std::string &name,bool value);
std::string VarStr(const std::string &name,int value);
std::string VarStr(const std::string &name,unsigned value);

std::string VarStr(const std::string &name,unsigned n,const int* values,std::string size="?");
std::string VarStr(const std::string &name,unsigned n,const unsigned* values,std::string size="?");
std::string VarStr(const std::string &name,unsigned n,const word* values,std::string size="?");
std::string VarStr(const std::string &name,unsigned n,const float* values,std::string size="?",const char* fmt="%f");
std::string VarStr(const std::string &name,unsigned n,const double* values,std::string size="?",const char* fmt="%f");

void PrintVar(const std::string &name,const char *value,const std::string &post="");
void PrintVar(const std::string &name,const std::string &value,const std::string &post="");
void PrintVar(const std::string &name,float value,const std::string &post="");
void PrintVar(const std::string &name,double value,const std::string &post="");
void PrintVar(const std::string &name,tfloat3 value,const std::string &post="");
void PrintVar(const std::string &name,tdouble3 value,const std::string &post="");
void PrintVar(const std::string &name,bool value,const std::string &post="");
void PrintVar(const std::string &name,int value,const std::string &post="");
void PrintVar(const std::string &name,unsigned value,const std::string &post="");

int FileType(const std::string &name);
inline bool FileExists(const std::string &name){ return(FileType(name)==2); }
inline bool DirExists(const std::string &name){ return(FileType(name)==1); }

std::string GetDirParent(const std::string &ruta);
std::string GetFile(const std::string &ruta);
std::string GetDirWithSlash(const std::string &ruta);
std::string GetDirWithoutSlash(const std::string &ruta);
std::string GetExtension(const std::string &file);
std::string GetWithoutExtension(const std::string &ruta);
void GetFileNameSplit(const std::string &file,std::string &dir,std::string &fname,std::string &fext);
std::string AddExtension(const std::string &file,const std::string &ext);
std::string FileNameSec(std::string fname,unsigned fnumber);
std::string ShortFileName(const std::string &file,unsigned maxlen,bool withpoints=true);

bool FileMask(std::string text,std::string mask);

typedef enum{ BigEndian=1,LittleEndian=0 }TpByteOrder;
TpByteOrder GetByteOrder();
void ReverseByteOrder(llong *data,int count,llong *result);
void ReverseByteOrder(int *data,int count,int *result);
void ReverseByteOrder(short *data,int count,short *result);
inline void ReverseByteOrder(llong *data,int count){ ReverseByteOrder(data,count,data); }
inline void ReverseByteOrder(int *data,int count){ ReverseByteOrder(data,count,data); }
inline void ReverseByteOrder(short *data,int count){ ReverseByteOrder(data,count,data); }

byte* ResizeAlloc(byte *data,unsigned ndata,unsigned newsize);
word* ResizeAlloc(word *data,unsigned ndata,unsigned newsize);
unsigned* ResizeAlloc(unsigned *data,unsigned ndata,unsigned newsize);
tuint3* ResizeAlloc(tuint3 *data,unsigned ndata,unsigned newsize);
float* ResizeAlloc(float *data,unsigned ndata,unsigned newsize);
tfloat3* ResizeAlloc(tfloat3 *data,unsigned ndata,unsigned newsize);
tfloat4* ResizeAlloc(tfloat4 *data,unsigned ndata,unsigned newsize);
double* ResizeAlloc(double *data,unsigned ndata,unsigned newsize);
tdouble2* ResizeAlloc(tdouble2 *data,unsigned ndata,unsigned newsize);
tdouble3* ResizeAlloc(tdouble3 *data,unsigned ndata,unsigned newsize);


}

#endif




