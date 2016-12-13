﻿/*
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

int      StrToInt    (const std::string &v);
tint3    StrToInt3   (std::string v);
double   StrToDouble (const std::string &v);
tdouble3 StrToDouble3(std::string v);
inline byte     StrToByte   (const std::string &v){ return(byte(StrToInt(v)));          }
inline word     StrToWord   (const std::string &v){ return(word(StrToInt(v)));          }
inline unsigned StrToUint   (const std::string &v){ return(unsigned(StrToInt(v)));      }
inline tuint3   StrToUint3  (const std::string &v){ return(ToTUint3(StrToInt3(v)));     }
inline float    StrToFloat  (const std::string &v){ return(float(StrToDouble(v)));      }
inline tfloat3  StrToFloat3 (const std::string &v){ return(ToTFloat3(StrToDouble3(v))); }

std::string StrUpper(const std::string &cad);
std::string StrLower(const std::string &cad);
std::string StrTrim(const std::string &cad);
std::string StrTrimRepeated(const std::string &cad);
std::string StrWithoutChar(const std::string &cad,char let);
std::string StrRepeat(const std::string &cad,unsigned count);
bool StrOnlyChars(const std::string &cad,const std::string &chars);

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

byte*     ResizeAlloc(byte     *data,unsigned ndata,unsigned newsize);
word*     ResizeAlloc(word     *data,unsigned ndata,unsigned newsize);
unsigned* ResizeAlloc(unsigned *data,unsigned ndata,unsigned newsize);
tuint3*   ResizeAlloc(tuint3   *data,unsigned ndata,unsigned newsize);
int*      ResizeAlloc(int      *data,unsigned ndata,unsigned newsize);
tint3*    ResizeAlloc(tint3    *data,unsigned ndata,unsigned newsize);
float*    ResizeAlloc(float    *data,unsigned ndata,unsigned newsize);
tfloat2*  ResizeAlloc(tfloat2  *data,unsigned ndata,unsigned newsize);
tfloat3*  ResizeAlloc(tfloat3  *data,unsigned ndata,unsigned newsize);
tfloat4*  ResizeAlloc(tfloat4  *data,unsigned ndata,unsigned newsize);
double*   ResizeAlloc(double   *data,unsigned ndata,unsigned newsize);
tdouble2* ResizeAlloc(tdouble2 *data,unsigned ndata,unsigned newsize);
tdouble3* ResizeAlloc(tdouble3 *data,unsigned ndata,unsigned newsize);

bool IsInfinity(float v);
bool IsInfinity(double v);
bool IsNAN(float v);
bool IsNAN(double v);

}

#endif




