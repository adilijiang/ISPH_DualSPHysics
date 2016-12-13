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

/// \file Functions.cpp \brief Implements basic/general functions for the entire application.

#include "Functions.h"
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdarg.h>
#include <algorithm>

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

namespace fun{

//==============================================================================
/// Returns date and time of the system + nseg using the format.
//==============================================================================
std::string GetDateTimeFormat(const char* format,int nseg){
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  rawtime+=nseg;
  timeinfo=localtime(&rawtime);
  char bufftime[64];
  strftime(bufftime,64,format,timeinfo);
  return(bufftime);
}

//==============================================================================
/// Returns duration in format xh ym zs.
//==============================================================================
std::string GetHoursOfSeconds(double s){
  int hours=int(s/3600);
  s-=double(hours*3600);
  int mins=int(s/60);
  s-=double(mins*60);
  char cad[64];
  sprintf(cad,"%dh %dm %.1fs",hours,mins,s);
  return(cad);
}

//==============================================================================
/// Returns text with the random code.
//==============================================================================
std::string GetTextRandomCode(unsigned length){
  const unsigned maxlength=1024;
  if(length<1)length=1;
  if(length>1024)length=1024;
  char code[maxlength+1];
  srand((unsigned)time(NULL));
  for(unsigned c=0;c<length;c++){
    char let=char(float(rand())/float(RAND_MAX)*36);
    code[c]=(let<10? let+48: let+87);
  } 
  code[length]=0;
  return(code);
}
  
//==============================================================================
/// Returns string using the same parameters used in printf().
//==============================================================================
std::string PrintStr(const char *format,...){
  std::string ret;
  const unsigned SIZE=1024;
  char buffer[SIZE+1];
  va_list args;
  va_start(args, format);
  unsigned size=vsnprintf(buffer,SIZE,format,args);
  if(size<SIZE)ret=buffer;
  else{
    char *buff=new char[size+1];
    vsnprintf(buff,size,format,args);
    ret=buff;
    delete[] buff;
  }
  va_end(args);
  return(ret);
}

//==============================================================================
/// Converts unsigned value to string filling with zeros.
//==============================================================================
std::string IntStrFill(int v,int vmax){
  unsigned len=unsigned(UintStr(vmax).length());
  std::string value=IntStr(v);
  while(unsigned(value.length())<len)value=std::string("0")+value;
  return(value);
}

//==============================================================================
/// Converts long long value to string.
//==============================================================================
std::string LongStr(llong v){
  char cad[128];
  sprintf(cad,"%lld",v);
  return(std::string(cad));
}

//==============================================================================
/// Converts unsigned long long value to string.
//==============================================================================
std::string UlongStr(ullong v){
  char cad[128];
  sprintf(cad,"%llu",v);
  return(std::string(cad));
}

//==============================================================================
/// Converts unsigned value to string.
//==============================================================================
std::string UintStr(unsigned v,const char* fmt){
  char cad[128];
  sprintf(cad,fmt,v);
  return(std::string(cad));
}

//==============================================================================
/// Converts int value to string.
//==============================================================================
std::string IntStr(int v){
  char cad[128];
  sprintf(cad,"%d",v);
  return(std::string(cad));
}

//==============================================================================
/// Converts tint3 value to string.
//==============================================================================
std::string Int3Str(const tint3 &v){
  char cad[128];
  sprintf(cad,"%d,%d,%d",v.x,v.y,v.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts tuint3 value to string.
//==============================================================================
std::string Uint3Str(const tuint3 &v){
  char cad[128];
  sprintf(cad,"%u,%u,%u",v.x,v.y,v.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string FloatStr(float v,const char* fmt){
  char cad[128];
  sprintf(cad,fmt,v);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string Float3Str(const tfloat3 &v,const char* fmt){
  char cad[1024];
  sprintf(cad,fmt,v.x,v.y,v.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string DoubleStr(double v,const char* fmt){
  char cad[256];
  sprintf(cad,fmt,v);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string Double3Str(const tdouble3 &v,const char* fmt){
  char cad[1024];
  sprintf(cad,fmt,v.x,v.y,v.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string Double4Str(const tdouble4 &v,const char* fmt){
  char cad[1024];
  sprintf(cad,fmt,v.x,v.y,v.z,v.w);
  return(std::string(cad));
}

//==============================================================================
/// Converts string to int value.
//==============================================================================
int StrToInt(const std::string &v){
  return(atoi(v.c_str()));
}

//==============================================================================
/// Converts string to tint3 value.
//==============================================================================
tint3 StrToInt3(std::string v){
  tint3 res=TInt3(0);
  if(!v.empty())res.x=atoi(fun::StrSplit(",",v).c_str());
  if(!v.empty())res.y=atoi(fun::StrSplit(",",v).c_str());
  if(!v.empty())res.z=atoi(fun::StrSplit(",",v).c_str());
  return(res);
}

//==============================================================================
/// Converts string to double value.
//==============================================================================
double StrToDouble(const std::string &v){
  return(atof(v.c_str()));
}

//==============================================================================
/// Converts string to tdouble3 value.
//==============================================================================
tdouble3 StrToDouble3(std::string v){
  tdouble3 res=TDouble3(0);
  if(!v.empty())res.x=atof(fun::StrSplit(",",v).c_str());
  if(!v.empty())res.y=atof(fun::StrSplit(",",v).c_str());
  if(!v.empty())res.z=atof(fun::StrSplit(",",v).c_str());
  return(res);
}

//==============================================================================
/// Gets string in uppercase.
//==============================================================================
std::string StrUpper(const std::string &cad){
  std::string ret;
  for(unsigned c=0;c<cad.length();c++)ret=ret+char(toupper(cad[c]));
  return(ret);
}

//==============================================================================
/// Gets string in lowercase.
//==============================================================================
std::string StrLower(const std::string &cad){
  std::string ret;
  for(unsigned c=0;c<cad.length();c++)ret=ret+char(tolower(cad[c]));
  return(ret);
}

//==============================================================================
/// Gets string without spaces at the beginning and end.
//==============================================================================
std::string StrTrim(const std::string &cad){
  std::string ret;
  int lsp=0,rsp=0;
  for(int c=0;c<int(cad.length())&&cad[c]==' ';c++)lsp++;
  for(int c=int(cad.length())-1;c<int(cad.length())&&cad[c]==' ';c--)rsp++;
  int size=int(cad.length())-(lsp+rsp);
  return(size>0? cad.substr(lsp,size): "");
}

//==============================================================================
/// Gets string without repeated spaces.
//==============================================================================
std::string StrTrimRepeated(const std::string &cad){
  std::string ret;
  bool lastsp=false;
  for(int c=0;c<int(cad.length());c++){
    const char let=cad[c];
    if(!lastsp || let!=' '){
      ret=ret+let;
      lastsp=(let==' ');
    }
  }
  return(ret);
}

//==============================================================================
/// Gets string without the character indicated.
//==============================================================================
std::string StrWithoutChar(const std::string &cad,char let){
  std::string ret;
  for(int c=0;c<int(cad.length());c++)if(cad[c]!=let)ret=ret+cad[c];
  return(ret);
}

//==============================================================================
/// Gets string with the string indicated n times.
//==============================================================================
std::string StrRepeat(const std::string &cad,unsigned count){
  std::string ret;
  for(unsigned c=0;c<count;c++)ret=ret+cad;
  return(ret);
}

//==============================================================================
/// Inidicates if the string cad only contains characters in the string chars.
//==============================================================================
bool StrOnlyChars(const std::string &cad,const std::string &chars){
  bool ok=true;
  const unsigned nc=unsigned(chars.length());
  for(int c=0;c<int(cad.length()) && ok;c++){
    const char let=cad[c];
    unsigned c2=0;
    for(;c2<nc && chars[c2]!=let;c2++);
    if(c2>=nc)ok=false;
  }
  return(ok);
}

//==============================================================================
/// Returns the text untill the indicated mark and saves the rest in text format.
//==============================================================================
std::string StrSplit(const std::string mark,std::string &text){
  const unsigned smark=unsigned(mark.size());
  int tpos=int(text.find(mark));
  std::string ret=(tpos>=0? text.substr(0,tpos): text);
  text=(tpos>=0? text.substr(tpos+smark): "");
  return(ret);
}

//==============================================================================
/// Returns the number of pieces of a string.
//==============================================================================
unsigned StrSplitCount(const std::string mark,std::string text){
  const unsigned smark=unsigned(mark.size());
  unsigned count=0;
  while(!text.empty()){
    int tpos=int(text.find(mark));
    //std::string ret=(tpos>=0? text.substr(0,tpos): text);
    text=(tpos>=0? text.substr(tpos+smark): "");
    count++;
  }
  return(count);
}

//==============================================================================
/// Returns the indicated piece of a string.
//==============================================================================
std::string StrSplitValue(const std::string mark,std::string text,unsigned value){
  const unsigned smark=unsigned(mark.size());
  std::string ret="";
  unsigned count=0;
  while(!text.empty()){
    int tpos=int(text.find(mark));
    if(count==value){
      ret=(tpos>=0? text.substr(0,tpos): text);
      text="";
    }
    else text=(tpos>=0? text.substr(tpos+smark): "");
    count++;
  }
  return(ret);
}

//==============================================================================
/// Loads unsigned list in a vector and returns size of vector.
//==============================================================================
unsigned VectorSplitInt(const std::string mark,const std::string &text,std::vector<int> &vec){
  std::string aux=text;
  while(!aux.empty()){
    std::string txv=StrSplit(mark,aux);
    if(!txv.empty())vec.push_back(atoi(txv.c_str()));
  }
  return((unsigned)vec.size());
}



//==============================================================================
/// Returns first double value after "pretex".
//==============================================================================
double GetFirstValueDouble(std::string tex,std::string pretex){
  if(!pretex.empty()){//-Elimina texto previo si lo hubiera.
    int pre=int(tex.find(pretex));
    if(pre>=0)tex=tex.substr(pre);
  }
  int pini=int(strcspn(tex.c_str(),"0123456789-"));//-Localiza principio de numero.
  tex=tex.substr(pini);
  int len=int(strspn(tex.c_str(),"0123456789-."));//-Calcula longitud de numero.
  return(atof(tex.substr(0,len).c_str()));
}

//==============================================================================
/// Returns first double value after "pretex" and returns the remaining text.
//==============================================================================
double GetFirstValueDouble(std::string tex,std::string &endtex,std::string pretex){
  if(!pretex.empty()){//-Elimina texto previo si lo hubiera.
    int pre=int(tex.find(pretex));
    if(pre>=0)tex=tex.substr(pre);
  }
  int pini=int(strcspn(tex.c_str(),"0123456789-"));//-Localiza principio de numero.
  tex=tex.substr(pini);
  int len=int(strspn(tex.c_str(),"0123456789-."));//-Calcula longitud de numero.
  endtex=tex.substr(len);
  return(atof(tex.substr(0,len).c_str()));
}

//==============================================================================
/// Returns first int value after "pretex".
//==============================================================================
int GetFirstValueInt(std::string tex,std::string pretex){
  if(!pretex.empty()){//-Elimina texto previo si lo hubiera.
    int pre=int(tex.find(pretex));
    if(pre>=0)tex=tex.substr(pre);
  }
  int pini=int(strcspn(tex.c_str(),"0123456789-"));//-Localiza principio de numero.
  tex=tex.substr(pini);
  int len=int(strspn(tex.c_str(),"0123456789-"));//-Calcula longitud de numero.
  return(atoi(tex.substr(0,len).c_str()));
}

//==============================================================================
/// Returns first int value after "pretex" and returns the remaining text.
//==============================================================================
int GetFirstValueInt(std::string tex,std::string &endtex,std::string pretex){
  if(!pretex.empty()){//-Elimina texto previo si lo hubiera.
    int pre=int(tex.find(pretex));
    if(pre>=0)tex=tex.substr(pre);
  }
  int pini=int(strcspn(tex.c_str(),"0123456789-"));//-Localiza principio de numero.
  tex=tex.substr(pini);
  int len=int(strspn(tex.c_str(),"0123456789-"));//-Calcula longitud de numero.
  endtex=tex.substr(len);
  return(atoi(tex.substr(0,len).c_str()));
}



//==============================================================================
/// Returns variable and its value in text format.
//==============================================================================
std::string VarStr(const std::string &name,const char *value){ return(name+"=\""+value+"\""); }
std::string VarStr(const std::string &name,const std::string &value){ return(name+"=\""+value+"\""); }
std::string VarStr(const std::string &name,float value){ return(name+"="+FloatStr(value)); }
std::string VarStr(const std::string &name,tfloat3 value){ return(name+"=("+FloatStr(value.x)+","+FloatStr(value.y)+","+FloatStr(value.z)+")"); }
std::string VarStr(const std::string &name,double value){ return(name+"="+DoubleStr(value)); }
std::string VarStr(const std::string &name,tdouble3 value){ return(name+"=("+DoubleStr(value.x)+","+DoubleStr(value.y)+","+DoubleStr(value.z)+")"); }
std::string VarStr(const std::string &name,bool value){ return(name+"="+(value? "True": "False")+""); }
std::string VarStr(const std::string &name,int value){
  char cad[30];
  sprintf(cad,"=%d",value);
  return(name+cad);
}
std::string VarStr(const std::string &name,unsigned value){
  char cad[30];
  sprintf(cad,"=%u",value);
  return(name+cad);
}
std::string VarStr(const std::string &name,unsigned n,const int* values,std::string size){
  std::string tex=name+"["+(size=="?"? UintStr(n): size)+"]=[";
  for(unsigned c=0;c<n;c++)tex=tex+(c? ",": "")+fun::IntStr(values[c]);
  return(tex+"]");
}
std::string VarStr(const std::string &name,unsigned n,const unsigned* values,std::string size){
  std::string tex=name+"["+(size=="?"? UintStr(n): size)+"]=[";
  for(unsigned c=0;c<n;c++)tex=tex+(c? ",": "")+fun::UintStr(values[c]);
  return(tex+"]");
}
std::string VarStr(const std::string &name,unsigned n,const word* values,std::string size){
  std::string tex=name+"["+(size=="?"? UintStr(n): size)+"]=[";
  for(unsigned c=0;c<n;c++)tex=tex+(c? ",": "")+fun::UintStr(values[c]);
  return(tex+"]");
}
std::string VarStr(const std::string &name,unsigned n,const float* values,std::string size,const char* fmt){
  std::string tex=name+"["+(size=="?"? UintStr(n): size)+"]=[";
  for(unsigned c=0;c<n;c++)tex=tex+(c? ",": "")+fun::FloatStr(values[c],fmt);
  return(tex+"]");
}
std::string VarStr(const std::string &name,unsigned n,const double* values,std::string size,const char* fmt){
  std::string tex=name+"["+(size=="?"? UintStr(n): size)+"]=[";
  for(unsigned c=0;c<n;c++)tex=tex+(c? ",": "")+fun::DoubleStr(values[c],fmt);
  return(tex+"]");
}

//==============================================================================
/// Prints on the screen a variable with its value.
//==============================================================================
void PrintVar(const std::string &name,const char *value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,const std::string &value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,float value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,double value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,tfloat3 value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,tdouble3 value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,bool value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,int value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,unsigned value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
    

//##############################################################################
//##############################################################################
//##############################################################################
//==============================================================================
/// Returns information about a file, indicates whether file or directory.
/// 0:No exists, 1:Directory, 2:File
//==============================================================================
int FileType(const std::string &name){
  int ret=0;
  struct stat stfileinfo;
  int intstat=stat(name.c_str(),&stfileinfo);
  if(intstat==0){
    if(stfileinfo.st_mode&S_IFDIR)ret=1;
    if(stfileinfo.st_mode&S_IFREG)ret=2;
  }
  return(ret);
}

//==============================================================================
/// Returns the parent directory with its path.
//==============================================================================
std::string GetDirParent(const std::string &ruta){
  std::string dir;
  int pos=int(ruta.find_last_of("/"));
  if(pos<=0)pos=int(ruta.find_last_of("\\"));
  if(pos>0)dir=ruta.substr(0,pos);
  return(dir);
}

//==============================================================================
/// Returns the filename or directory of a path.
//==============================================================================
std::string GetFile(const std::string &ruta){
  std::string file;
  int c;
  for(c=int(ruta.size())-1;c>=0&&ruta[c]!='\\'&&ruta[c]!='/';c--);
  file=(c<0? ruta: ruta.substr(c+1));
  return(file);
}

//==============================================================================
/// Returns the path with slash.
//==============================================================================
std::string GetDirWithSlash(const std::string &ruta){
  std::string rut=ruta;
  if(ruta!=""){
    char last=ruta[ruta.length()-1];
    if(last!='\\'&&last!='/')rut=ruta+"/";
  }
  return(rut);
}

//==============================================================================
/// Returns the path without slash.
//==============================================================================
std::string GetDirWithoutSlash(const std::string &ruta){
  char last=ruta[ruta.length()-1];
  if(last=='\\'||last=='/')return(ruta.substr(0,ruta.length()-1));
  return(ruta);
}

//==============================================================================
/// Returns the extension of a file.
//==============================================================================
std::string GetExtension(const std::string &file){
  std::string ext;
  int pos=(int)file.find_last_of(".");
  int posmin=std::max((int)file.find_last_of("/"),(int)file.find_last_of("\\"));
  if(pos>=0&&pos>posmin)ext=file.substr(pos+1);
  return(ext);
}

//==============================================================================
/// Returns the path of a file without the extension (and without the point).
//==============================================================================
std::string GetWithoutExtension(const std::string &ruta){
  int pos=(int)ruta.find_last_of(".");
  int posmin=std::max((int)ruta.find_last_of("/"),(int)ruta.find_last_of("\\"));
  return(pos>=0&&pos>posmin? ruta.substr(0,pos): ruta);
}

//==============================================================================
/// Returns the parent directory, name and extension of a file.
//==============================================================================
void GetFileNameSplit(const std::string &file,std::string &dir,std::string &fname,std::string &fext){
  dir=GetDirParent(file);
  fname=GetFile(file);
  fext=GetExtension(fname);
  if(!fext.empty())fname=fname.substr(0,fname.size()-fext.size()-1);
}

//==============================================================================
/// Adds extension (without point) to the path of a file.
//==============================================================================
std::string AddExtension(const std::string &file,const std::string &ext){
  std::string file2=file;
  if(file2.empty()||file2[file2.length()-1]!='.')file2+='.';
  file2+=ext;
  return(file2);
}

//==============================================================================
/// Returns the filename with number.
//==============================================================================
std::string FileNameSec(std::string fname,unsigned fnumber){
  std::string fext=GetExtension(fname);
  if(!fext.empty())fname=fname.substr(0,fname.size()-fext.size()-1);
  char cad[64];
  sprintf(cad,"_%04d.",fnumber);
  return(fname+cad+fext);
}

//==============================================================================
/// Returns the filename with a requested size of characteres.
//==============================================================================
std::string ShortFileName(const std::string &file,unsigned maxlen,bool withpoints){
  std::string file2;
  if(file.length()<=maxlen)file2=file;
  else{
    file2=file.substr(file.length()-maxlen);
    int pos1=(int)file2.find("\\");
    int pos2=(int)file2.find("/");
    if(pos1<0||(pos2>=0&&pos2<pos1))pos1=pos2;
    if(pos1>=0)file2=file2.substr(pos1);
    if(withpoints){
      if(file2.length()+3>maxlen)file2=ShortFileName(file2,maxlen-3,false);
      file2=std::string("...")+file2;
    }
  }
  return(file2);
}

//==============================================================================
/// Indicates whether there is concordance between filename and mask.
///  Following special characters can be used:
///   - '?': Replaces for any character.
///   - '*': Replaces one or several characters to zero.
///   - '|': Allows to indicate several masks. Example: *.vtk|*.ply
//==============================================================================
bool FileMask(std::string text,std::string mask){
  /*/-Removes '*' consecutives.
  int pos=(int)mask.find("**");
  while(pos>=0){
    mask=mask.substr(0,pos)+mask.substr(pos+1);
    pos=(int)mask.find("**");
  }*/
  //-Checks multiple masks.
  int pos=(int)mask.find("|");
  if(pos>=0)return(FileMask(text,mask.substr(0,pos))||FileMask(text,mask.substr(pos+1)));
  else{
  //-Checks corrleation of text with mask.
    int stext=(int)text.length();
    int smask=(int)mask.length();
    if(!stext&&!smask)return(true);
    else if(smask==1&&mask[0]=='*')return(true);
    else if((smask&&!stext)||(!smask&&stext))return(false);
    else if(mask[0]!='*'){
      if(mask[0]=='?'||mask[0]==text[0])return(FileMask(text.substr(1),mask.substr(1)));
      else return(false);
    }
    else{
      bool res=false;
      for(int c=0;c<stext;c++)res|=FileMask(text.substr(c),mask.substr(1));
      return(res);
    }
  }
} 


//##############################################################################
//##############################################################################
//##############################################################################
//==============================================================================
/// Returns the type of codification using BigEndian or LittleEndian.
//==============================================================================
TpByteOrder GetByteOrder(){
  int i=1;
  return(*((char*)&i)==1? LittleEndian: BigEndian);
}

//==============================================================================
/// Reverses the order of the bytes to exchange BigEndian and LittleEndian.
//==============================================================================
void ReverseByteOrder(llong *data,int count,llong *result){
  for(int c=0;c<count;c++){
    unsigned int v=((unsigned int*)data)[c*2+1];
    unsigned int v2=((unsigned int*)data)[c*2];
    ((unsigned int*)result)[c*2]=((v<<24)&0xFF000000)|((v<<8)&0x00FF0000)|((v>>8)&0x0000FF00)|((v>>24)&0x000000FF);
    ((unsigned int*)result)[c*2+1]=((v2<<24)&0xFF000000)|((v2<<8)&0x00FF0000)|((v2>>8)&0x0000FF00)|((v2>>24)&0x000000FF);
  }
}
//==============================================================================
void ReverseByteOrder(int *data,int count,int *result){
  for(int c=0;c<count;c++){
    unsigned int v=((unsigned int*)data)[c];
    result[c]=((v<<24)&0xFF000000)|((v<<8)&0x00FF0000)|((v>>8)&0x0000FF00)|((v>>24)&0x000000FF);
  }
}
//==============================================================================
void ReverseByteOrder(short *data,int count,short *result){
  for(int c=0;c<count;c++){
    unsigned short v=((unsigned short*)data)[c];
    result[c]=((v<<8)&0xFF00)|((v>>8)&0x00FF);
  }
}


//==============================================================================
/// Resizes the allocated memory, keeping the data.
//==============================================================================
byte* ResizeAlloc(byte *data,unsigned ndata,unsigned newsize){
  byte* data2=new byte[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(byte)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
word* ResizeAlloc(word *data,unsigned ndata,unsigned newsize){
  word* data2=new word[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(word)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
unsigned* ResizeAlloc(unsigned *data,unsigned ndata,unsigned newsize){
  unsigned* data2=new unsigned[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(unsigned)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tuint3* ResizeAlloc(tuint3 *data,unsigned ndata,unsigned newsize){
  tuint3* data2=new tuint3[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tuint3)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
int* ResizeAlloc(int *data,unsigned ndata,unsigned newsize){
  int* data2=new int[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(unsigned)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tint3* ResizeAlloc(tint3 *data,unsigned ndata,unsigned newsize){
  tint3* data2=new tint3[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tuint3)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
float* ResizeAlloc(float *data,unsigned ndata,unsigned newsize){
  float* data2=new float[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(float)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tfloat2* ResizeAlloc(tfloat2 *data,unsigned ndata,unsigned newsize){
  tfloat2* data2=new tfloat2[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tfloat2)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tfloat3* ResizeAlloc(tfloat3 *data,unsigned ndata,unsigned newsize){
  tfloat3* data2=new tfloat3[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tfloat3)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tfloat4* ResizeAlloc(tfloat4 *data,unsigned ndata,unsigned newsize){
  tfloat4* data2=new tfloat4[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tfloat4)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
double* ResizeAlloc(double *data,unsigned ndata,unsigned newsize){
  double* data2=new double[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(double)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tdouble2* ResizeAlloc(tdouble2 *data,unsigned ndata,unsigned newsize){
  tdouble2* data2=new tdouble2[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tdouble2)*ndata);
  delete[] data;
  return(data2);
}
//==============================================================================
tdouble3* ResizeAlloc(tdouble3 *data,unsigned ndata,unsigned newsize){
  tdouble3* data2=new tdouble3[newsize];
  ndata=std::min(ndata,newsize);
  if(ndata)memcpy(data2,data,sizeof(tdouble3)*ndata);
  delete[] data;
  return(data2);
}


//==============================================================================
/// Returns if float value is + or - infinity.
//==============================================================================
bool IsInfinity(float v){
 return(std::numeric_limits<float>::has_infinity && (v==std::numeric_limits<float>::infinity() || v==-std::numeric_limits<float>::infinity()));
 //return(v > FLT_MAX || v < -FLT_MAX); //-Otra opcion mas sencilla.
}

//==============================================================================
/// Returns if double value is + or - infinity.
//==============================================================================
bool IsInfinity(double v){
 return(std::numeric_limits<float>::has_infinity && (v==std::numeric_limits<float>::infinity() || v==-std::numeric_limits<float>::infinity()));
 //return(v > DBL_MAX || v < -DBL_MAX); //-Otra opcion mas sencilla.
}

//==============================================================================
/// Returns if float value is + or - infinity.
//==============================================================================
bool IsNAN(float v){
 return(v!=v);
}

//==============================================================================
/// Returns if double value is + or - infinity.
//==============================================================================
bool IsNAN(double v){
 return(v!=v);
}

}






