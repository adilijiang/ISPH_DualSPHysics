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

/// \file JPtxasInfo.cpp \brief Implements the class \ref JPtxasInfo.

#include "JPtxasInfo.h"
#include "Functions.h"
#include <algorithm>
#include <cstring>

using std::string;
using std::ifstream;
using std::ofstream;
using std::endl;

//##############################################################################
//# JPtxasInfoKer
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JPtxasInfoKer::JPtxasInfoKer(){
  ClassName="JPtxasInfoKer";
  Reset();
}

//==============================================================================
/// Constructor.
//==============================================================================
JPtxasInfoKer::JPtxasInfoKer(const JPtxasInfoKer &ker){
  ClassName="JPtxasInfoKer";
  Reset();
  *this=ker;
}

//==============================================================================
/// Overloading operator for a correct allocation.
//==============================================================================
JPtxasInfoKer &JPtxasInfoKer::operator=(const JPtxasInfoKer &ker){
  SetName(ker.GetName());
  SetNameSpace(ker.GetNameSpace());
  SetArgs(ker.GetArgs());
  SetRegs(10,ker.GetRegs(10));
  SetRegs(13,ker.GetRegs(13));
  SetRegs(20,ker.GetRegs(20));
  SetRegs(30,ker.GetRegs(30));
  SetRegs(35,ker.GetRegs(35));
  SetStackFrame(10,ker.GetStackFrame(10));
  SetStackFrame(13,ker.GetStackFrame(13));
  SetStackFrame(20,ker.GetStackFrame(20));
  SetStackFrame(30,ker.GetStackFrame(30));
  SetStackFrame(35,ker.GetStackFrame(35));
  for(unsigned c=0;c<ker.CountTemplateArgs();c++)AddTemplateArg(ker.GetTemplateArgsType(c),ker.GetTemplateArgsValue(c));
  return(*this);
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPtxasInfoKer::Reset(){
  Code=""; CodeMin="";
  NameSpace=""; Name=""; Args="";
  Regs_sm10=0; Regs_sm13=0; Regs_sm20=0; Regs_sm30=0; Regs_sm35=0;
  StackFrame_sm10=0; StackFrame_sm13=0; StackFrame_sm20=0; StackFrame_sm30=0; StackFrame_sm35=0;
  for(unsigned c=0;c<TemplateArgs.size();c++)delete TemplateArgs[c];
  TemplateArgs.clear();
}

//==============================================================================
/// Adds an argument to the template.
//==============================================================================
void JPtxasInfoKer::UpdateCode(){
  CodeMin=NameSpace+"_"+Name;
  for(unsigned c=0;c<TemplateArgs.size();c++)CodeMin=CodeMin+"_"+TemplateArgs[c]->value;
  Code=CodeMin+"_"+Args;
}

//==============================================================================
/// Adds an argument to the template.
//==============================================================================
void JPtxasInfoKer::AddTemplateArg(const std::string &type,const std::string &value){
  StTemplateArg* arg=new StTemplateArg;
  arg->type=type; arg->value=value;
  TemplateArgs.push_back(arg);
  UpdateCode();
}

//==============================================================================
/// Modifies the number of registers according to sm.
//==============================================================================
void JPtxasInfoKer::SetRegs(unsigned sm,unsigned regs){
  if(sm==10)Regs_sm10=regs;
  else if(sm==13)Regs_sm13=regs;
  else if(sm==20)Regs_sm20=regs;
  else if(sm==30)Regs_sm30=regs;
  else if(sm==35)Regs_sm35=regs;
}

//==============================================================================
/// Returns the number of registers according to sm.
//==============================================================================
unsigned JPtxasInfoKer::GetRegs(unsigned sm)const{
  if(sm==10)return(Regs_sm10);
  else if(sm==13)return(Regs_sm13);
  else if(sm==20)return(Regs_sm20);
  else if(sm==30)return(Regs_sm30);
  else if(sm==35)return(Regs_sm35);
  return(0);
}

//==============================================================================
/// Modifies the number of bytes from local memory according to sm.
//==============================================================================
void JPtxasInfoKer::SetStackFrame(unsigned sm,unsigned mem){
  if(sm==10)StackFrame_sm10=mem;
  else if(sm==13)StackFrame_sm13=mem;
  else if(sm==20)StackFrame_sm20=mem;
  else if(sm==30)StackFrame_sm30=mem;
  else if(sm==35)StackFrame_sm35=mem;
}

//==============================================================================
/// Returns the number of bytes from local memory according to sm.
//==============================================================================
unsigned JPtxasInfoKer::GetStackFrame(unsigned sm)const{
  if(sm==10)return(StackFrame_sm10);
  else if(sm==13)return(StackFrame_sm13);
  else if(sm==20)return(StackFrame_sm20);
  else if(sm==30)return(StackFrame_sm30);
  else if(sm==35)return(StackFrame_sm35);
  return(0);
}

//==============================================================================
/// Shows data for debug.
//==============================================================================
void JPtxasInfoKer::Print()const{
  printf("JPtxasInfoKer{\n");
  printf("  Code=[%s]\n",Code.c_str());
  printf("  Name=[%s]\n",Name.c_str());
  printf("  Args=[%s]\n",Args.c_str());
  printf("  Regs_sm10=%u  StackFrame_sm10=%u\n",Regs_sm10,StackFrame_sm10);
  printf("  Regs_sm13=%u  StackFrame_sm13=%u\n",Regs_sm13,StackFrame_sm13);
  printf("  Regs_sm20=%u  StackFrame_sm20=%u\n",Regs_sm20,StackFrame_sm20);
  printf("  Regs_sm30=%u  StackFrame_sm30=%u\n",Regs_sm30,StackFrame_sm30);
  printf("  Regs_sm35=%u  StackFrame_sm35=%u\n",Regs_sm35,StackFrame_sm35);
  for(unsigned c=0;c<TemplateArgs.size();c++)printf("  TemplateArgs[%u]=[%s]:[%s]\n",c,TemplateArgs[c]->type.c_str(),TemplateArgs[c]->value.c_str());
}

//##############################################################################
//# JPtxasInfo
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JPtxasInfo::JPtxasInfo(){
  ClassName="JPtxasInfo";
  SmValues[0]=10;
  SmValues[1]=13;
  SmValues[2]=20;
  SmValues[3]=30;
  SmValues[4]=35;
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPtxasInfo::Reset(){
  for(unsigned c=0;c<Kernels.size();c++)delete Kernels[c];
  Kernels.clear();
}

//==============================================================================
/// Returns the index of the requested kernel (-1: It does not exist)
//==============================================================================
int JPtxasInfo::GetIndexKernel(const std::string &code)const{
  int pos=-1;
  for(unsigned c=0;c<Kernels.size()&&pos<0;c++)if(Kernels[c]->GetCode()==code)pos=c;
  return(pos);
}

//==============================================================================
/// Returns the pointer to the requested kernel.
//==============================================================================
const JPtxasInfoKer* JPtxasInfo::GetKernel(unsigned pos)const{ 
  const char met[]="GetKernel";
  if(pos>=Count())RunException(met,"Kernel number requested is not valid.");
  return(Kernels[pos]);
}

//==============================================================================
/// Adds a new kernel.
//==============================================================================
void JPtxasInfo::AddKernel(const JPtxasInfoKer &ker){
  int pos=GetIndexKernel(ker.GetCode());
  if(pos<0){
    JPtxasInfoKer* k=new JPtxasInfoKer(ker);
    Kernels.push_back(k);
    pos=int(Kernels.size())-1;
  }
  for(unsigned c=0;c<SM_COUNT;c++){
    unsigned sm=SmValues[c];
    if(ker.GetRegs(sm))Kernels[pos]->SetRegs(sm,ker.GetRegs(sm));
    if(ker.GetStackFrame(sm))Kernels[pos]->SetStackFrame(sm,ker.GetStackFrame(sm));
  }
}

//==============================================================================
/// Loads ptxas info from the file obtained after compilation.
//==============================================================================
void JPtxasInfo::LoadFile(const std::string &file){
  const char met[]="LoadFile";
  ifstream pf;
  pf.open(file.c_str());
  if(pf){
    JPtxasInfoKer ker;
    unsigned kersm;
    unsigned stackframe;
    bool fin=false;
    while(!pf.eof()&&!fin){
      char buff[1024];
      pf.getline(buff,1024);
      string line=buff;
      int pos=int(line.find("ptxas info"));
      if(pos<0)pos=int(line.find("ptxas : info"));
        if(pos>=0 && int(line.find("Function properties"))<0){
        pos=int(line.find("Compiling entry function '_Z")); 
        int pos2=int(line.find("' for 'sm_")); 
        if(pos>=0 && pos2>=0){//-Name of the kernel.
          ker.Reset(); kersm=0; stackframe=0;
          //-Obtains the compute capability.
          string tx=line.substr(pos2+string("' for 'sm_").length());
          int len=int(strspn(tx.c_str(),"0123456789"));
          kersm=atoi((tx.substr(0,len)).c_str());
          //-Obtains the name of namespace.
          tx=line.substr(0,pos2);
          tx=tx.substr(pos+string("Compiling entry function '_Z").length());
          if(!tx.empty()&&tx[0]=='N'){
            tx=tx.substr(1);
            len=int(strspn(tx.c_str(),"0123456789"));
            int n=atoi((tx.substr(0,len)).c_str());
            ker.SetNameSpace(tx.substr(len,n));
            tx=tx.substr(len+n);
          }
          //-Obtains name of the function.
          len=int(strspn(tx.c_str(),"0123456789"));
          int n=atoi((tx.substr(0,len)).c_str());
          ker.SetName(tx.substr(len,n));
          tx=tx.substr(len+n);
          //-Obtains parameters of the template.
          if(!tx.empty()&&tx[0]=='I'){
            tx=tx.substr(1);
            while(!tx.empty()&&tx[0]=='L'){
              tx=tx.substr(1);
              int len=int(strspn(tx.c_str(),"0123456789"));
              //-Obtains type of argument.
              string typearg;
              if(len){//-Type with name.
                int n=atoi((tx.substr(0,len)).c_str());
                typearg=tx.substr(len,n);  tx=tx.substr(len+n);
              }
              else{//-Basic type (b:bool, j:unsigned, i:int,...)
                typearg=tx[0];  tx=tx.substr(1);
              }
              //-Obtains value of argument.
              pos=int(tx.find("E")); 
              if(pos<0)RunException(met,"Error in interpreting the template arguments.",file);
              ker.AddTemplateArg(typearg,tx.substr(0,pos));
              tx=tx.substr(pos+1);
            }
            tx=tx.substr(1);
          }
          ker.SetArgs(tx);
          AddKernel(ker);
        }
        else if(kersm){
          //-Obtains registers of kernel.
          pos=int(line.find("Used ")); 
          int pos2=int(line.find(" registers")); 
          if(pos>=0 && pos2>=0){//-Registers of the kernel.
            pos+=int(string("Used ").length());
            string tx=line.substr(pos,pos2-pos);
            ker.SetRegs(kersm,atoi(tx.c_str()));
            ker.SetStackFrame(kersm,stackframe);
            AddKernel(ker);
          }
          ker.Reset(); kersm=0; stackframe=0;
        }
      }
      else if(pos<0 && int(line.find(" bytes stack frame"))>=0){
        //-Obtains stack frame of kernel.
        int pos2=int(line.find(" bytes stack frame")); 
        string tx=fun::StrTrim(line.substr(0,pos2));
        stackframe=atoi(tx.c_str());
      }
    } 
    if(!pf.eof()&&pf.fail())RunException(met,"Failed reading to file.",file);
    pf.close();
  }
  else RunException(met,"File could not be opened.",file);
  Sort();
}

//==============================================================================
/// Function to order kernels.
//==============================================================================
bool JPtxasInfoKerSort(JPtxasInfoKer* i,JPtxasInfoKer* j){ 
  return (i->GetCode()<j->GetCode());
}

//==============================================================================
/// Sorts kernels by code.
//==============================================================================
void JPtxasInfo::Sort(){
  sort(Kernels.begin(),Kernels.end(),JPtxasInfoKerSort);
}

//==============================================================================
/// Shows data for debug.
//==============================================================================
void JPtxasInfo::Print()const{
  for(unsigned c=0;c<Count();c++){
    printf("\nJPtxasInfoKer[%u]\n",c);
    GetKernel(c)->Print();
  }
}

//==============================================================================
/// Stores data in csv file.
//==============================================================================
void JPtxasInfo::SaveCsv(const std::string &file)const{
  const char met[]="SaveCsv";
  ofstream pf;
  pf.open(file.c_str());
  if(pf){
    bool sm10=false,sm13=false,sm20=false,sm30=false,sm35=false;
    //-Checks SM with data.
    for(unsigned c=0;c<Count();c++){
      JPtxasInfoKer* ker=Kernels[c];
      if(ker->GetRegs(10))sm10=true;
      if(ker->GetRegs(13))sm13=true;
      if(ker->GetRegs(20))sm20=true;
      if(ker->GetRegs(30))sm30=true;
      if(ker->GetRegs(35))sm35=true;
    }   
    //-Creates CSV file.
    pf << "Sp;Kernel;TemplateVars";
    if(sm10)pf << ";Regs_sm10";
    if(sm13)pf << ";Regs_sm13";
    if(sm20)pf << ";Regs_sm20";
    if(sm30)pf << ";Regs_sm30";
    if(sm35)pf << ";Regs_sm35";
    if(sm10)pf << ";Stack_sm10";
    if(sm13)pf << ";Stack_sm13";
    if(sm20)pf << ";Stack_sm20";
    if(sm30)pf << ";Stack_sm30";
    if(sm35)pf << ";Stack_sm35";
    pf << ";Args" << endl;
    for(unsigned c=0;c<Count();c++){
      JPtxasInfoKer* ker=Kernels[c];
      string tmpvar;
      for(unsigned t=0;t<ker->CountTemplateArgs();t++){
        if(t)tmpvar=tmpvar+", ";
        tmpvar=tmpvar+ker->GetTemplateArgsType(t)+"="+ker->GetTemplateArgsValue(t);
      }
      pf << ker->GetNameSpace() << ";" << ker->GetName() << ";" << tmpvar;
      if(sm10)pf << ";" << ker->GetRegs(10);
      if(sm13)pf << ";" << ker->GetRegs(13);
      if(sm20)pf << ";" << ker->GetRegs(20);
      if(sm30)pf << ";" << ker->GetRegs(30);
      if(sm35)pf << ";" << ker->GetRegs(35);
      if(sm10)pf << ";" << ker->GetStackFrame(10);
      if(sm13)pf << ";" << ker->GetStackFrame(13);
      if(sm20)pf << ";" << ker->GetStackFrame(20);
      if(sm30)pf << ";" << ker->GetStackFrame(30);
      if(sm35)pf << ";" << ker->GetStackFrame(35);
      pf << ";" << ker->GetArgs() << endl;
    }   
    if(pf.fail())RunException(met,"Failed writing to file.",file);
    pf.close();
  }
  else RunException(met,"File could not be opened.",file);
}

//==============================================================================
/// Check information of the requested sm.
//==============================================================================
bool JPtxasInfo::CheckSm(unsigned sm)const{
  bool ok=false;
  for(unsigned c=0;c<Count()&&!ok;c++)ok=(Kernels[c]->GetRegs(sm)!=0);
  return(ok);
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername)const{
  int idx=-1,n=0;
  for(unsigned c=0;c<Count();c++)if(Kernels[c]->GetCodeMin()==kername){ 
    idx=c; n++;
  }
  return(n!=1? -1: idx);
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1)const{
  std::string kname=kername+fun::PrintStr("_%u",v1);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2)const{
  std::string kname=kername+fun::PrintStr("_%u_%u",v1,v2);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u",v1,v2,v3);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u",v1,v2,v3,v4);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5,v6);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of requested kernel (-1 it was not found).
//==============================================================================
int JPtxasInfo::GetKernelIdx(const std::string &kername,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6,unsigned v7)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5,v6,v7);
  return(GetKernelIdx(kname));
}

//==============================================================================
/// Returns the number of registers of the requested kernel.
//==============================================================================
unsigned JPtxasInfo::GetRegs(int keridx,unsigned sm)const{
  unsigned ret=0;
  if(keridx>=int(Count()))RunException("GetRegs","Number of kernel is invalid.");
  if(keridx>=0)ret=Kernels[keridx]->GetRegs(sm);
  return(ret);
}

//==============================================================================
/// Returns the number of stack frame of the requested kernel.
//==============================================================================
unsigned JPtxasInfo::GetStackFrame(int keridx,unsigned sm)const{
  unsigned ret=0;
  if(keridx>=int(Count()))RunException("GetStackFrame","Number of kernel is invalid.");
  if(keridx>=0)ret=Kernels[keridx]->GetStackFrame(sm);
  return(ret);
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm)const{
  tuint2 ret=TUint2(0);
  unsigned n=0;
  for(unsigned c=0;c<Count();c++){
    JPtxasInfoKer* ker=Kernels[c];
    if(ker->GetCodeMin()==kername){ 
      ret.x=ker->GetRegs(sm);
      ret.y=ker->GetStackFrame(sm);
      n++; 
    }
  }
  return(n!=1? TUint2(0): ret);
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1)const{
  std::string kname=kername+fun::PrintStr("_%u",v1);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2)const{
  std::string kname=kername+fun::PrintStr("_%u_%u",v1,v2);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u",v1,v2,v3);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u",v1,v2,v3,v4);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5,v6);
  return(GetData(kname,sm));
}

//==============================================================================
/// Returns the data of the requested kernel.
//==============================================================================
tuint2 JPtxasInfo::GetData(const std::string &kername,unsigned sm,unsigned v1,unsigned v2,unsigned v3,unsigned v4,unsigned v5,unsigned v6,unsigned v7)const{
  std::string kname=kername+fun::PrintStr("_%u_%u_%u_%u_%u_%u_%u",v1,v2,v3,v4,v5,v6,v7);
  return(GetData(kname,sm));
}








