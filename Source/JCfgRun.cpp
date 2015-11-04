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

/// \file JCfgRun.cpp \brief Implements the class \ref JCfgRun.

#include "JCfgRun.h"
#include "JSpaceEParms.h"

using std::string;
using std::ifstream;
using fun::PrintVar;
using fun::VarStr;
using fun::StrTrim;
using fun::StrUpper;

//==============================================================================
/// Constructor.
//==============================================================================
JCfgRun::JCfgRun(){
  ClassName="JCfgRun";
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JCfgRun::Reset(){
  PrintInfo=false; SvDef=false; DirsDef=0;
  Cpu=false;
  Gpu=false; GpuId=-1; GpuFree=false;
  Stable=false;
  Psimple=true;
  SvDouble=false;
  OmpThreads=0;
  SvTimers=true; SvTimersStep=0;
  CellOrder=ORDER_None;
  CellMode=CELLMODE_2H;
  DomainMode=0;
  DomainParticlesMin=DomainParticlesMax=TDouble3(0);
  DomainParticlesPrcMin=DomainParticlesPrcMax=TDouble3(0);
  DomainFixedMin=DomainFixedMax=TDouble3(0);
  TStep=STEP_None; VerletSteps=-1;
  TVisco=VISCO_None; Visco=0; ViscoBoundFactor=-1;
  DeltaSph=-1;
  RenCorrection=-1;
  Shifting=-1;
  SvRes=true; SvDomainVtk=false;
  Sv_Binx=false; Sv_Info=false; Sv_Vtk=false; Sv_Csv=false;
  CaseName=""; DirOut=""; RunName=""; 
  PartBegin=0; PartBeginFirst=0; PartBeginDir="";
  TimeMax=-1; TimePart=-1;
  RhopOutModif=false; RhopOutMin=700; RhopOutMax=1300;
  PtxasFile="";
  FtPause=-1;
}

//==============================================================================
/// Shows information about execution parameters.
//==============================================================================
void JCfgRun::VisuInfo()const{
  printf("Information about execution parameters:\n\n");
  printf("  DualSPHysics [name_case [dir_out]] [options]\n\n");
  printf("  Options:\n");
  printf("    -h          Shows information about parameters\n\n");
  printf("    -opt <file> Load a file configuration\n\n");
  printf("    -cpu        Execution on Cpu (option by default)\n");
  printf("    -gpu[:id]   Execucion on Gpu and id of the device\n");
  printf("    -stable     The result is always the same but the execution is slower\n");
  printf("    -psimple    Use position with low precision (option by default)\n");
  printf("    -pdouble    Use position with high precision (enables SvDouble)\n");
  printf("\n");

#ifdef _WITHOMP
  printf("    -ompthreads:<int>  Only for Cpu execution, indicates the number of threads\n");
  printf("                   by host for parallel execution, it takes the number of \n");
  printf("                   cores of the device by default (or using zero value)\n\n");
#endif
  //printf("    -cellorder:<axis> Indicates the order of the axes. (xyz/xzy/yxz/yzx/zxy/zyx)\n");
  printf("    -cellmode:<mode>  Specify the cell division mode\n");
  printf("        2h        Lowest and the least expensive in memory (by default)\n");
  printf("        h         Fastest and the most expensive in memory\n\n");
  printf("    -symplectic      Symplectic algorithm as time step algorithm\n");
  printf("    -verlet[:steps]  Verlet algorithm as time step algorithm and number of\n");
  printf("                     time steps to switch equations\n\n");
  printf("    -viscoart:<float>          Artifitical viscosity [0-1]\n");
  printf("    -viscolamsps:<float>       Laminar+SPS viscosity [order of 1E-6]\n");  
  printf("    -viscoboundfactor:<float>  Multiplies the viscosity value of boundary\n");
  printf("\n");
  printf("    -deltasph:<float> Constant for DeltaSPH. Typical value is 0.1 (0 by default)\n\n");
  printf("    -rencorrection:<float> Constant for Ren correction (0-1), with 0 disabled (def=0)\n\n");
  printf("    -shifting:<mode> Specify the use of Shifting correction\n");
  printf("        none       Shifting is disabled (by default)\n");
  printf("        nobound    Shifting is not applied near boundary\n");
  printf("        nofixed    Shifting is not applied near fixed boundary\n");
  printf("        full       Shifting is always applied\n\n");
  printf("    -sv:[formas,...] Specify the output formats.\n");
  printf("        none    No particles files are generated\n");
  printf("        binx    Bynary files (option by default)\n");
  printf("        info    Information about execution in ibi4 format\n");
  printf("        vtk     VTK files\n");
  printf("        csv     CSV files\n");
  printf("    -svdouble        Save position with high precision (disabled by default)\n");
  printf("    -svres:<0/1>     Generate file that summarizes the execution process\n");
  printf("    -svtimers:<0/1>  Obtain timing for each individual process\n");
  printf("    -svtimersstep:<float> Obtain timing for each individual process. Specify\n");
  printf("     the time between output information. By default 0 (disabled)\n");
  printf("    -svdomainvtk:<0/1>  Generate VTK file with domain limits\n");
  printf("    -name <string>      Specify path and name of the case \n");
  printf("    -runname <string>   Specify name for case execution\n");
  printf("    -dirout <dir>       Specify the out directory \n\n");
  printf("    -partbegin:begin[:first] dir \n");
  printf("     Specify the beginning of the simulation starting from a given PART\n");
  printf("     (begin) and located in the directory (dir), (first) indicates the\n");
  printf("     number of the first PART to be generated\n\n");
  printf("    -incz:<float>    Allowable increase in Z+ direction \n");
  printf("    -rhopout:min:max Exclude fluid particles out of these density limits\n\n");
  printf("    -ftpause:<float> Time to start floating bodies movement. By default 0\n");
  printf("    -tmax:<float>   Maximum time of simulation\n");
  printf("    -tout:<float>   Time between output files\n\n");
  printf("    -domain_particles[:xmin,ymin,zmin,xmax,ymax,zmax]  El dominio del caso se\n");
  printf("     fija en funcion de las particulas iniciales y es modificado por xmin,...\n");
  printf("    -domain_particles_prc:xmin,ymin,zmin,xmax,ymax,zmax  Los valores en \n");
  printf("     proporcion con las dimensiones del caso segun las particulas iniciales\n");
  printf("    -domain_fixed:xmin,ymin,zmin,xmax,ymax,zmax    El dominio del caso se\n");
  printf("     fija con los valores especificados\n\n");
  printf("    -move:x,y,z       Desplaza las particulas durante el calculo.\n");
  printf("    -ptxasfile <file> Indicate the file with information about the compilation\n");
  printf("     of kernels in CUDA to adjust the size of the blocks depending on the \n");
  printf("     needed registers for each kernel (only for gpu)\n\n");
  printf("  Examples:\n");
  printf("    DualSPHysics case out_case -sv:binx2,csv \n");
}

//==============================================================================
/// Shows current configuration.
//==============================================================================
void JCfgRun::VisuConfig()const{
  printf("\nConfiguration of execution:\n");
  string ln="\n";
  PrintVar("  CaseName",CaseName,ln);
  PrintVar("  RunName",RunName,ln);
  PrintVar("  DirOut",DirOut,ln);
  PrintVar("  PartBegin",PartBegin,ln);
  PrintVar("  PartBeginFirst",PartBeginFirst,ln);
  PrintVar("  PartBeginDir",PartBeginDir,ln);
  PrintVar("  Cpu",Cpu,ln);
  printf("  %s  %s\n",VarStr("Gpu",Gpu).c_str(),VarStr("GpuId",GpuId).c_str());
  PrintVar("  GpuFree",GpuFree,ln);
  PrintVar("  Stable",Stable,ln);
  PrintVar("  Psimple",Psimple,ln);
  PrintVar("  OmpThreads",OmpThreads,ln);
  PrintVar("  CellOrder",GetNameCellOrder(CellOrder),ln);
  PrintVar("  CellMode",GetNameCellMode(CellMode),ln);
  PrintVar("  TStep",TStep,ln);
  PrintVar("  VerletSteps",VerletSteps,ln);
  PrintVar("  TVisco",TVisco,ln);
  PrintVar("  Visco",Visco,ln);
  PrintVar("  ViscoBoundFactor",ViscoBoundFactor,ln);
  PrintVar("  DeltaSph",DeltaSph,ln);
  PrintVar("  RenCorrection",RenCorrection,ln);
  PrintVar("  Shifting",Shifting,ln);
  PrintVar("  SvRes",SvRes,ln);
  PrintVar("  SvTimers",SvTimers,ln);
  PrintVar("  SvTimersStep",SvTimers,ln);
  PrintVar("  SvDomainVtk",SvDomainVtk,ln);
  PrintVar("  Sv_Binx",Sv_Binx,ln);
  PrintVar("  Sv_Info",Sv_Info,ln);
  PrintVar("  Sv_Vtk",Sv_Vtk,ln);
  PrintVar("  Sv_Csv",Sv_Csv,ln);
  PrintVar("  SvDouble",SvDouble,ln);
  PrintVar("  RhopOutModif",RhopOutModif,ln);
  if(RhopOutModif){
    PrintVar("  RhopOutMin",RhopOutMin,ln);
    PrintVar("  RhopOutMax",RhopOutMax,ln);
  }
  PrintVar("  TimeMax",TimeMax,ln);
  PrintVar("  TimePart",TimePart,ln);
  if(DomainMode==1){
    PrintVar("  DomainParticlesMin",DomainParticlesMin,ln);
    PrintVar("  DomainParticlesMax",DomainParticlesMax,ln);
    PrintVar("  DomainParticlesPrcMin",DomainParticlesPrcMin,ln);
    PrintVar("  DomainParticlesPrcMax",DomainParticlesPrcMax,ln);
  }
  else if(DomainMode==2){
    PrintVar("  DomainFixedMin",DomainFixedMin,ln);
    PrintVar("  DomainFixedMax",DomainFixedMax,ln);
  }
  PrintVar("  PtxasFile",PtxasFile,ln);
  PrintVar("  FtPause",FtPause,ln);
}

//==============================================================================
/// Loads execution parameters from the command line.
//==============================================================================
void JCfgRun::LoadArgv(int argc,char** argv){
  const char met[]="LoadArgv";
  Reset();
  const int MAXOPTS=100;
  string *optlis=new string[MAXOPTS];
  int optn=0;
  for(int c=0;c<argc-1;c++){
    string tex=StrTrim(argv[c+1]);
    int pos=int(tex.find(" "));
    if(pos>0){
      while(pos>0){
        if(optn>=MAXOPTS)RunException(met,"It has exceeded the maximum configuration options.");
        optlis[optn]=tex.substr(0,pos); optn++;
        tex=tex.substr(pos+1);
        pos=int(tex.find(" "));
      }
    }
    if(optn>=MAXOPTS)RunException(met,"It has exceeded the maximum configuration options.");
    optlis[optn]=tex; optn++;
  }
  if(optn)LoadOpts(optlis,optn,0,"");
  delete[] optlis;
  if(!optn)PrintInfo=true;
  if(!PrintInfo){ //-Configuracion por defecto //-default configuration
    if(!Cpu&&!Gpu)Cpu=true;
    if(!SvDef){ Sv_Binx=true; Sv_Info=true; }
  }
  else VisuInfo();
  if(PtxasFile.empty())PtxasFile=fun::GetWithoutExtension(argv[0])+"_ptxasinfo";
}

//==============================================================================
/// Loads execution parameters from a text file.
//==============================================================================
void JCfgRun::LoadFile(string fname,int lv){
  const char met[]="LoadFile";
  const int MAXOPTS=50;
  int optn=0;
  string *optlis=new string[MAXOPTS];
  ifstream pf;
  pf.open(fname.c_str());
  if(pf){
    while(!pf.eof()&&optn<MAXOPTS){
      string tex;  pf >> tex;
      if(tex!=""){
        if(optn<MAXOPTS)optlis[optn]=tex;
        optn++;
      }
    } 
    if(!pf.eof()&&pf.fail())RunException(met,"Error reading data from the file.",fname);
    pf.close();
  }
  else RunException(met,"The file can not be opened.",fname);
  if(optn>=MAXOPTS)RunException(met,fun::PrintStr("File with too many lines (Maximum=%d)",MAXOPTS),fname);
  if(optn>0)LoadOpts(optlis,optn,lv,fname);
  delete[] optlis;
}

//==============================================================================
/// Generates error of unknown parameter.
//==============================================================================
void JCfgRun::ErrorParm(const std::string &opt,int optc,int lv,const std::string &file)const{
  const char met[]="ErrorParm";
  std::string tx=fun::PrintStr("Parameter \"%s\" unrecognised or invalid. ",opt.c_str());
  tx=tx+fun::PrintStr("(Level cfg:%d, Parameter:%d)",lv,optc);
  if(file!="")RunException(met,tx,file); else RunException(met,tx);
}

//==============================================================================
/// Loads execution parameters.
//==============================================================================
void JCfgRun::LoadOpts(string *optlis,int optn,int lv,string file){
  const char met[]="LoadOpts";
  if(lv>=10)RunException(met,"No more than 10 levels of recursive configuration.");
  for(int c=0;c<optn;c++){
    string opt=optlis[c];
    if(opt[0]!='-'&&opt[0]!='#'){
      if(!DirsDef){ CaseName=opt; DirsDef++; }
      else if(DirsDef==1){ DirOut=opt; DirsDef++; }
      else ErrorParm(opt,c,lv,file);
    }
    else if(opt[0]=='-'){
      string tx=opt.substr(1);
      int pos=int(tx.find("#"));
      if(pos>0)tx=tx.substr(0,pos);
      pos=int(tx.find(":"));
      string txopt="",txopt2="",txopt3="",txopt4="",txword=StrUpper(pos>0? tx.substr(0,pos): tx);
      if(pos>=0)txopt=tx.substr(pos+1);
      tx=txopt;
      pos=int(tx.find(":"));
      txopt=(pos>=0? tx.substr(0,pos): tx);
      if(pos>=0)txopt2=tx.substr(pos+1);
      tx=txopt2;
      pos=int(tx.find(":"));
      txopt2=(pos>=0? tx.substr(0,pos): tx);
      if(pos>=0)txopt3=tx.substr(pos+1);
      tx=txopt3;
      pos=int(tx.find(":"));
      txopt3=(pos>=0? tx.substr(0,pos): tx);
      if(pos>=0)txopt4=tx.substr(pos+1);
      if(txword=="CPU"){ Cpu=true; Gpu=false; }
      else if(txword=="GPU"){ Gpu=true; Cpu=false;
        if(txopt!="")GpuId=atoi(txopt.c_str()); 
      }
      else if(txword=="STABLE")Stable=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
      else if(txword=="PSIMPLE"){
        Psimple=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
        if(!Psimple)SvDouble=true;
      }
      else if(txword=="PDOUBLE"){
        Psimple=!((txopt!=""? atoi(txopt.c_str()): 1)!=0);
        if(!Psimple)SvDouble=true;
      }
      else if(txword=="SVDOUBLE")SvDouble=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
#ifdef _WITHOMP
      else if(txword=="OMPTHREADS"){ 
        OmpThreads=atoi(txopt.c_str()); if(OmpThreads<0)OmpThreads=0;
      } 
#endif
      //else if(txword=="CELLORDER"){
      //  bool ok=true;
      //  if(!txopt.empty()){
      //    txopt=StrUpper(txopt);
      //    if(txopt=="XYZ")CellOrder=ORDER_XYZ;
      //    else if(txopt=="XZY")CellOrder=ORDER_XZY;
      //    else if(txopt=="YXZ")CellOrder=ORDER_YXZ;
      //    else if(txopt=="YZX")CellOrder=ORDER_YZX;
      //    else if(txopt=="ZXY")CellOrder=ORDER_ZXY;
      //    else if(txopt=="ZYX")CellOrder=ORDER_ZYX;
      //    else ok=false;
      //  }
      //  else ok=false;
      //  if(!ok)ErrorParm(opt,c,lv,file);
      //}
      else if(txword=="CELLMODE"){
        bool ok=true;
        if(!txopt.empty()){
          txopt=StrUpper(txopt);
          if(txopt=="H")CellMode=CELLMODE_H;
          else if(txopt=="2H")CellMode=CELLMODE_2H;
          else ok=false;
        }
        else ok=false;
        if(!ok)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="SYMPLECTIC")TStep=STEP_Symplectic;
      else if(txword=="VERLET"){ TStep=STEP_Verlet; 
        if(txopt!="")VerletSteps=atoi(txopt.c_str()); 
      }
      else if(txword=="VISCOART"){ 
        Visco=float(atof(txopt.c_str())); 
        if(Visco>10)ErrorParm(opt,c,lv,file);
        TVisco=VISCO_Artificial;
      }
      else if(txword=="VISCOLAMSPS"){ 
        Visco=float(atof(txopt.c_str())); 
        if(Visco>0.001)ErrorParm(opt,c,lv,file);
        TVisco=VISCO_LaminarSPS;
      }
      else if(txword=="VISCOBOUNDFACTOR"){ 
        ViscoBoundFactor=float(atof(txopt.c_str())); 
        if(ViscoBoundFactor<0)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="DELTASPH"){
        DeltaSph=float(atof(txopt.c_str())); 
        if(DeltaSph<0||DeltaSph>1)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="RENCORRECTION"){
        RenCorrection=float(atof(txopt.c_str())); 
        if(RenCorrection<0||RenCorrection>1)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="SHIFTING"){
        const string tx=fun::StrUpper(txopt);
        if(tx=="NONE")Shifting=0;
        else if(tx=="NOBOUND")Shifting=1;
        else if(tx=="NOFIXED")Shifting=2;
        else if(tx=="FULL")Shifting=3;
        else ErrorParm(opt,c,lv,file);
      }
      else if(txword=="SVRES")SvRes=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
      else if(txword=="SVTIMERS")SvTimers=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
      else if(txword=="SVTIMERSSTEP"){ 
        SvTimersStep=float(atof(txopt.c_str())); 
        if(SvTimersStep<0)SvTimersStep=0;
        if(SvTimersStep>0)SvTimers=true;
      }
      else if(txword=="SVDOMAINVTK")SvDomainVtk=(txopt!=""? atoi(txopt.c_str()): 1)!=0;
      else if(txword=="SV"){
        string txop=StrUpper(txopt);
        while(txop.length()>0){
          pos=int(txop.find(","));
          string op=(pos>=0? txop.substr(0,pos): txop);
          txop=(pos>=0? txop.substr(pos+1): "");
          if(op=="NONE"){ 
            SvDef=true; Sv_Binx=false; Sv_Info=false; 
            Sv_Csv=false; Sv_Vtk=false;
          }
          else if(op=="BINX"){    SvDef=true; Sv_Binx=true; }
          else if(op=="INFO"){    SvDef=true; Sv_Info=true; }
          else if(op=="VTK"){     SvDef=true; Sv_Vtk=true; }
          else if(op=="CSV"){     SvDef=true; Sv_Csv=true; }
          else ErrorParm(opt,c,lv,file);
        }
      }
      else if(txword=="NAME"&&c+1<optn){ CaseName=optlis[c+1]; c++; }
      else if(txword=="RUNNAME"&&c+1<optn){ RunName=optlis[c+1]; c++; }
      else if(txword=="DIROUT"&&c+1<optn){ DirOut=optlis[c+1]; c++; }
      else if(txword=="PARTBEGIN"&&c+1<optn){ 
        int v1=atoi(txopt.c_str());
        int v2=atoi(txopt2.c_str());
        if(v1<0||v2<0)ErrorParm(opt,c,lv,file);
        else{
          PartBegin=unsigned(v1);
          PartBeginFirst=(txopt2.empty()? PartBegin: unsigned(v2));
        }
        PartBeginDir=optlis[c+1]; c++; 
      }
      else if(txword=="RHOPOUT"){ 
        RhopOutMin=float(atof(txopt.c_str())); 
        RhopOutMax=float(atof(txopt2.c_str())); 
        RhopOutModif=true;
      }
      else if(txword=="FTPAUSE"){ 
        FtPause=float(atof(txopt.c_str())); 
        if(FtPause<0)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="TMAX"){ 
        TimeMax=float(atof(txopt.c_str())); 
        if(TimeMax<0)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="TOUT"){ 
        TimePart=float(atof(txopt.c_str())); 
        if(TimePart<0)ErrorParm(opt,c,lv,file);
      }
      else if(txword=="DOMAIN_PARTICLES"){
        LoadDouble6(txopt+":"+txopt2+":"+txopt3,0,DomainParticlesMin,DomainParticlesMax);
        DomainMode=1;
      }
      else if(txword=="DOMAIN_PARTICLES_PRC"){
        LoadDouble6(txopt+":"+txopt2+":"+txopt3,0,DomainParticlesPrcMin,DomainParticlesPrcMax);
        DomainMode=1;
      }
      else if(txword=="DOMAIN_FIXED"){
        LoadDouble6(txopt+":"+txopt2+":"+txopt3,0,DomainFixedMin,DomainFixedMax);
        DomainMode=2;
      }
      else if(txword=="INCZ"){ 
        double incz=atof(txopt.c_str()); 
        if(incz<0)ErrorParm(opt,c,lv,file);
        DomainMode=1;
        DomainParticlesMin=DomainParticlesMax=TDouble3(0);
        DomainParticlesPrcMin=DomainParticlesPrcMax=TDouble3(0);
        DomainParticlesPrcMax.z=incz;
      }
      else if(txword=="PTXASFILE"&&c+1<optn){ PtxasFile=optlis[c+1]; c++; }
      else if(txword=="OPT"&&c+1<optn){ LoadFile(optlis[c+1],lv+1); c++; }
      else if(txword=="H"||txword=="HELP"||txword=="?")PrintInfo=true;
      else ErrorParm(opt,c,lv,file);
    }
  }
}

//==============================================================================
/// Load 1 value tdouble3 using command options.
//==============================================================================
void JCfgRun::LoadDouble3(std::string txopt,double def,tdouble3 &v1){
  double values[3]={def,def,def};
  string ttx=txopt;
  for(int tc=0;ttx!=""&&tc<3;tc++){
    int tpos=int(ttx.find(":"));
    string ttxopt=(tpos>0? ttx.substr(0,tpos): ttx);
    string ttxopt2;
    if(tpos>0)ttxopt2=ttx.substr(tpos+1);
    values[tc]=atof(ttxopt.c_str());
    ttx=ttxopt2;
  } 
  v1=TDouble3(values[0],values[1],values[2]);
}

//==============================================================================
/// Load 1 value tfloat3 using command options.
//==============================================================================
void JCfgRun::LoadFloat3(std::string txopt,float def,tfloat3 &v1){
  tdouble3 v1d;
  LoadDouble3(txopt,def,v1d);
  v1=ToTFloat3(v1d);
}

//==============================================================================
/// Load 2 values tdouble3 using command options.
//==============================================================================
void JCfgRun::LoadDouble6(std::string txopt,double def,tdouble3 &v1,tdouble3 &v2){
  double values[6]={def,def,def,def,def,def};
  string ttx=txopt;
  for(int tc=0;ttx!=""&&tc<6;tc++){
    int tpos=int(ttx.find(":"));
    string ttxopt=(tpos>0? ttx.substr(0,tpos): ttx);
    string ttxopt2;
    if(tpos>0)ttxopt2=ttx.substr(tpos+1);
    values[tc]=atof(ttxopt.c_str());
    ttx=ttxopt2;
  } 
  v1=TDouble3(values[0],values[1],values[2]);
  v2=TDouble3(values[3],values[4],values[5]);
}

//==============================================================================
/// Load 2 values tfloat3 using command options.
//==============================================================================
void JCfgRun::LoadFloat6(std::string txopt,float def,tfloat3 &v1,tfloat3 &v2){
  tdouble3 v1d,v2d;
  LoadDouble6(txopt,def,v1d,v2d);
  v1=ToTFloat3(v1d);
  v2=ToTFloat3(v2d);
}



