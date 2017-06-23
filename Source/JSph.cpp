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

/// \file JSph.cpp \brief Implements the class \ref JSph

#include "JSph.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JXml.h"
#include "JSpaceCtes.h"
#include "JSpaceEParms.h"
#include "JSpaceParts.h"
#include "JFormatFiles2.h"
#include "JCellDivCpu.h"
#include "JFormatFiles2.h"
#include "JSphDtFixed.h"
#include "JSaveDt.h"
#include "JTimeOut.h"
#include "JSphVisco.h"
#include "JWaveGen.h"
#include "JSphAccInput.h"
#include "JPartDataBi4.h"
#include "JPartOutBi4Save.h"
#include "JPartFloatBi4.h"
#include "JPartsOut.h"
#include <climits>

//using namespace std;
using std::string;
using std::ofstream;
using std::endl;

//==============================================================================
/// Constructor.
//==============================================================================
JSph::JSph(bool cpu,bool withmpi):Cpu(cpu),WithMpi(withmpi){
  ClassName="JSph";
  DataBi4=NULL;
  DataOutBi4=NULL;
  DataFloatBi4=NULL;
  PartsOut=NULL;
  Log=NULL;
  ViscoTime=NULL;
  DtFixed=NULL;
  SaveDt=NULL;
  TimeOut=NULL;
  MkList=NULL;
  Motion=NULL;
  FtObjs=NULL;
  WaveGen=NULL;
  AccInput=NULL;
  InitVars();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSph::~JSph(){
  delete DataBi4;
  delete DataOutBi4;
  delete DataFloatBi4;
  delete PartsOut;
  delete ViscoTime;
  delete DtFixed;
  delete SaveDt;
  delete TimeOut;
  ResetMkInfo();
  delete Motion;
  AllocMemoryFloating(0);
  delete WaveGen;
  delete AccInput;
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSph::InitVars(){
  ClearCfgDomain();
  OutPosCount=OutRhopCount=OutMoveCount=0;
  Simulate2D=false;
  Stable=false;
  SvDouble=false;
  RunCode=CalcRunCode();
  RunTimeDate="";
  CaseName=""; DirCase=""; DirOut=""; RunName="";
  TStep=STEP_None;
  TKernel=KERNEL_Wendland;
  Awen=Bwen=0;
  TVisco=VISCO_None;
  TShifting=SHIFT_None; ShiftCoef=0;
  FreeSurface=0;
  TensileN=0;
  TensileR=0;
  TPrecond=PRECOND_Jacobi;
  TAMGInter=AMGINTER_AG;
  Iterations=0;
  Tolerance=0;
  StrongConnection=0; JacobiWeight=0; Presmooth=0; Postsmooth=0; CoarseCutoff=0;
	PistonPosX=0; PistonPosZ=0;
	NegativePressureBound=true;
  Visco=0; ViscoBoundFactor=1;
  UseDEM=false;  //(DEM)
  DemDtForce=0;  //(DEM)
  memset(DemObjs,0,sizeof(StDemData)*DemObjsSize);  //(DEM)
  RhopOut=true; RhopOutMin=700; RhopOutMax=1300;
  TimeMax=TimePart=0;
  DtIni=DtMin=0; CoefDtMin=0; DtAllParticles=false;
  PartsOutMax=0;
  NpMinimum=0;

  SvData=byte(SDAT_Binx)|byte(SDAT_Info);
  SvRes=false;
  SvTimers=false;
  SvDomainVtk=false;

  H=CteB=Gamma=RhopZero=CFLnumber=0;
  Dp=0;
  Cs0=0;
  Delta2H=0;
  MassFluid=MassBound=0;
  Gravity=TFloat3(0);
  Dosh=H2=Fourh2=Eta2=0;

  CasePosMin=CasePosMax=TDouble3(0);
  CaseNp=CaseNbound=CaseNfixed=CaseNmoving=CaseNfloat=CaseNfluid=CaseNpb=0;

  ResetMkInfo();

  memset(&PeriodicConfig,0,sizeof(StPeriodic));
  PeriActive=0;
  PeriX=PeriY=PeriZ=PeriXY=PeriXZ=PeriYZ=false;
  PeriXinc=PeriYinc=PeriZinc=TDouble3(0);

  PartBeginDir=""; 
  PartBegin=PartBeginFirst=0;
  PartBeginTimeStep=0; 
  PartBeginTotalNp=0;

  MotionTimeMod=0;
  MotionObjCount=0;
  memset(MotionObjBegin,0,sizeof(unsigned)*256);

  FtCount=0;
  FtPause=0;

  AllocMemoryFloating(0);

  CellOrder=ORDER_None;
  CellMode=CELLMODE_None;
  Hdiv=0;
  Scell=0;
  MovLimit=0;

  Map_PosMin=Map_PosMax=Map_Size=TDouble3(0);
  Map_Cells=TUint3(0);
  MapRealPosMin=MapRealPosMax=MapRealSize=TDouble3(0);

  DomCelIni=DomCelFin=TUint3(0);
  DomCells=TUint3(0);
  DomPosMin=DomPosMax=DomSize=TDouble3(0);
  DomRealPosMin=DomRealPosMax=TDouble3(0);
  DomCellCode=0;

  NpDynamic=ReuseIds=false;
  TotalNp=0; IdMax=0;

  DtModif=0;
  PartDtMin=DBL_MAX; PartDtMax=-DBL_MAX;

  MaxMemoryCpu=MaxMemoryGpu=MaxParticles=MaxCells=0;

  PartIni=Part=0; 
  Nstep=0; PartNstep=-1;
  PartOut=0;

  TimeStepIni=0;
  TimeStep=TimeStepM1=0;
  TimePartNext=0;
}

//==============================================================================
/// Generates a random code to identify the file of the results of the execution.
//==============================================================================
std::string JSph::CalcRunCode()const{
  srand((unsigned)time(NULL));
  const unsigned len=8;
  char code[len+1];
  for(unsigned c=0;c<len;c++){
    char let=char(float(rand())/float(RAND_MAX)*36);
    code[c]=(let<10? let+48: let+87);
  } 
  code[len]=0;
  return(code);
}

//============================================================================== 
/// Returns the code version in text format.
//==============================================================================
std::string JSph::GetVersionStr(){
  return(fun::PrintStr("%1.2f",float(VersionMajor)/100));
}

//==============================================================================
/// Sets the configuration of the domain limits by default.
//==============================================================================
void JSph::ClearCfgDomain(){
  CfgDomainParticles=true;
  CfgDomainParticlesMin=CfgDomainParticlesMax=TDouble3(0);
  CfgDomainParticlesPrcMin=CfgDomainParticlesPrcMax=TDouble3(0);
  CfgDomainFixedMin=CfgDomainFixedMax=TDouble3(0);
}

//==============================================================================
/// Sets the configuration of the domain limits using given values.
//==============================================================================
void JSph::ConfigDomainFixed(tdouble3 vmin,tdouble3 vmax){
  ClearCfgDomain();
  CfgDomainParticles=false;
  CfgDomainFixedMin=vmin; CfgDomainFixedMax=vmax;
}

//==============================================================================
/// Sets the configuration of the domain limits using positions of particles.
//==============================================================================
void JSph::ConfigDomainParticles(tdouble3 vmin,tdouble3 vmax){
  CfgDomainParticles=true;
  CfgDomainParticlesMin=vmin; CfgDomainParticlesMax=vmax;
}

//==============================================================================
/// Sets the configuration of the domain limits using positions plus a percentage.
//==============================================================================
void JSph::ConfigDomainParticlesPrc(tdouble3 vmin,tdouble3 vmax){
  CfgDomainParticles=true;
  CfgDomainParticlesPrcMin=vmin; CfgDomainParticlesPrcMax=vmax;
}

//==============================================================================
/// Allocates memory of floating objectcs.
//==============================================================================
void JSph::AllocMemoryFloating(unsigned ftcount){
  delete[] FtObjs; FtObjs=NULL;
  if(ftcount)FtObjs=new StFloatingData[ftcount];
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
llong JSph::GetAllocMemoryCpu()const{  
  //-Allocated in AllocMemoryCase().
  llong s=0;
  //-Allocated in AllocMemoryFloating().
  if(FtObjs)s+=sizeof(StFloatingData)*FtCount;
  //-Allocated in other objects.
  if(PartsOut)s+=PartsOut->GetAllocMemory();
  if(ViscoTime)s+=ViscoTime->GetAllocMemory();
  if(DtFixed)s+=DtFixed->GetAllocMemory();
  if(AccInput)s+=AccInput->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSph::LoadConfig(const JCfgRun *cfg){
  const char* met="LoadConfig";
  TimerTot.Start();
  Stable=cfg->Stable;
  SvDouble=false;
  DirOut=fun::GetDirWithSlash(cfg->DirOut);
  CaseName=cfg->CaseName; 
  DirCase=fun::GetDirWithSlash(fun::GetDirParent(CaseName));
  CaseName=CaseName.substr(DirCase.length());
  if(!CaseName.length())RunException(met,"Name of the case for execution was not indicated.");
  RunName=(cfg->RunName.length()? cfg->RunName: CaseName);
  PartBeginDir=cfg->PartBeginDir; PartBegin=cfg->PartBegin; PartBeginFirst=cfg->PartBeginFirst;

  //-Opciones de salida:
  SvData=byte(SDAT_None); 
  if(cfg->Sv_Csv&&!WithMpi)SvData|=byte(SDAT_Csv);
  if(cfg->Sv_Binx)SvData|=byte(SDAT_Binx);
  if(cfg->Sv_Info)SvData|=byte(SDAT_Info);
  if(cfg->Sv_Vtk)SvData|=byte(SDAT_Vtk);

  SvRes=cfg->SvRes;
  SvTimers=cfg->SvTimers;
  SvDomainVtk=cfg->SvDomainVtk;

  printf("\n");
  RunTimeDate=fun::GetDateTime();
  Log->Printf("[Initialising %s v%s  %s]",ClassName.c_str(),GetVersionStr().c_str(),RunTimeDate.c_str());

  string tx=fun::VarStr("CaseName",CaseName);
  tx=tx+";\n"+fun::VarStr("DirCase",DirCase)+";\n"+fun::VarStr("RunName",RunName)+";\n"+fun::VarStr("DirOut",DirOut)+";";
  if(PartBegin){
    Log->Print(fun::VarStr("PartBegin",PartBegin));
    Log->Print(fun::VarStr("PartBeginDir",PartBeginDir));
    Log->Print(fun::VarStr("PartBeginFirst",PartBeginFirst));
  }

  LoadCaseConfig();

  //-Aplies configuration using command line.
  if(cfg->PosDouble==0){      SvDouble=false; }
  else if(cfg->PosDouble==1){ SvDouble=false; }
  else if(cfg->PosDouble==2){ SvDouble=true;  }
  if(cfg->TStep)TStep=cfg->TStep;
  if(cfg->TVisco){ TVisco=cfg->TVisco; Visco=cfg->Visco; }
  if(cfg->ViscoBoundFactor>=0)ViscoBoundFactor=cfg->ViscoBoundFactor;

  if(cfg->Shifting>=0){
    switch(cfg->Shifting){
      case 0:  TShifting=SHIFT_None;     break;
      case 1:  TShifting=SHIFT_Full;     break; 
      /*case 1:  TShifting=SHIFT_NoBound;  break;
      case 2:  TShifting=SHIFT_NoFixed;  break;
      case 3:  TShifting=SHIFT_Full;     break;*/
      default: RunException(met,"Shifting mode is not valid.");
    }
    if(TShifting!=SHIFT_None){
      ShiftCoef=-2;
      TensileN=0.1f;
      TensileR=3.0f;
    }
    else ShiftCoef=0;
  }

  if(cfg->FtPause>=0)FtPause=cfg->FtPause;
  if(cfg->TimeMax>0)TimeMax=cfg->TimeMax;
  //-Configuration of JTimeOut with TimePart.
  TimeOut=new JTimeOut();
  if(cfg->TimePart>=0){
    TimePart=cfg->TimePart;
    TimeOut->Config(TimePart);
  }
  else TimeOut->Config(FileXml,"case.execution.special.timeout",TimePart);

  CellOrder=cfg->CellOrder;
  CellMode=cfg->CellMode;
  if(cfg->DomainMode==1){
    ConfigDomainParticles(cfg->DomainParticlesMin,cfg->DomainParticlesMax);
    ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin,cfg->DomainParticlesPrcMax);
  }
  else if(cfg->DomainMode==2)ConfigDomainFixed(cfg->DomainFixedMin,cfg->DomainFixedMax);
  if(cfg->RhopOutModif){
    RhopOutMin=cfg->RhopOutMin; RhopOutMax=cfg->RhopOutMax;
  }
  RhopOut=(RhopOutMin<RhopOutMax);
  if(!RhopOut){ RhopOutMin=-FLT_MAX; RhopOutMax=FLT_MAX; }
}

//==============================================================================
/// Loads the case configuration to be executed.
//==============================================================================
void JSph::LoadCaseConfig(){
  const char* met="LoadCaseConfig";
  FileXml=DirCase+CaseName+".xml";
  if(!fun::FileExists(FileXml))RunException(met,"Case configuration was not found.",FileXml);
  JXml xml; xml.LoadFile(FileXml);
  JSpaceCtes ctes;     ctes.LoadXmlRun(&xml,"case.execution.constants");
  JSpaceEParms eparms; eparms.LoadXml(&xml,"case.execution.parameters");
  JSpaceParts parts;   parts.LoadXml(&xml,"case.execution.particles");

  //-Execution parameters.
  switch(eparms.GetValueInt("PosDouble",true,0)){
    case 0:  SvDouble=false;  break;
    case 1:  SvDouble=false;  break;
    case 2:  SvDouble=true;   break;
    default: RunException(met,"PosDouble value is not valid.");
  }
  switch(eparms.GetValueInt("RigidAlgorithm",true,1)){ //(DEM)
    case 1:  UseDEM=false;  break;
    case 2:  UseDEM=true;   break;
    default: RunException(met,"Rigid algorithm is not valid.");
  }
  switch(eparms.GetValueInt("StepAlgorithm",true,2)){
    case 2:  TStep=STEP_Symplectic;  break;
    default: RunException(met,"Step algorithm is not valid.");
  }

	switch(eparms.GetValueInt("Kernel",true,0)){
    case 0:  TKernel=KERNEL_Quintic;  break;
		case 1:  TKernel=KERNEL_Wendland;  break;
    default: RunException(met,"Kernel choice is not valid.");
  }

  switch(eparms.GetValueInt("ViscoTreatment",true,1)){
    case 1:  TVisco=VISCO_Artificial;  break;
    default: RunException(met,"Viscosity treatment is not valid.");
  }
  Visco=eparms.GetValueFloat("Visco");
  ViscoBoundFactor=eparms.GetValueFloat("ViscoBoundFactor",true,1.f);
  string filevisco=eparms.GetValueStr("ViscoTime",true);
  if(!filevisco.empty()){
    ViscoTime=new JSphVisco();
    ViscoTime->LoadFile(DirCase+filevisco);
  }

  switch(eparms.GetValueInt("Slip/No Slip Conditions",true,0)){
    case 0:  TSlipCond=SLIPCOND_None;     break;
    case 1:  TSlipCond=SLIPCOND_NoSlip;     break;
    case 2:  TSlipCond=SLIPCOND_Slip;  break;
    default: RunException(met,"Slip/No Slip Condition mode is not valid.");
  }

  switch(eparms.GetValueInt("Shifting",true,1)){
    case 0:  TShifting=SHIFT_None;     break;
    case 1:  TShifting=SHIFT_Full;     break;
		case 2:	 TShifting=SHIFT_Max;			 break;
    default: RunException(met,"Shifting mode is not valid.");
  }

  if(TShifting!=SHIFT_None){
    ShiftCoef=eparms.GetValueFloat("ShiftCoef",true,0.1f);
    ShiftOffset=eparms.GetValueFloat("ShiftOffset",true,0.2f);
    TensileN=eparms.GetValueFloat("TensileN",true,0.1f);
    TensileR=eparms.GetValueFloat("TensileR",true,3.0f);
		BetaShift=eparms.GetValueFloat("BetaShift",true,45.0f);
		AlphaShift=eparms.GetValueDouble("AlphaShift",true,0.0f);
  }

  FreeSurface=eparms.GetValueFloat("FreeSurface",true,1.6f);
	

  Tolerance=eparms.GetValueDouble("Solver Tolerance",true,1e-5f);
  Iterations=eparms.GetValueInt("Max Iterations",true,100);
	Restart=eparms.GetValueInt("Restart",true,50);
	MatrixMemory=eparms.GetValueInt("MatrixMemory",true,80);

  switch(eparms.GetValueInt("Preconditioner",true,0)){
    case 0:  TPrecond=PRECOND_Jacobi;   break;
    case 1:  TPrecond=PRECOND_AMG;      break;
    default: RunException(met,"Preconditioner is not valid.");
  }

  if(TPrecond==PRECOND_AMG){   
    switch(eparms.GetValueInt("AMG Interpolation",true,0)){
      case 0:  TAMGInter=AMGINTER_AG;   break;
      case 1:  TAMGInter=AMGINTER_SAG;  break;
      default: RunException(met,"AMG Interpolation is not valid.");
  }
    StrongConnection=eparms.GetValueFloat("Strong Connection Threshold",true,0.3f);
    JacobiWeight=eparms.GetValueFloat("Jacobi Weight",true,0.9999999f);
    Presmooth=eparms.GetValueInt("Presmooth Steps",true,0);
    Postsmooth=eparms.GetValueInt("Postsmooth Steps",true,0);
    CoarseCutoff=eparms.GetValueInt("Coarsening Cutoff",true,2500);
    CoarseLevels=eparms.GetValueInt("Coarse Levels",true,0);
  }

	switch(eparms.GetValueInt("NegativePressureBound",true,1)){
    case 0:  NegativePressureBound=false;     break;
    case 1:  NegativePressureBound=true;     break;
    default: RunException(met,"NegativePressureBound can only be 0 (no) or 1 (yes)");
  }

  FtPause=eparms.GetValueFloat("FtPause",true,0);
  TimeMax=eparms.GetValueDouble("TimeMax");
  TimePart=eparms.GetValueDouble("TimeOut");

  DtIni=eparms.GetValueDouble("DtIni",true,0);
  DtMin=eparms.GetValueDouble("DtMin",true,0);
  CoefDtMin=eparms.GetValueFloat("CoefDtMin",true,0.05f);
  DtAllParticles=(eparms.GetValueInt("DtAllParticles",true,0)==1);

  string filedtfixed=eparms.GetValueStr("DtFixed",true);
  if(!filedtfixed.empty()){
    DtFixed=new JSphDtFixed();
    DtFixed->LoadFile(DirCase+filedtfixed);
  }
  if(eparms.Exists("RhopOutMin"))RhopOutMin=eparms.GetValueFloat("RhopOutMin");
  if(eparms.Exists("RhopOutMax"))RhopOutMax=eparms.GetValueFloat("RhopOutMax");
  PartsOutMax=eparms.GetValueFloat("PartsOutMax",true,1);

  //-Configuration of periodic boundaries.
  if(eparms.Exists("XPeriodicIncY")){ PeriXinc.y=eparms.GetValueDouble("XPeriodicIncY"); PeriX=true; }
  if(eparms.Exists("XPeriodicIncZ")){ PeriXinc.z=eparms.GetValueDouble("XPeriodicIncZ"); PeriX=true; }
  if(eparms.Exists("YPeriodicIncX")){ PeriYinc.x=eparms.GetValueDouble("YPeriodicIncX"); PeriY=true; }
  if(eparms.Exists("YPeriodicIncZ")){ PeriYinc.z=eparms.GetValueDouble("YPeriodicIncZ"); PeriY=true; }
  if(eparms.Exists("ZPeriodicIncX")){ PeriZinc.x=eparms.GetValueDouble("ZPeriodicIncX"); PeriZ=true; }
  if(eparms.Exists("ZPeriodicIncY")){ PeriZinc.y=eparms.GetValueDouble("ZPeriodicIncY"); PeriZ=true; }
  if(eparms.Exists("XYPeriodic")){ PeriXY=PeriX=PeriY=true; PeriXZ=PeriYZ=false; PeriXinc=PeriYinc=TDouble3(0); }
  if(eparms.Exists("XZPeriodic")){ PeriXZ=PeriX=PeriZ=true; PeriXY=PeriYZ=false; PeriXinc=PeriZinc=TDouble3(0); }
  if(eparms.Exists("YZPeriodic")){ PeriYZ=PeriY=PeriZ=true; PeriXY=PeriXZ=false; PeriYinc=PeriZinc=TDouble3(0); }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);

  //-Configuration of domain size.
  float incz=eparms.GetValueFloat("IncZ",true,0.f);
  if(incz){
    ClearCfgDomain();
    CfgDomainParticlesPrcMax.z=incz;
  }
  if(eparms.Exists("DomainParticles")){
    string key="DomainParticles";
    ConfigDomainParticles(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  }
  if(eparms.Exists("DomainParticlesPrc")){
    string key="DomainParticlesPrc";
    ConfigDomainParticlesPrc(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  }
  if(eparms.Exists("DomainFixed")){
    string key="DomainFixed";
    ConfigDomainFixed(TDouble3(eparms.GetValueNumDouble(key,0),eparms.GetValueNumDouble(key,1),eparms.GetValueNumDouble(key,2)),TDouble3(eparms.GetValueNumDouble(key,3),eparms.GetValueNumDouble(key,4),eparms.GetValueNumDouble(key,5)));
  }

  //-Predefined constantes.
  if(ctes.GetEps()!=0)Log->Print("\n*** Attention: Eps value is not used (this correction is deprecated).\n");
  H=(float)ctes.GetH();
  CteB=(float)ctes.GetB();
  Gamma=(float)ctes.GetGamma();
  RhopZero=(float)ctes.GetRhop0();
  CFLnumber=(float)ctes.GetCFLnumber();
  Dp=ctes.GetDp();
  Gravity=ToTFloat3(ctes.GetGravity());
  GravityDbl=ctes.GetGravity();
  MassFluid=(float)ctes.GetMassFluid();
  MassBound=(float)ctes.GetMassBound();

  //-Particle data.
  CaseNp=parts.Count();
  CaseNfixed=parts.Count(PT_Fixed);
  CaseNmoving=parts.Count(PT_Moving);
  CaseNfloat=parts.Count(PT_Floating);
  CaseNfluid=parts.Count(PT_Fluid);
  CaseNbound=CaseNp-CaseNfluid;
  CaseNpb=CaseNbound-CaseNfloat;

  NpDynamic=ReuseIds=false;
  TotalNp=CaseNp; IdMax=CaseNp-1;

  //-Loads and configures MK of particles.
  LoadMkInfo(&parts);
	
  //-Configuration of WaveGen.
  if(xml.GetNode("case.execution.special.wavepaddles",false)){
    WaveGen=new JWaveGen(Log,DirCase,&xml,"case.execution.special.wavepaddles");
		PistonPosX=eparms.GetValueDouble("PistonPosX",true,0.0f)+0.5*Dp;
		PistonPosZ=eparms.GetValueDouble("PistonPosZ",true,0.0f)+0.5*Dp;
		PistonYmin=eparms.GetValueDouble("PistonYmin",true,0.0f)+0.5*Dp;
		PistonYmax=eparms.GetValueDouble("PistonYmax",true,0.0f)-0.5*Dp;
		DampingPointX=eparms.GetValueDouble("DampingPointX",true,0.0f);
		DampingLengthX=eparms.GetValueDouble("DampingLengthX",true,0.0f);
  }

  //-Configuration of AccInput.
  if(xml.GetNode("case.execution.special.accinputs",false)){
    AccInput=new JSphAccInput(Log,DirCase,&xml,"case.execution.special.accinputs");
  }

  //-Loads and configures MOTION.
  MotionObjCount=0;
  for(unsigned c=0;c<parts.CountBlocks();c++){
    const JSpacePartBlock &block=parts.GetBlock(c);
    if(block.Type==PT_Moving){
      if(MotionObjCount>=255)RunException(met,"The number of mobile objects exceeds the maximum.");
      MotionObjBegin[MotionObjCount]=block.GetBegin();
      MotionObjBegin[MotionObjCount+1]=MotionObjBegin[MotionObjCount]+block.GetCount();
      if(WaveGen)WaveGen->ConfigPaddle(block.GetMkType(),MotionObjCount,block.GetBegin(),block.GetCount());
      MotionObjCount++;
    }
  }

  if(MotionObjCount){
    Motion=new JSphMotion();
    if(int(MotionObjCount)<Motion->Init(&xml,"case.execution.motion",DirCase))RunException(met,"The number of mobile objects is lower than expected.");
  }

  //-Loads floating objects.
  FtCount=parts.CountBlocks(PT_Floating);
  if(FtCount){
    AllocMemoryFloating(FtCount);
    unsigned cobj=0;
    for(unsigned c=0;c<parts.CountBlocks()&&cobj<FtCount;c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type==PT_Floating){
        const JSpacePartBlock_Floating &fblock=(const JSpacePartBlock_Floating &)block;
        StFloatingData* fobj=FtObjs+cobj;
        fobj->mkbound=fblock.GetMkType();
        fobj->begin=fblock.GetBegin();
        fobj->count=fblock.GetCount();
        fobj->mass=(float)fblock.GetMassbody();
        fobj->massp=fobj->mass/fobj->count;
        fobj->radius=0;
        fobj->center=fblock.GetCenter();
        fobj->fvel=ToTFloat3(fblock.GetVelini());
        fobj->fomega=ToTFloat3(fblock.GetOmegaini());
        cobj++;
      }
    }
  }
  else UseDEM=false;

  //-Carga datos DEM de objetos. (DEM)
  //-Loads DEM data for the objects. (DEM)
  if(UseDEM){
    memset(DemObjs,0,sizeof(StDemData)*DemObjsSize);
    for(unsigned c=0;c<parts.CountBlocks();c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type!=PT_Fluid){
        unsigned cmk=0;
        for(;cmk<MkListBound && MkList[cmk].mk!=unsigned(block.GetMk());cmk++);
        if(cmk>=MkListBound)RunException(met,"Error loading DEM objects.");
        const unsigned tav=CODE_GetTypeAndValue(MkList[c].code);
        if(block.Type==PT_Floating){
          const JSpacePartBlock_Floating &fblock=(const JSpacePartBlock_Floating &)block;
          DemObjs[tav].mass=(float)fblock.GetMassbody();
          DemObjs[tav].massp=(float)(fblock.GetMassbody()/fblock.GetCount());
        }
        else DemObjs[tav].massp=MassBound;
        if(!block.ExistsSubValue("Young_Modulus","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Young_Modulus is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("PoissonRatio","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of PoissonRatio is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("Kfric","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Kfric is invalid.",block.GetMk()));
        if(!block.ExistsSubValue("Restitution_Coefficient","value"))RunException(met,fun::PrintStr("Object mk=%u - Value of Restitution_Coefficient is invalid.",block.GetMk()));
        DemObjs[tav].young=block.GetSubValueFloat("Young_Modulus","value",true,0);
        DemObjs[tav].poisson=block.GetSubValueFloat("PoissonRatio","value",true,0);
        DemObjs[tav].tau=(DemObjs[tav].young? (1-DemObjs[tav].poisson*DemObjs[tav].poisson)/DemObjs[tav].young: 0);
        DemObjs[tav].kfric=block.GetSubValueFloat("Kfric","value",true,0);
        DemObjs[tav].restitu=block.GetSubValueFloat("Restitution_Coefficient","value",true,0);
        if(block.ExistsValue("Restitution_Coefficient_User"))DemObjs[tav].restitu=block.GetValueFloat("Restitution_Coefficient_User");
      }
    }
  }

  NpMinimum=CaseNp-unsigned(PartsOutMax*CaseNfluid);
  Log->Print("**Basic case configuration is loaded");
}

//==============================================================================
// Shows coefficients used for DEM objects.
//==============================================================================
void JSph::VisuDemCoefficients()const{
  //-Gets info for each block of particles.
  Log->Printf("Coefficients for DEM:");
  for(unsigned c=0;c<MkListSize;c++){
    const word code=MkList[c].code;
    const word type=CODE_GetType(code);
    const unsigned tav=CODE_GetTypeAndValue(MkList[c].code);
    if(type==CODE_TYPE_FIXED || type==CODE_TYPE_MOVING || type==CODE_TYPE_FLOATING){
      Log->Printf("  Object %s  mkbound:%u  mk:%u",(type==CODE_TYPE_FIXED? "Fixed": (type==CODE_TYPE_MOVING? "Moving": "Floating")),MkList[c].mktype,MkList[c].mk);
      //Log->Printf("    type: %u",type);
      Log->Printf("    Young_Modulus: %g",DemObjs[tav].young);
      Log->Printf("    PoissonRatio.: %g",DemObjs[tav].poisson);
      Log->Printf("    Kfric........: %g",DemObjs[tav].kfric);
      Log->Printf("    Restitution..: %g",DemObjs[tav].restitu);
    }
  }
}

//==============================================================================
/// Initialisation of MK information.
//==============================================================================
void JSph::ResetMkInfo(){
  delete[] MkList; MkList=NULL;
  MkListSize=MkListFixed=MkListMoving=MkListFloat=MkListBound=MkListFluid=0;
}

//==============================================================================
/// Load MK information of particles.
//==============================================================================
void JSph::LoadMkInfo(const JSpaceParts *parts){
  ResetMkInfo();
  MkListSize=parts->CountBlocks();
  MkListFixed=parts->CountBlocks(PT_Fixed);
  MkListMoving=parts->CountBlocks(PT_Moving);
  MkListFloat=parts->CountBlocks(PT_Floating);
  MkListFluid=parts->CountBlocks(PT_Fluid);
  MkListBound=MkListFixed+MkListMoving+MkListFloat;
  //-Allocates memory.
  MkList=new StMkInfo[MkListSize];
  //-Gets info for each block of particles.
  for(unsigned c=0;c<MkListSize;c++){
    const JSpacePartBlock &block=parts->GetBlock(c);
    MkList[c].begin=block.GetBegin();
    MkList[c].count=block.GetCount();
    MkList[c].mk=block.GetMk();
    MkList[c].mktype=block.GetMkType();
    switch(block.Type){
      case PT_Fixed:     MkList[c].code=CodeSetType(0,PART_BoundFx,c);                           break;
      case PT_Moving:    MkList[c].code=CodeSetType(0,PART_BoundMv,c-MkListFixed);               break;
      case PT_Floating:  MkList[c].code=CodeSetType(0,PART_BoundFt,c-MkListFixed-MkListMoving);  break;
      case PT_Fluid:     MkList[c].code=CodeSetType(0,PART_Fluid,c-MkListBound);                 break;
    }
  }
}

//==============================================================================
/// Returns the block in MkList according to a given Id.
//==============================================================================
unsigned JSph::GetMkBlockById(unsigned id)const{
  unsigned c=0;
  for(;c<MkListSize && id>=(MkList[c].begin+MkList[c].count);c++);
  return(c);
}

//==============================================================================
/// Returns the block in MkList according to a given MK.
//==============================================================================
unsigned JSph::GetMkBlockByMk(word mk)const{
  unsigned c=0;
  for(;c<MkListSize && unsigned(mk)!=MkList[c].mk;c++);
  return(c);
}

//==============================================================================
/// Returns the code of a particle according to the given parameters.
//==============================================================================
word JSph::CodeSetType(word code,TpParticle type,unsigned value)const{
  const char met[]="CodeSetType"; 
  //-Chooses type.
  word tp;
  if(type==PART_BoundFx)tp=CODE_TYPE_FIXED;
  else if(type==PART_BoundMv)tp=CODE_TYPE_MOVING;
  else if(type==PART_BoundFt)tp=CODE_TYPE_FLOATING;
  else if(type==PART_Fluid)tp=CODE_TYPE_FLUID;
  else RunException(met,"Type of particle is invalid.");
  //-Checks the value.
  word v=word(value&CODE_MASKVALUE);
  if(unsigned(v)!=value)RunException(met,"The value is invalid.");
  //-Returns the new code.
  return(code&(~CODE_MASKTYPEVALUE)|tp|v);
}

//==============================================================================
/// ES:
/// Carga el codigo de grupo de las particulas y marca las nout ultimas
/// particulas como excluidas.
/// - EN:
/// Loads the code of a particle group and flags the last "nout" 
/// particles as excluded. 
//==============================================================================
void JSph::LoadCodeParticles(unsigned np,const unsigned *idp,word *code)const{
  const char met[]="LoadCodeParticles"; 
  //-Assigns code to each group of particles (moving & floating).
  const unsigned finfixed=CaseNfixed;
  const unsigned finmoving=finfixed+CaseNmoving;
  const unsigned finfloating=finmoving+CaseNfloat;
  for(unsigned p=0;p<np;p++){
    const unsigned id=idp[p];
    word cod=0;
    unsigned cmk=GetMkBlockById(id);
    if(id<finfixed)cod=CodeSetType(cod,PART_BoundFx,cmk);
    else if(id<finmoving){
      cod=CodeSetType(cod,PART_BoundMv,cmk-MkListFixed);
      if(cmk-MkListFixed>=MotionObjCount)RunException(met,"Motion code of particles was not found.");
    }
    else if(id<finfloating){
      cod=CodeSetType(cod,PART_BoundFt,cmk-MkListFixed-MkListMoving);
      if(cmk-MkListFixed-MkListMoving>=FtCount)RunException(met,"Floating code of particles was not found.");
    }
    else{
      cod=CodeSetType(cod,PART_Fluid,cmk-MkListBound);
      if(cmk-MkListBound>=MkListSize)RunException(met,"Fluid code of particles was not found.");
    }
    code[p]=cod;
  }
}

//==============================================================================
/// Resizes limits of the map according to case configuration.
//==============================================================================
void JSph::ResizeMapLimits(){
  Log->Print(string("MapRealPos(border)=")+fun::Double3gRangeStr(MapRealPosMin,MapRealPosMax));
  tdouble3 dmin=MapRealPosMin,dmax=MapRealPosMax;
  if(CfgDomainParticles){
    tdouble3 dif=dmax-dmin;
    dmin=dmin-dif*CfgDomainParticlesPrcMin;
    dmax=dmax+dif*CfgDomainParticlesPrcMax;
    dmin=dmin-CfgDomainParticlesMin;
    dmax=dmax+CfgDomainParticlesMax;
  }
  else{ dmin=CfgDomainFixedMin; dmax=CfgDomainFixedMax; }
  if(dmin.x>MapRealPosMin.x||dmin.y>MapRealPosMin.y||dmin.z>MapRealPosMin.z||dmax.x<MapRealPosMax.x||dmax.y<MapRealPosMax.y||dmax.z<MapRealPosMax.z)RunException("ResizeMapLimits","Domain limits is not valid.");
  if(!PeriX){ MapRealPosMin.x=dmin.x; MapRealPosMax.x=dmax.x; }
  if(!PeriY){ MapRealPosMin.y=dmin.y; MapRealPosMax.y=dmax.y; }
  if(!PeriZ){ MapRealPosMin.z=dmin.z; MapRealPosMax.z=dmax.z; }
}

//==============================================================================
/// Configures value of constants.
//==============================================================================
void JSph::ConfigConstants(bool simulate2d){
  const char* met="ConfigConstants";
  //-Computation of constants.
  const double h=H;
  Cs0=sqrt(double(Gamma)*double(CteB)/double(RhopZero));
  if(!DtIni)DtIni=h/Cs0;
  if(!DtMin)DtMin=(h/Cs0)*CoefDtMin; 
  Eta2=float((h*1.0e-5)*(h*1.0e-5));
  H2=float(h*h);
	
	if(TKernel==KERNEL_Quintic){
		//QUINTIC SPLINE
		Dosh=float(h*3); 
		Fourh2=float(h*h*9.0f); 
		if(simulate2d){
			Awen=float(7.0/(478.0*PI*h*h)); 
			Bwen=float(7.0/(478.0*PI*h*h*h));
		}
		else{
			Awen=float(1.0/(120.0*PI*h*h*h)); 
			Bwen=float(1.0/(120.0*PI*h*h*h*h));
		}
	}
	else if(TKernel==KERNEL_Wendland){
		//WENDLAND KERNEL
		Dosh=float(h*2); 
		Fourh2=float(h*h*4.0f); 
		if(simulate2d){
			Awen=float(7.0/(4.0*PI*h*h)); 
			Bwen=-float(35.0/(4.0*PI*h*h*h));
		}
		else{
			Awen=float(0.41778/(h*h*h));
			Bwen=-float(2.08891/(h*h*h*h));
		}
	}

  VisuConfig();
}

//==============================================================================
/// Prints out configuration of the case.
//==============================================================================
void JSph::VisuConfig()const{
  const char* met="VisuConfig";
  Log->Print(Simulate2D? "**2D-Simulation parameters:": "**3D-Simulation parameters:");
  Log->Print(fun::VarStr("CaseName",CaseName));
  Log->Print(fun::VarStr("RunName",RunName));
  Log->Print(fun::VarStr("SvDouble",SvDouble));
  Log->Print(fun::VarStr("SvTimers",SvTimers));
  Log->Print(fun::VarStr("StepAlgorithm",GetStepName(TStep)));
  if(TStep==STEP_None)RunException(met,"StepAlgorithm value is invalid.");
  Log->Print(fun::VarStr("Kernel",GetKernelName(TKernel)));
  Log->Print(fun::VarStr("Viscosity",GetViscoName(TVisco)));
  Log->Print(fun::VarStr("Visco",Visco));
  Log->Print(fun::VarStr("ViscoBoundFactor",ViscoBoundFactor));
  if(ViscoTime)Log->Print(fun::VarStr("ViscoTime",ViscoTime->GetFile()));
  Log->Print(fun::VarStr("Shifting",GetShiftingName(TShifting)));
  if(TShifting!=SHIFT_None){
    Log->Print(fun::VarStr("ShiftCoef",ShiftCoef));
    Log->Print(fun::VarStr("TensileN",TensileN));
    Log->Print(fun::VarStr("TensileR",TensileR));
  }
  Log->Print(fun::VarStr("FreeSurface",FreeSurface));
  Log->Print(fun::VarStr("FloatingFormulation",(!FtCount? "None": (UseDEM? "SPH+DEM": "SPH"))));
  Log->Print(fun::VarStr("FloatingCount",FtCount));
  if(FtCount)Log->Print(fun::VarStr("FtPause",FtPause));
  Log->Print(fun::VarStr("CaseNp",CaseNp));
  Log->Print(fun::VarStr("CaseNbound",CaseNbound));
  Log->Print(fun::VarStr("CaseNfixed",CaseNfixed));
  Log->Print(fun::VarStr("CaseNmoving",CaseNmoving));
  Log->Print(fun::VarStr("CaseNfloat",CaseNfloat));
  Log->Print(fun::VarStr("CaseNfluid",CaseNfluid));
  Log->Print(fun::VarStr("PeriodicActive",PeriActive));
  if(PeriXY)Log->Print(fun::VarStr("PeriodicXY",PeriXY));
  if(PeriXZ)Log->Print(fun::VarStr("PeriodicXZ",PeriXZ));
  if(PeriYZ)Log->Print(fun::VarStr("PeriodicYZ",PeriYZ));
  if(PeriX)Log->Print(fun::VarStr("PeriodicXinc",PeriXinc));
  if(PeriY)Log->Print(fun::VarStr("PeriodicYinc",PeriYinc));
  if(PeriZ)Log->Print(fun::VarStr("PeriodicZinc",PeriZinc));
  Log->Print(fun::VarStr("Dx",Dp));
  Log->Print(fun::VarStr("H",H));
  Log->Print(fun::VarStr("CoefficientH",H/(Dp*sqrt(Simulate2D? 2.f: 3.f))));
  Log->Print(fun::VarStr("CteB",CteB));
  Log->Print(fun::VarStr("Gamma",Gamma));
  Log->Print(fun::VarStr("RhopZero",RhopZero));
  Log->Print(fun::VarStr("Eps",0));
  Log->Print(fun::VarStr("Cs0",Cs0));
  Log->Print(fun::VarStr("CFLnumber",CFLnumber));
  Log->Print(fun::VarStr("DtIni",DtIni));
  Log->Print(fun::VarStr("DtMin",DtMin));
  Log->Print(fun::VarStr("DtAllParticles",DtAllParticles));
  if(DtFixed)Log->Print(fun::VarStr("DtFixed",DtFixed->GetFile()));
  Log->Print(fun::VarStr("MassFluid",MassFluid));
  Log->Print(fun::VarStr("MassBound",MassBound));
  if(TKernel==KERNEL_Wendland){
    Log->Print(fun::VarStr("Bwen (wendland)",Bwen));
  }
  if(UseDEM)VisuDemCoefficients();
  if(CaseNfloat)Log->Print(fun::VarStr("FtPause",FtPause));
  Log->Print(fun::VarStr("TimeMax",TimeMax));
  Log->Print(fun::VarStr("TimePart",TimePart));
  Log->Print(fun::VarStr("Gravity",Gravity));
  Log->Print(fun::VarStr("NpMinimum",NpMinimum));
  Log->Print(fun::VarStr("RhopOut",RhopOut));
  if(RhopOut){
    Log->Print(fun::VarStr("RhopOutMin",RhopOutMin));
    Log->Print(fun::VarStr("RhopOutMax",RhopOutMax));
  }
}

//==============================================================================
/// ES:
/// Calcula celda de las particulas y comprueba que no existan mas particulas
/// excluidas de las previstas.
/// - EN:
/// Computes cell particles and checks if there are more particles
/// excluded than expected.
//==============================================================================
void JSph::LoadDcellParticles(unsigned n,const word *code,const tdouble3 *pos,unsigned *dcell)const{
  const char met[]="LoadDcellParticles";
  for(unsigned p=0;p<n;p++){
    word codeout=CODE_GetSpecialValue(code[p]);
    if(codeout<CODE_OUTIGNORE){
      const tdouble3 ps=pos[p];
      if(ps>=DomRealPosMin && ps<DomRealPosMax){//-Particle in
        const double dx=ps.x-DomPosMin.x;
        const double dy=ps.y-DomPosMin.y;
        const double dz=ps.z-DomPosMin.z;
        unsigned cx=unsigned(dx/Scell),cy=unsigned(dy/Scell),cz=unsigned(dz/Scell);
        dcell[p]=PC__Cell(DomCellCode,cx,cy,cz);
      }
      else{ //-Particle out
        RunException(met,"Found new particles out."); //-No puede haber nuevas particulas excluidas. //-There cannot be new particles excluded.
        dcell[p]=PC__CodeOut;
      }
    }
    else dcell[p]=PC__CodeOut;
  }
}

//==============================================================================
// Configura CellOrder y ajusta orden de componentes en datos.
//==============================================================================
void JSph::ConfigCellOrder(TpCellOrder order,unsigned np,tdouble3* pos,tfloat4* velrhop){
  //-Guarda configuracion periodica en PeriodicConfig.
  PeriodicConfig.PeriActive=PeriActive;
  PeriodicConfig.PeriX=PeriX;
  PeriodicConfig.PeriY=PeriY;
  PeriodicConfig.PeriZ=PeriZ;
  PeriodicConfig.PeriXY=PeriXY;
  PeriodicConfig.PeriXZ=PeriXZ;
  PeriodicConfig.PeriYZ=PeriYZ;
  PeriodicConfig.PeriXinc=PeriXinc;
  PeriodicConfig.PeriYinc=PeriYinc;
  PeriodicConfig.PeriZinc=PeriZinc;
  //-Aplica CellOrder.
  CellOrder=order;
  if(CellOrder==ORDER_None)CellOrder=ORDER_XYZ;
  if(Simulate2D&&CellOrder!=ORDER_XYZ&&CellOrder!=ORDER_ZYX)RunException("ConfigCellOrder","In 2D simulations the value of CellOrder must be XYZ or ZYX.");
  Log->Print(fun::VarStr("CellOrder",string(GetNameCellOrder(CellOrder))));
  if(CellOrder!=ORDER_XYZ){
    //-Modifica datos iniciales de particulas.
    OrderCodeData(CellOrder,np,pos);
    OrderCodeData(CellOrder,np,velrhop);
    //-Modifica otras constantes.
    Gravity=OrderCodeValue(CellOrder,Gravity);
    MapRealPosMin=OrderCodeValue(CellOrder,MapRealPosMin);
    MapRealPosMax=OrderCodeValue(CellOrder,MapRealPosMax);
    MapRealSize=OrderCodeValue(CellOrder,MapRealSize);
    Map_PosMin=OrderCodeValue(CellOrder,Map_PosMin);
    Map_PosMax=OrderCodeValue(CellOrder,Map_PosMax);
    Map_Size=OrderCodeValue(CellOrder,Map_Size);
    //-Modifica config periodica.
    bool perix=PeriX,periy=PeriY,periz=PeriZ;
    bool perixy=PeriXY,perixz=PeriXZ,periyz=PeriYZ;
    tdouble3 perixinc=PeriXinc,periyinc=PeriYinc,perizinc=PeriZinc;
    tuint3 v={1,2,3};
    v=OrderCode(v);
    if(v.x==2){ PeriX=periy; PeriXinc=OrderCode(periyinc); }
    if(v.x==3){ PeriX=periz; PeriXinc=OrderCode(perizinc); }
    if(v.y==1){ PeriY=perix; PeriYinc=OrderCode(perixinc); }
    if(v.y==3){ PeriY=periz; PeriYinc=OrderCode(perizinc); }
    if(v.z==1){ PeriZ=perix; PeriZinc=OrderCode(perixinc); }
    if(v.z==2){ PeriZ=periy; PeriZinc=OrderCode(periyinc); }
    if(perixy){
      PeriXY=(CellOrder==ORDER_XYZ||CellOrder==ORDER_YXZ);
      PeriXZ=(CellOrder==ORDER_XZY||CellOrder==ORDER_YZX);
      PeriYZ=(CellOrder==ORDER_ZXY||CellOrder==ORDER_ZYX);
    }
    if(perixz){
      PeriXY=(CellOrder==ORDER_XZY||CellOrder==ORDER_ZXY);
      PeriXZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_ZYX);
      PeriYZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_YZX);
    }
    if(periyz){
      PeriXY=(CellOrder==ORDER_YZX||CellOrder==ORDER_ZYX);
      PeriXZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_ZXY);
      PeriYZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_XZY);
    }
  }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);
}

//==============================================================================
// Convierte pos[] y vel[] al orden dimensional original.
//==============================================================================
void JSph::DecodeCellOrder(unsigned np,tdouble3 *pos,tfloat3 *vel)const{
  if(CellOrder!=ORDER_XYZ){
    OrderDecodeData(CellOrder,np,pos);
    OrderDecodeData(CellOrder,np,vel);
  }
}

//==============================================================================
// Modifica orden de componentes de un array de tipo tfloat3.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
// Modifica orden de componentes de un array de tipo tdouble3.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tdouble3 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
// Modifica orden de componentes de un array de tipo tfloat4.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tfloat4 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
// Configura division en celdas.
//==============================================================================
void JSph::ConfigCellDivision(){
  if(CellMode!=CELLMODE_2H && CellMode!=CELLMODE_H)RunException("ConfigCellDivision","The CellMode is invalid.");
  Hdiv=(CellMode==CELLMODE_2H? 1: 2);
  Scell=Dosh/Hdiv;
  MovLimit=Scell*0.9f;
  Map_Cells=TUint3(unsigned(ceil(Map_Size.x/Scell)),unsigned(ceil(Map_Size.y/Scell)),unsigned(ceil(Map_Size.z/Scell)));
  //-Print configurantion.
  Log->Print(fun::VarStr("CellMode",string(GetNameCellMode(CellMode))));
  Log->Print(fun::VarStr("Hdiv",Hdiv));
  Log->Print(string("MapCells=(")+fun::Uint3Str(OrderDecode(Map_Cells))+")");
  //-Creates VTK file with cells of map.
  if(SvDomainVtk){
    const llong n=llong(Map_Cells.x)*llong(Map_Cells.y)*llong(Map_Cells.z);
    if(n<1000000)SaveMapCellsVtk(Scell);
    else Log->Print("\n*** Attention: File MapCells.vtk was not created because number of cells is too high.\n");
  }
}

//==============================================================================
// Establece dominio local de simulacion dentro de Map_Cells y calcula DomCellCode.
//==============================================================================
void JSph::SelecDomain(tuint3 celini,tuint3 celfin){
  const char met[]="SelecDomain";
  DomCelIni=celini;
  DomCelFin=celfin;
  DomCells=DomCelFin-DomCelIni;
  if(DomCelIni.x>=Map_Cells.x || DomCelIni.y>=Map_Cells.y || DomCelIni.z>=Map_Cells.z )RunException(met,"DomCelIni is invalid.");
  if(DomCelFin.x>Map_Cells.x || DomCelFin.y>Map_Cells.y || DomCelFin.z>Map_Cells.z )RunException(met,"DomCelFin is invalid.");
  if(DomCells.x<1 || DomCells.y<1 || DomCells.z<1 )RunException(met,"The domain of cells is invalid.");
  //-Calcula limites del dominio local.
  DomPosMin.x=Map_PosMin.x+(DomCelIni.x*Scell);
  DomPosMin.y=Map_PosMin.y+(DomCelIni.y*Scell);
  DomPosMin.z=Map_PosMin.z+(DomCelIni.z*Scell);
  DomPosMax.x=Map_PosMin.x+(DomCelFin.x*Scell);
  DomPosMax.y=Map_PosMin.y+(DomCelFin.y*Scell);
  DomPosMax.z=Map_PosMin.z+(DomCelFin.z*Scell);
  //-Ajusta limites finales.
  if(DomPosMax.x>Map_PosMax.x)DomPosMax.x=Map_PosMax.x;
  if(DomPosMax.y>Map_PosMax.y)DomPosMax.y=Map_PosMax.y;
  if(DomPosMax.z>Map_PosMax.z)DomPosMax.z=Map_PosMax.z;
  //-Calcula limites reales del dominio local.
  DomRealPosMin=DomPosMin;
  DomRealPosMax=DomPosMax;
  if(DomRealPosMax.x>MapRealPosMax.x)DomRealPosMax.x=MapRealPosMax.x;
  if(DomRealPosMax.y>MapRealPosMax.y)DomRealPosMax.y=MapRealPosMax.y;
  if(DomRealPosMax.z>MapRealPosMax.z)DomRealPosMax.z=MapRealPosMax.z;
  if(DomRealPosMin.x<MapRealPosMin.x)DomRealPosMin.x=MapRealPosMin.x;
  if(DomRealPosMin.y<MapRealPosMin.y)DomRealPosMin.y=MapRealPosMin.y;
  if(DomRealPosMin.z<MapRealPosMin.z)DomRealPosMin.z=MapRealPosMin.z;
  //-Calcula codificacion de celdas para el dominio seleccionado.
  DomCellCode=CalcCellCode(DomCells+TUint3(1));
  if(!DomCellCode)RunException(met,string("Failed to select a valid CellCode for ")+fun::UintStr(DomCells.x)+"x"+fun::UintStr(DomCells.y)+"x"+fun::UintStr(DomCells.z)+" cells (CellMode="+GetNameCellMode(CellMode)+").");
  //-Print configurantion.
  Log->Print(string("DomCells=(")+fun::Uint3Str(OrderDecode(DomCells))+")");
  Log->Print(fun::VarStr("DomCellCode",fun::UintStr(PC__GetSx(DomCellCode))+"_"+fun::UintStr(PC__GetSy(DomCellCode))+"_"+fun::UintStr(PC__GetSz(DomCellCode))));
}

//==============================================================================
// Selecciona un codigo adecuado para la codificion de celda.
//==============================================================================
unsigned JSph::CalcCellCode(tuint3 ncells){
  unsigned sxmin=2; for(;ncells.x>>sxmin;sxmin++);
  unsigned symin=2; for(;ncells.y>>symin;symin++);
  unsigned szmin=2; for(;ncells.z>>szmin;szmin++);
  unsigned smin=sxmin+symin+szmin;
  unsigned ccode=0;
  if(smin<=32){
    unsigned sx=sxmin,sy=symin,sz=szmin;
    unsigned rest=32-smin;
    while(rest){
      if(rest){ sx++; rest--; }
      if(rest){ sy++; rest--; }
      if(rest){ sz++; rest--; }
    }
    ccode=PC__GetCode(sx,sy,sz);
  }
  return(ccode);
}

//==============================================================================
// Calcula distancia maxima entre particulas y centro de cada floating.
//==============================================================================
void JSph::CalcFloatingRadius(unsigned np,const tdouble3 *pos,const unsigned *idp){
  const char met[]="CalcFloatingsRadius";
  const float overradius=1.2f; //-Porcentaje de incremento de radio
  unsigned *ridp=new unsigned[CaseNfloat];
  //-Asigna valores UINT_MAX
  memset(ridp,255,sizeof(unsigned)*CaseNfloat); 
  //-Calcula posicion segun id suponiendo que todas las particulas son normales (no periodicas).
  const unsigned idini=CaseNpb,idfin=CaseNpb+CaseNfloat;
  for(unsigned p=0;p<np;p++){
    const unsigned id=idp[p];
    if(idini<=id && id<idfin)ridp[id-idini]=p;
  }
  //-Comprueba que todas las particulas floating estan localizadas.
  for(unsigned fp=0;fp<CaseNfloat;fp++){
    if(ridp[fp]==UINT_MAX)RunException(met,"There are floating particles not found.");
  }
  //-Calcula distancia maxima entre particulas y centro de floating (todas son validas).
  float radiusmax=0;
  for(unsigned cf=0;cf<FtCount;cf++){
    StFloatingData *fobj=FtObjs+cf;
    const unsigned fpini=fobj->begin-CaseNpb;
    const unsigned fpfin=fpini+fobj->count;
    const tdouble3 fcen=fobj->center;
    double r2max=0;
    for(unsigned fp=fpini;fp<fpfin;fp++){
      const int p=ridp[fp];
      const double dx=fcen.x-pos[p].x,dy=fcen.y-pos[p].y,dz=fcen.z-pos[p].z;
      double r2=dx*dx+dy*dy+dz*dz;
      if(r2max<r2)r2max=r2;
    }
    fobj->radius=float(sqrt(r2max)*overradius);
    if(radiusmax<fobj->radius)radiusmax=fobj->radius;
  }
  //-Libera memoria.
  delete[] ridp; ridp=NULL;
  //-Comprueba que el radio maximo sea menor que las dimensiones del dominio periodico.
  if(PeriX && fabs(PeriXinc.x)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in X (%g).",radiusmax,abs(PeriXinc.x)));
  if(PeriY && fabs(PeriYinc.y)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in Y (%g).",radiusmax,abs(PeriYinc.y)));
  if(PeriZ && fabs(PeriZinc.z)<=radiusmax)RunException(met,fun::PrintStr("The floating radius (%g) is too large for periodic distance in Z (%g).",radiusmax,abs(PeriZinc.z)));
}

//==============================================================================
// Devuelve la posicion corregida tras aplicar condiciones periodicas.
//==============================================================================
tdouble3 JSph::UpdatePeriodicPos(tdouble3 ps)const{
  double dx=ps.x-MapRealPosMin.x;
  double dy=ps.y-MapRealPosMin.y;
  double dz=ps.z-MapRealPosMin.z;
  const bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=MapRealSize.x || dy>=MapRealSize.y || dz>=MapRealSize.z);
  //-Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
  if(PeriActive && out){
    bool xperi=((PeriActive&1)!=0),yperi=((PeriActive&2)!=0),zperi=((PeriActive&4)!=0);
    if(xperi){
      if(dx<0)             { dx-=PeriXinc.x; dy-=PeriXinc.y; dz-=PeriXinc.z; }
      if(dx>=MapRealSize.x){ dx+=PeriXinc.x; dy+=PeriXinc.y; dz+=PeriXinc.z; }
    }
    if(yperi){
      if(dy<0)             { dx-=PeriYinc.x; dy-=PeriYinc.y; dz-=PeriYinc.z; }
      if(dy>=MapRealSize.y){ dx+=PeriYinc.x; dy+=PeriYinc.y; dz+=PeriYinc.z; }
    }
    if(zperi){
      if(dz<0)             { dx-=PeriZinc.x; dy-=PeriZinc.y; dz-=PeriZinc.z; }
      if(dz>=MapRealSize.z){ dx+=PeriZinc.x; dy+=PeriZinc.y; dz+=PeriZinc.z; }
    }
    ps=TDouble3(dx,dy,dz)+MapRealPosMin;
  }
  return(ps);
}

//==============================================================================
// Muestra un mensaje con la memoria reservada para los datos basicos de las
// particulas.
//==============================================================================
void JSph::PrintSizeNp(unsigned np,llong size)const{
  Log->Printf("**Requested %s memory for %u particles: %.1f MB.",(Cpu? "cpu": "gpu"),np,double(size)/(1024*1024));
}

//==============================================================================
// Visualiza cabeceras de PARTs
//==============================================================================
void JSph::PrintHeadPart(){
  Log->Print("PART       PartTime      TotalSteps    Steps    Time/Sec   Finish time        ");
  Log->Print("=========  ============  ============  =======  =========  ===================");
  fflush(stdout);
}

//==============================================================================
// Establece configuracion para grabacion de particulas.
//==============================================================================
void JSph::ConfigSaveData(unsigned piece,unsigned pieces,std::string div){
  const char met[]="ConfigSaveData";
  //-Configura objeto para grabacion de particulas e informacion.
  if(SvData&SDAT_Info || SvData&SDAT_Binx){
    DataBi4=new JPartDataBi4();
    DataBi4->ConfigBasic(piece,pieces,RunCode,AppName,CaseName,Simulate2D,DirOut);
    DataBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,CasePosMin,CasePosMax,NpDynamic,ReuseIds);
    DataBi4->ConfigCtes(Dp,H,CteB,RhopZero,Gamma,MassBound,MassFluid);
    DataBi4->ConfigSimMap(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax));
    JPartDataBi4::TpPeri tperi=JPartDataBi4::PERI_None;
    if(PeriodicConfig.PeriActive){
      if(PeriodicConfig.PeriXY)tperi=JPartDataBi4::PERI_XY;
      else if(PeriodicConfig.PeriXZ)tperi=JPartDataBi4::PERI_XZ;
      else if(PeriodicConfig.PeriYZ)tperi=JPartDataBi4::PERI_YZ;
      else if(PeriodicConfig.PeriX)tperi=JPartDataBi4::PERI_X;
      else if(PeriodicConfig.PeriY)tperi=JPartDataBi4::PERI_Y;
      else if(PeriodicConfig.PeriZ)tperi=JPartDataBi4::PERI_Z;
      else RunException(met,"The periodic configuration is invalid.");
    }
    DataBi4->ConfigSimPeri(tperi,PeriodicConfig.PeriXinc,PeriodicConfig.PeriYinc,PeriodicConfig.PeriZinc);
    if(div.empty())DataBi4->ConfigSimDiv(JPartDataBi4::DIV_None);
    else if(div=="X")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_X);
    else if(div=="Y")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Y);
    else if(div=="Z")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Z);
    else RunException(met,"The division configuration is invalid.");
  }
  //-Configura objeto para grabacion de particulas excluidas.
  if(SvData&SDAT_Binx){
    DataOutBi4=new JPartOutBi4Save();
    DataOutBi4->ConfigBasic(piece,pieces,RunCode,AppName,Simulate2D,DirOut);
    DataOutBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid);
    DataOutBi4->ConfigLimits(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax),(RhopOut? RhopOutMin: 0),(RhopOut? RhopOutMax: 0));
    DataOutBi4->SaveInitial();
  }
  //-Configura objeto para grabacion de datos de floatings.
  if(SvData&SDAT_Binx && FtCount){
    DataFloatBi4=new JPartFloatBi4Save();
    DataFloatBi4->Config(AppName,DirOut,FtCount);
    for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddHeadData(cf,FtObjs[cf].mkbound,FtObjs[cf].begin,FtObjs[cf].count,FtObjs[cf].mass,FtObjs[cf].radius);
    DataFloatBi4->SaveInitial();
  }
  //-Crea objeto para almacenar las particulas excluidas hasta su grabacion.
  PartsOut=new JPartsOut();
}

//==============================================================================
// Almacena nuevas particulas excluidas hasta la grabacion del proximo PART.
//==============================================================================
void JSph::AddParticlesOut(unsigned nout,const unsigned *idp,const tdouble3* pos,const tfloat3 *vel,const float *rhop,unsigned noutrhop,unsigned noutmove){
  PartsOut->AddParticles(nout,idp,pos,vel,rhop,noutrhop,noutmove);
}

//==============================================================================
// Devuelve puntero de memoria dinamica con los datos transformados en tfloat3.
// EL PUNTERO DEBE SER LIBERADO DESPUES DE USARLO.
//==============================================================================
tfloat3* JSph::GetPointerDataFloat3(unsigned n,const tdouble3* v)const{
  tfloat3* v2=new tfloat3[n];
  for(unsigned c=0;c<n;c++)v2[c]=ToTFloat3(v[c]);
  return(v2);
}

//==============================================================================
// Graba los ficheros de datos de particulas.
//==============================================================================
void JSph::SavePartData(unsigned npok,unsigned nout,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus){
  //-Graba datos de particulas y/o informacion en formato bi4.
  if(DataBi4){
    tfloat3* posf3=NULL;
    TimerPart.Stop();
    JBinaryData* bdpart=DataBi4->AddPartInfo(Part,TimeStep,npok,nout,Nstep,TimerPart.GetElapsedTimeD()/1000.,vdom[0],vdom[1],TotalNp);
    if(infoplus && SvData&SDAT_Info){
      bdpart->SetvDouble("dtmean",(!Nstep? 0: (TimeStep-TimeStepM1)/(Nstep-PartNstep)));
      bdpart->SetvDouble("dtmin",(!Nstep? 0: PartDtMin));
      bdpart->SetvDouble("dtmax",(!Nstep? 0: PartDtMax));
      if(DtFixed)bdpart->SetvDouble("dterror",DtFixed->GetDtError(true));
      bdpart->SetvDouble("timesim",infoplus->timesim);
      bdpart->SetvUint("nct",infoplus->nct);
      bdpart->SetvUint("npbin",infoplus->npbin);
      bdpart->SetvUint("npbout",infoplus->npbout);
      bdpart->SetvUint("npf",infoplus->npf);
      bdpart->SetvUint("npbper",infoplus->npbper);
      bdpart->SetvUint("npfper",infoplus->npfper);
      bdpart->SetvLlong("cpualloc",infoplus->memorycpualloc);
      if(infoplus->gpudata){
        bdpart->SetvLlong("nctalloc",infoplus->memorynctalloc);
        bdpart->SetvLlong("nctused",infoplus->memorynctused);
        bdpart->SetvLlong("npalloc",infoplus->memorynpalloc);
        bdpart->SetvLlong("npused",infoplus->memorynpused);
      }
    }
    if(SvData&SDAT_Binx){
      if(SvDouble)DataBi4->AddPartData(npok,idp,pos,vel,rhop);
      else{
        posf3=GetPointerDataFloat3(npok,pos);
        DataBi4->AddPartData(npok,idp,posf3,vel,rhop);
      }
      /*float *press=NULL;
      if(0){//-Example saving a new array (Pressure) in files BI4.
        press=new float[npok];
        for(unsigned p=0;p<npok;p++)press[p]=(idp[p]>=CaseNbound? CteB*(pow(rhop[p]/RhopZero,Gamma)-1.0f): 0.f);
        DataBi4->AddPartData("Pressure",npok,press);
      }*/
      DataBi4->SaveFilePart();
      //delete[] press; press=NULL;//-Memory must to be deallocated after saving file because DataBi4 uses this memory space.
    }
    if(SvData&SDAT_Info)DataBi4->SaveFileInfo();
    delete[] posf3;
  }

  //-Graba ficheros VKT y/o CSV.
  if((SvData&SDAT_Csv)||(SvData&SDAT_Vtk)){
    //-Genera array con posf3 y tipo de particula.
    tfloat3* posf3=GetPointerDataFloat3(npok,pos);
    byte *type=new byte[npok];
    for(unsigned p=0;p<npok;p++){
      const unsigned id=idp[p];
      type[p]=(id>=CaseNbound? 3: (id<CaseNfixed? 0: (id<CaseNpb? 1: 2)));
    }
    //-Define campos a grabar.
    JFormatFiles2::StScalarData fields[8];
    unsigned nfields=0;
    if(idp){   fields[nfields]=JFormatFiles2::DefineField("Idp",JFormatFiles2::UInt32,1,idp);      nfields++; }
    if(vel){   fields[nfields]=JFormatFiles2::DefineField("Vel",JFormatFiles2::Float32,3,vel);    nfields++; }
    if(rhop){  fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,rhop);  nfields++; }
    if(type){  fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UChar8,1,type);   nfields++; }
    if(SvData&SDAT_Vtk)JFormatFiles2::SaveVtk(DirOut+fun::FileNameSec("PartVtk.vtk",Part),npok,posf3,nfields,fields);
    if(SvData&SDAT_Csv)JFormatFiles2::SaveCsv(DirOut+fun::FileNameSec("PartCsv.csv",Part),npok,posf3,nfields,fields);
    //-libera memoria.
    //-release of memory.
    delete[] posf3;
    delete[] type; 
  }

  //-Graba datos de particulas excluidas.
  if(DataOutBi4 && PartsOut->GetCount()){
    if(SvDouble)DataOutBi4->SavePartOut(Part,TimeStep,PartsOut->GetCount(),PartsOut->GetIdpOut(),PartsOut->GetPosOut(),PartsOut->GetVelOut(),PartsOut->GetRhopOut());
    else{
      const tfloat3* posf3=GetPointerDataFloat3(PartsOut->GetCount(),PartsOut->GetPosOut());
      DataOutBi4->SavePartOut(Part,TimeStep,PartsOut->GetCount(),PartsOut->GetIdpOut(),posf3,PartsOut->GetVelOut(),PartsOut->GetRhopOut());
      delete[] posf3;
    }
  }

  //-Graba datos de floatings.
  if(DataFloatBi4){
    if(CellOrder==ORDER_XYZ)for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddPartData(cf,FtObjs[cf].center,FtObjs[cf].fvel,FtObjs[cf].fomega);
    else                    for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddPartData(cf,OrderDecodeValue(CellOrder,FtObjs[cf].center),OrderDecodeValue(CellOrder,FtObjs[cf].fvel),OrderDecodeValue(CellOrder,FtObjs[cf].fomega));
    DataFloatBi4->SavePartFloat(Part,TimeStep,(UseDEM? DemDtForce: 0));
  }

  //-Vacia almacen de particulas excluidas.
  PartsOut->Clear();
}

//==============================================================================
// Genera los ficheros de salida de datos
//==============================================================================
void JSph::SaveData(unsigned npok,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop
  ,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus)
{
  const char met[]="SaveData";
  string suffixpartx=fun::PrintStr("_%04d",Part);

  //-Contabiliza nuevas particulas excluidas
  const unsigned noutpos=PartsOut->GetOutPosCount(),noutrhop=PartsOut->GetOutRhopCount(),noutmove=PartsOut->GetOutMoveCount();
  const unsigned nout=noutpos+noutrhop+noutmove;
  AddOutCount(noutpos,noutrhop,noutmove);

  //-Graba ficheros con datos de particulas.
  SavePartData(npok,nout,idp,pos,vel,rhop,ndom,vdom,infoplus);
  
  //-Reinicia limites de dt
  PartDtMin=DBL_MAX; PartDtMax=-DBL_MAX;

  //-Calculo de tiempo
  if(Part>PartIni||Nstep){
    TimerPart.Stop();
    double tpart=TimerPart.GetElapsedTimeD()/1000;
    double tseg=tpart/(TimeStep-TimeStepM1);
    TimerSim.Stop();
    double tcalc=TimerSim.GetElapsedTimeD()/1000;
    double tleft=(tcalc/(TimeStep-TimeStepIni))*(TimeMax-TimeStep);
    Log->Printf("Part%s  %12.6f  %12d  %7d  %9.2f  %14s",suffixpartx.c_str(),TimeStep,(Nstep+1),Nstep-PartNstep,tseg,fun::GetDateTimeAfter(int(tleft)).c_str());
  }
  else Log->Printf("Part%s        %u particles successfully stored",suffixpartx.c_str(),npok);   
  
  //-Muestra info de particulas excluidas.
  if(nout){
    PartOut+=nout;
    Log->Printf("  Particles out: %u  (total: %u)",nout,PartOut);
  }

  if(SvDomainVtk)SaveDomainVtk(ndom,vdom);
}

//==============================================================================
// Genera fichero VTK con el dominio de las particulas.
//==============================================================================
void JSph::SaveDomainVtk(unsigned ndom,const tdouble3 *vdom)const{ 
  if(vdom){
    string fname=fun::FileNameSec("Domain.vtk",Part);
    tfloat3 *vdomf3=new tfloat3[ndom*2];
    for(unsigned c=0;c<ndom*2;c++)vdomf3[c]=ToTFloat3(vdom[c]);
    JFormatFiles2::SaveVtkBoxes(DirOut+fname,ndom,vdomf3,H*0.5f);
    delete[] vdomf3;
  }
}

//==============================================================================
// Genera fichero VTK con las celdas del mapa.
//==============================================================================
void JSph::SaveMapCellsVtk(float scell)const{
  JFormatFiles2::SaveVtkCells(DirOut+"MapCells.vtk",ToTFloat3(OrderDecode(MapRealPosMin)),OrderDecode(Map_Cells),scell);
}

//==============================================================================
// Añade la informacion basica de resumen a hinfo y dinfo.
//==============================================================================
void JSph::GetResInfo(float tsim,float ttot,const std::string &headplus,const std::string &detplus,std::string &hinfo,std::string &dinfo){
  hinfo=hinfo+"#RunName;RunCode;DateTime;Np;TSimul;TSeg;TTotal;MemCpu;MemGpu;Steps;PartFiles;PartsOut;MaxParticles;MaxCells;Hw;StepAlgo;Kernel;Viscosity;ViscoValue;DeltaSPH;TMax;Nbound;Nfixed;H;RhopOut;PartsRhopOut;PartsVelOut;CellMode"+headplus;
  dinfo=dinfo+ RunName+ ";"+ RunCode+ ";"+ RunTimeDate+ ";"+ fun::UintStr(CaseNp);
  dinfo=dinfo+ ";"+ fun::FloatStr(tsim)+ ";"+ fun::FloatStr(tsim/float(TimeStep))+ ";"+ fun::FloatStr(ttot);
  dinfo=dinfo+ ";"+ fun::LongStr(MaxMemoryCpu)+ ";"+ fun::LongStr(MaxMemoryGpu);
  const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
  dinfo=dinfo+ ";"+ fun::IntStr(Nstep)+ ";"+ fun::IntStr(Part)+ ";"+ fun::UintStr(nout);
  dinfo=dinfo+ ";"+ fun::UintStr(MaxParticles)+ ";"+ fun::UintStr(MaxCells);
  dinfo=dinfo+ ";"+ Hardware+ ";"+ GetStepName(TStep)+ ";"+ GetKernelName(TKernel)+ ";"+ GetViscoName(TVisco)+ ";"+ fun::FloatStr(Visco);
  dinfo=dinfo+ ";"+ fun::FloatStr(float(TimeMax));
  dinfo=dinfo+ ";"+ fun::UintStr(CaseNbound)+ ";"+ fun::UintStr(CaseNfixed)+ ";"+ fun::FloatStr(H);
  std::string rhopcad;
  if(RhopOut)rhopcad=fun::PrintStr("(%G-%G)",RhopOutMin,RhopOutMax); else rhopcad="None";
  dinfo=dinfo+ ";"+ rhopcad+ ";"+ fun::UintStr(GetOutRhopCount())+ ";"+ fun::UintStr(GetOutMoveCount())+ ";"+ GetNameCellMode(CellMode)+ detplus;
}

//==============================================================================
// Genera fichero Run.csv con resumen de ejecucion
//==============================================================================
void JSph::SaveRes(float tsim,float ttot,const std::string &headplus,const std::string &detplus){
  const char* met="SaveRes";
  string fname=DirOut+"Run.csv";
  ofstream pf;
  pf.open(fname.c_str());
  if(pf){
    string hinfo,dinfo;
    GetResInfo(tsim,ttot,headplus,detplus,hinfo,dinfo);
    pf << hinfo << endl << dinfo << endl;
    if(pf.fail())RunException(met,"Failed writing to file.",fname);
    pf.close();
  }
  else RunException(met,"File could not be opened.",fname);
}

//==============================================================================
// Muestra resumen de ejecucion.
//==============================================================================
void JSph::ShowResume(bool stop,float tsim,float ttot,bool all,std::string infoplus){
  Log->Printf("\n[Simulation %s  %s]",(stop? "INTERRUPTED": "finished"),fun::GetDateTime().c_str());
  Log->Printf("Particles of simulation (initial): %u",CaseNp);
  if(NpDynamic)Log->Printf("Particles of simulation (total)..: %llu",TotalNp);
  if(all){
    Log->Printf("DTs adjusted to DtMin............: %d",DtModif);
    const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
    Log->Printf("Excluded particles...............: %d",nout);
    if(GetOutRhopCount())Log->Printf("Excluded particles due to RhopOut: %u",GetOutRhopCount());
    if(GetOutMoveCount())Log->Printf("Excluded particles due to Velocity: %u",GetOutMoveCount());
  }
  Log->Printf("Total Runtime....................: %f sec.",ttot);
  Log->Printf("Simulation Runtime...............: %f sec.",tsim);
  if(all){
    float tseg=tsim/float(TimeStep);
    float nstepseg=float(Nstep)/tsim;
    Log->Printf("Time per second of simulation....: %f sec.",tseg);
    Log->Printf("Steps per second.................: %f",nstepseg);
    Log->Printf("Steps of simulation..............: %d",Nstep);
    Log->Printf("PART files.......................: %d",Part-PartIni);
    while(!infoplus.empty()){
      string lin=fun::StrSplit("#",infoplus);
      if(!lin.empty()){
        string tex=fun::StrSplit("=",lin);
        string val=fun::StrSplit("=",lin);
        while(tex.size()<33)tex=tex+".";
        Log->Print(tex+": "+val);
      }
    }
  }
  Log->Printf("Maximum number of particles......: %u",MaxParticles);
  Log->Printf("Maximum number of cells..........: %u",MaxCells);
  Log->Printf("CPU Memory.......................: %lld (%.2f MB)",MaxMemoryCpu,double(MaxMemoryCpu)/(1024*1024));
  if(MaxMemoryGpu)Log->Printf("GPU Memory.......................: %lld (%.2f MB)",MaxMemoryGpu,double(MaxMemoryGpu)/(1024*1024));
}

//==============================================================================
// Devuelve el nombre del algoritmo en texto.
//==============================================================================
std::string JSph::GetStepName(TpStep tstep){
  string tx;
  if(tstep==STEP_Symplectic)tx="Symplectic";
  else tx="???";
  return(tx);
}

//==============================================================================
// Devuelve el nombre del kernel en texto.
//==============================================================================
std::string JSph::GetKernelName(TpKernel tkernel){
  string tx;
	if(tkernel==KERNEL_Quintic)tx="Quintic";
  else if(tkernel==KERNEL_Wendland)tx="Wendland";
  else tx="???";
  return(tx);
}

//==============================================================================
// Devuelve el nombre de la viscosidad en texto.
//==============================================================================
std::string JSph::GetViscoName(TpVisco tvisco){
  string tx;
  if(tvisco==VISCO_Artificial)tx="Artificial";
  else tx="???";
  return(tx);
}

//==============================================================================
// Devuelve el valor de DeltaSPH en texto.
//==============================================================================
std::string JSph::GetDeltaSphName(TpDeltaSph tdelta){
  string tx;
  if(tdelta==DELTA_None)tx="None";
  else if(tdelta==DELTA_Dynamic)tx="Dynamic";
  else if(tdelta==DELTA_DynamicExt)tx="DynamicExt";
  else tx="???";
  return(tx);
}

//==============================================================================
// Devuelve el valor de Shifting en texto.
//==============================================================================
std::string JSph::GetShiftingName(TpShifting tshift){
  string tx;
  if(tshift==SHIFT_None)tx="None";
  //else if(tshift==SHIFT_NoBound)tx="NoBound";
  //else if(tshift==SHIFT_NoFixed)tx="NoFixed";
  else if(tshift/*==SHIFT_Full*/)tx="Full";
  else tx="???";
  return(tx);
}

//==============================================================================
// Devuelve string con el nombre del temporizador y su valor.
//==============================================================================
std::string JSph::TimerToText(const std::string &name,float value){
  string ret=name;
  while(ret.length()<33)ret+=".";
  return(ret+": "+fun::FloatStr(value/1000)+" sec.");
}





