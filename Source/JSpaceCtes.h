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

/// \file JSpaceCtes.h \brief Declares the class \ref JSpaceCtes.

#ifndef _JSpaceCtes_
#define _JSpaceCtes_

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - Se paso a usar double en lugar de float. (25-11-2013)
//# - El valor de Eps pasa a ser opcional para mantener compatibilidad. (08-01-2015)
//# - Se cambio Coefficient por CoefH pero manteniendo compatibilidad. (08-01-2015)
//# - Se a�adio SpeedSound para asignar valor de forma explicita. (08-01-2015)
//# - Se a�adieron comentarios al escribir el XML. (08-01-2015)
//# - Se ampliaron los limites de CFLnumber de (0.1-0.5) a (0.001-1). (08-01-2015)
//# - <speedsystem> y <speedsound> pasan a ser opcionales. (20-01-2015)
//# - <eps> solo se pasa a <constants> cuando este definido en <constantsdef>. (20-01-2015)
//# - EN:
//# Changes:
//# =========
//# - Uses double instead of float. (25-11-2013)
//# - The value of Eps becomes optional to maintain compatibility. (08-01-2015)
//# - Coefficient CoefH was changed while maintaining compatibility. (08-01-2015)
//# - Speedsound is added to assign value explicitly. (08-01-2015)
//# - Added XML comments. (08-01-2015)
//# - CFLnumber limits of (0.1-0.5) to (0001-1) were expanded. (08-01-2015)
//# - <speedsystem> and <speedsound> become optional. (20-01-2015)
//# - <eps> passes only <constants> when is set to <constantsdef>. (20-01-2015)
//#############################################################################

#include <string>
#include <vector>
#include "JObject.h"
#include "TypesDef.h"

class JXml;
class TiXmlElement;

//##############################################################################
//# JSpaceCtes
//##############################################################################
/// \brief Manages the info of constants from the input XML file.

class JSpaceCtes : protected JObject 
{
private:
  int LatticeBound;       ///<Lattice to create boundary particles on its nodes.
  int LatticeFluid;       ///<Lattice to create fluid particles on its nodes.
  tdouble3 Gravity;       ///<Gravity acceleration.
  double CFLnumber;       ///<CFL number (0.001-1).
  bool HSwlAuto;          ///<Activates the automatic computation of H_Swl.
  double HSwl;            ///<Maximum height of the volume of fluid.
  bool SpeedSystemAuto;   ///<Activates the automatic computation of SpeedSystem.
  double SpeedSystem;     ///<Maximum system speed.
  double CoefSound;       ///<Coefficient to multiply speedsystem.
  bool SpeedSoundAuto;    ///<Activates the automatic computation of SpeedSound.
  double SpeedSound;      ///<Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem).

  double CoefH;           ///<Coefficient to calculate the smoothing length H (H=coefficient*sqrt(3*dp^2) in 3D).
  double CoefHdp;         ///<Relationship between h and dp. (it is optional).
  double Gamma;           ///<Politropic constant. (1-7).
  double Rhop0;           ///<Density of reference.

  double Eps;             ///<Epsilon constant for XSPH variant.
  bool EpsDefined;        ///<Epsilon was defined in constantsdef.

  bool HAuto;             ///<Activates the automatic computation of H.
  bool BAuto;             ///<Activates the automatic computation of B.
  bool MassBoundAuto;     ///<Activates the automatic computation of MassBound.
  bool MassFluidAuto;     ///<Activates the automatic computation of MassFluid.
  double H;               ///<Smoothing length.
  double B;               ///<Constant that sets a limit for the maximum change in density.
  double MassBound;       ///<Mass of a boundary particle.
  double MassFluid;       ///<Mass of a fluid particle.

  //-Computed values:
  double Dp;              ///<Inter-particle distance.

  void ReadXmlElementAuto(JXml *sxml,TiXmlElement* node,bool optional,std::string name,double &value,bool &valueauto);
  void WriteXmlElementAuto(JXml *sxml,TiXmlElement* node,std::string name,double value,bool valueauto,std::string comment="")const;

  void ReadXmlDef(JXml *sxml,TiXmlElement* ele);
  void WriteXmlDef(JXml *sxml,TiXmlElement* ele)const;
  void ReadXmlRun(JXml *sxml,TiXmlElement* ele);
  void WriteXmlRun(JXml *sxml,TiXmlElement* ele)const;
public:
  
  JSpaceCtes();
  void Reset();
  void LoadDefault();
  void LoadXmlDef(JXml *sxml,const std::string &place);
  void SaveXmlDef(JXml *sxml,const std::string &place)const;
  void LoadXmlRun(JXml *sxml,const std::string &place);
  void SaveXmlRun(JXml *sxml,const std::string &place)const;

  int GetLatticeBound()const{ return(LatticeBound); }
  int GetLatticeFluid()const{ return(LatticeFluid); }
  tdouble3 GetGravity()const{ return(Gravity); }
  double GetCFLnumber()const{ return(CFLnumber); }
  bool GetHSwlAuto()const{ return(HSwlAuto); }
  double GetHSwl()const{ return(HSwl); }
  bool GetSpeedSystemAuto()const{ return(SpeedSystemAuto); }
  double GetSpeedSystem()const{ return(SpeedSystem); }
  double GetCoefSound()const{ return(CoefSound); }
  bool GetSpeedSoundAuto()const{ return(SpeedSoundAuto); }
  double GetSpeedSound()const{ return(SpeedSound); }
  double GetCoefH()const{ return(CoefH); }
  double GetCoefHdp()const{ return(CoefHdp); }
  double GetCoefficient()const{ return(GetCoefH()); }
  double GetGamma()const{ return(Gamma); }
  double GetRhop0()const{ return(Rhop0); }
  double GetEps()const{ return(Eps); }

  void SetLatticeBound(bool simple){ LatticeBound=(simple? 1: 2); }
  void SetLatticeFluid(bool simple){ LatticeFluid=(simple? 1: 2); }
  void SetGravity(const tdouble3& g){ Gravity=g; }
  void SetCFLnumber(double v){ 
    if(!v)RunException("SetCFLnumber","Value can not be zero.");
    if(v>1)RunException("SetCFLnumber","Value can not be higher than 1.");
    CFLnumber=v;
  }
  void SetHSwlAuto(bool on){ HSwlAuto=on; }
  void SetHSwl(double v){ HSwl=v; }
  void SetSpeedSystemAuto(bool on){ SpeedSystemAuto=on; }
  void SetSpeedSystem(double v){ SpeedSystem=v; }
  void SetCoefSound(double v){ CoefSound=v; }
  void SetSpeedSoundAuto(bool on){ SpeedSoundAuto=on; }
  void SetSpeedSound(double v){ SpeedSound=v; }
  void SetCoefH(double v){ CoefH=v; CoefHdp=0; }
  void SetCoefHdp(double v){ if(v){ CoefHdp=v; CoefH=0; } }
  void SetCoefficient(double v){ SetCoefH(v); }
  void SetGamma(double v){ Gamma=v; }
  void SetRhop0(double v){ Rhop0=v; }
  void SetEps(double v){ Eps=v; }

  bool GetHAuto()const{ return(HAuto); }
  bool GetBAuto()const{ return(BAuto); }
  bool GetMassBoundAuto()const{ return(MassBoundAuto); }
  bool GetMassFluidAuto()const{ return(MassFluidAuto); }
  double GetH()const{ return(H); }
  double GetB()const{ return(B); }
  double GetMassBound()const{ return(MassBound); }
  double GetMassFluid()const{ return(MassFluid); }

  void SetHAuto(bool on){ HAuto=on; }
  void SetBAuto(bool on){ BAuto=on; }
  void SetMassBoundAuto(bool on){ MassBoundAuto=on; }
  void SetMassFluidAuto(bool on){ MassFluidAuto=on; }
  void SetH(double v){ H=v; }
  void SetB(double v){ B=v; }
  void SetMassBound(double v){ MassBound=v; }
  void SetMassFluid(double v){ MassFluid=v; }

  double GetDp()const{ return(Dp); }
  void SetDp(double v){ Dp=v; }
};

#endif




