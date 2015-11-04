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
//# - Clase para encapsular la generacion de oleaje. (11-03-2015) 
//# - EN:
//# Changes:
//# ========
//# - Class to encapsulate the wave generation. (11-03-2015)
//#############################################################################

#ifndef _JWaveGen_
#define _JWaveGen_

#include "Types.h"
#include <string>
#ifdef _WITHGPU
  #include <cuda_runtime_api.h>
#endif

class JXml;
class JLog2;
class JWavePaddles;


//##############################################################################
//# JWaveGen
//##############################################################################
class JWaveGen
{
private:
  JWavePaddles* WavPad; 
  
  //-Vars. auxiliares cargadas tras Init().
  //-Auxiliary variables loaded after Init().
  bool Use_Awas;       //-AWAS-Vel or AWAS-Zsurf.
  bool Use_AwasZsurf;  //-AWAS-Zsurf.
  unsigned Count;

public:

  //==============================================================================
  /// Constructor.
  //==============================================================================
  JWaveGen(JLog2* log,std::string dirdata,JXml *sxml,const std::string &place);

  //==============================================================================
  /// Destructor.
  //==============================================================================
  ~JWaveGen();

  //==============================================================================
  /// Configura paddle con datos de las particulas.
  /// Set paddle with the particle data.
  //==============================================================================
  bool ConfigPaddle(word mkbound,word paddleid,unsigned idbegin,unsigned np);

  //==============================================================================
  /// Prepara movimiento de paddles.
  /// Prepares paddle movement.
  //==============================================================================
  void Init(double timemax,tfloat3 gravity,bool simulate2d,TpCellOrder cellorder,float massfluid,double dp,float dosh,float scell,int hdiv,tdouble3 domposmin,tdouble3 domrealposmin,tdouble3 domrealposmax);

 #ifndef HIDE_AWAS
  //==============================================================================
  /// Checks limits of paddle particles.
  //==============================================================================
  void CheckPaddleParticles(unsigned np,const tdouble3* pos,const unsigned* idp);
 #endif

  //==============================================================================
  /// Shows object configuration using Log.
  //==============================================================================
  void VisuConfig(std::string txhead,std::string txfoot);

  //==============================================================================
  /// Devuelve si es un movimiento lineal y los datos de movimiento para el intervalo indicado.
  /// Returns if it is a linear motion and the motion data for the specified interval.
  //==============================================================================
  bool GetMotion(unsigned cp,double timestep,double dt,tdouble3 &mvsimple,tmatrix4d &mvmatrix,unsigned &np,unsigned &idbegin);

  //==============================================================================
  /// Indica si debe recalcular datos para el Awas.
  /// Indicates whether to recompute data for the Awas.
  //==============================================================================
  bool CheckAwasRun(double timestep)const;

  //==============================================================================
  /// Actualiza datos del Awas para ejecucion en CPU.
  /// Updates Awas data for CPU execution.
  //==============================================================================
  void RunAwasCpu(double timestep,bool svdata,tuint3 ncells,tuint3 cellmin,const unsigned *begincell,const tdouble3 *pos,const word *code,const tfloat4 *velrhop);

 #ifdef _WITHGPU
  //==============================================================================
  /// Actualiza datos del Awas para ejecucion en GPU.
  /// Updates Awas data for GPU execution.
  //==============================================================================
  void RunAwasGpu(double timestep,bool svdata,tuint3 ncells,tuint3 cellmin,const int2 *beginendcell,const double2 *posxy,const double *posz,const word *code,const float4 *velrhop,float3 *aux);
 #endif
  
  unsigned GetCount()const{ return(Count); }
  bool UseAwas()const{ return(Use_Awas); } 
  bool UseAwasZsurf()const{ return(Use_AwasZsurf); } 

};


#endif

