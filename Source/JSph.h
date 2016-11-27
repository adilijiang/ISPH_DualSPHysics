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

#ifndef _JSph_
#define _JSph_

//#############################################################################
//# ES:
//# Cambios:
//# =========
//# - El calculo de constantes en ConfigConstants() se hace usando double aunque
//#   despues se convierte a float (22/04/2013)
//# - EN:
//# Changes:
//# =========
//# - Calculating constants in ConfigConstants() is done using double although
//#	  they are then converted to float (04/22/2013)
//#############################################################################

#include "Types.h"
#include "JObject.h"
#include "JCfgRun.h"
#include "JLog2.h"
#include "JTimer.h"
#include "JTimersStep.h"
#include <float.h>
#include <string>
#include <cmath>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>

class JSphMotion;
class JPartData;
class JPartPData;
class JSphDtFixed;
class JSaveDt;
class JSphVisco;
class JWaveGen;
class JSphVarAcc;
class JSpaceParts;
class JPartDataBi4;
class JPartOutBi4Save;
class JPartFloatBi4Save;
class JPartsOut;
class JXml;

class JSph : protected JObject
{
public:
  static const unsigned int VersionMajor=70;
  static const unsigned int VersionMinor=0;
  static std::string GetVersionStr();

/// Structure with mk information.
  typedef struct {
    unsigned begin;
    unsigned count;
    unsigned mktype;
    unsigned mk;
    word code;
  }StMkInfo;

  typedef struct {
    double timesim;     //-Segundos desde el inicio de la simulacion (despues de cargar los datos iniciales).		//-Seconds from the start of the simulation (after loading the initial data).
    unsigned nct;       //-Numero de celdas usadas en el divide.													//-Number of cells used in the divide.
    unsigned npbin;     //-Numero de particulas bound dentro del area del divide (incluye particulas periodicas).	//-Number of boundary particles within the area of the divide (includes periodic particles).
    unsigned npbout;    //-Numero de particulas bound fuera del area del divide (incluye particulas periodicas).	//-Number of boundary particles outside of the area of the divide (includes periodic particles).
    unsigned npf;       //-Numero de particulas fluid (incluye particulas periodicas).								//-Number of fluid particles (includes periodic particles).
    unsigned npbper;    //-Numero de particulas bound periodicas (dentro y fuera del area del divide).				//-Number of periodic boundary particles (inside and outside the area of the split).
    unsigned npfper;    //-Numero de particulas fluid periodicas.													//-Number of periodic fluid particles.
    unsigned newnpok;   //-Numero de nuevas particulas fluid (inlet conditions).									//-Number of new fluid particles (inlet conditions)
    unsigned newnpfail; //-Numero de nuevas particulas fluid descartadas (inlet conditions).						//-Number of discarded new fluid particles (inlet conditions)
    llong memorycpualloc;
    bool gpudata;
    llong memorynpalloc;
    llong memorynpused;
    llong memorynctalloc;
    llong memorynctused;
  }StInfoPartPlus;

/// Structure with Periodic information.
  typedef struct{
    byte PeriActive;
    bool PeriX;        //-Condiciones periodicas en X.										//-Periodic conditions in X.
    bool PeriY;        //-Condiciones periodicas en Y.										//-Periodic conditions in Y.
    bool PeriZ;        //-Condiciones periodicas en Z.										//-Periodic conditions in Z.
    bool PeriXY;       //-Condiciones periodicas en X-Y.									//-Periodic conditions in X-Y.
    bool PeriXZ;       //-Condiciones periodicas en X-Z.									//-Periodic conditions in X-Z.
    bool PeriYZ;       //-Condiciones periodicas en Y-Z.									//-Periodic conditions in Y-Z.
    tdouble3 PeriXinc; //-Valor que se suma al extremo final para modificar coordenadas.	//-Value that is added at the outer limit to modify coordinates.
    tdouble3 PeriYinc; //-Valor que se suma al extremo final para modificar coordenadas.	//-Value that is added at the outer limit to modify coordinates.
    tdouble3 PeriZinc; //-Valor que se suma al extremo final para modificar coordenadas.	//-Value that is added at the outer limit to modify coordinates.
  }StPeriodic;

private:
  //-Variables de configuracion para calcular el limite del caso.
  //-Configuration variables to compute the case limits.
  bool CfgDomainParticles;
  tdouble3 CfgDomainParticlesMin,CfgDomainParticlesMax;
  tdouble3 CfgDomainParticlesPrcMin,CfgDomainParticlesPrcMax;
  tdouble3 CfgDomainFixedMin,CfgDomainFixedMax;

  //-Objeto para la grabacion de particulas e informacion en ficheros.
  //-Object for saving particles and information in files.
  JPartDataBi4 *DataBi4;           //-Para grabar particulas e info en formato bi4.			//-To store particles and info in bi4 format.
  JPartOutBi4Save *DataOutBi4;     //-Para grabar particulas excluidas en formato bi4.		//-To store excluded particles in bi4 format.
  JPartFloatBi4Save *DataFloatBi4; //-Para grabar datos de floatings en formato bi4.		//-To store floating data in bi4 format.
  JPartsOut *PartsOut;         //-Almacena las particulas excluidas hasta su grabacion.		//-Stores excluded particles until they are saved.

  //-Numero acumulado de particulas excluidas segun motivo.
  //-Total number of excluded particles according to reason for exclusion.
  unsigned OutPosCount,OutRhopCount,OutMoveCount;

  void InitVars();
  std::string CalcRunCode()const;
  void AddOutCount(unsigned outpos,unsigned outrhop,unsigned outmove){ OutPosCount+=outpos; OutRhopCount+=outrhop; OutMoveCount+=outmove; }
  void ClearCfgDomain();
  void ConfigDomainFixed(tdouble3 vmin,tdouble3 vmax);
  void ConfigDomainParticles(tdouble3 vmin,tdouble3 vmax);
  void ConfigDomainParticlesPrc(tdouble3 vmin,tdouble3 vmax);

protected:
  const bool Cpu;
  const bool WithMpi;
  JLog2 *Log;

  bool Simulate2D;  //-Activa o desactiva simulacion en 2D (anula fuerzas en eje Y).	//-Toggles 2D simulation (cancels forces in Y axis).
  bool Stable;
	bool Psimple;
  bool SvDouble;  //-Indica si en los ficheros bi4 se guarda Pos como double.			//-Indicates whether Pos is saved as double in bi4 files.

  std::string AppName;
  std::string Hardware;
  std::string RunCode;
  std::string RunTimeDate;
  std::string CaseName,DirCase,DirOut,RunName;
  std::string FileXml;

  //-Opciones de ejecucion.
  //-Execution options.
  TpStep TStep;               //-Algitmo de paso: Verlet o Symplectic.											//-Step Algorithm: Verlet or Symplectic.
  TpKernel TKernel;           //-Tipo de kernel: Cubic o Wendland.												//-Kernel type: Cubic or Wendland.
  float Awen;                 //-Constante para kernel wendland (awen).											//-Wendland kernel constant (awen).
  float Bwen;                 //-Constante para kernel wendland (bwen).											//-Wendland kernel constant (bwen).
  TpVisco TVisco;             //-Tipo de viscosidad: Artificial,...												//-Viscosity type: Artificial,...

  TpShifting TShifting; //-Tipo de Shifting: None, NoBound, NoFixed, Full.										//-Shifting type: None, NoBound, NoFixed, Full.	
  float ShiftCoef;      //-Coefficient for shifting computation.
  float FreeSurface;    //-Threshold to detect free surface.
  float TensileN;       //-Tensile Instability Coefficient
  float TensileR;       //-Tensile Instability Coefficient

  TpSlipCond TSlipCond;
  
  TpPrecond TPrecond;
  TpAMGInter TAMGInter;
  int Iterations;
  double Tolerance;
  float StrongConnection;
  float JacobiWeight;
  int Presmooth;
  int Postsmooth;
  int CoarseCutoff;
	bool NegativePressureBound; //Allow negative pressure in the boundary

  float Visco;  
  float ViscoBoundFactor;     //-Para interaccion con contorno usa Visco*ViscoBoundFactor.						//-Forboundary interaction use Visco*ViscoBoundFactor.
  JSphVisco* ViscoTime;       //-Proporciona un valor de viscosidad en funcion del instante de la simulacion.	//-Provides a viscosity value as a function of simulation time.

  bool RhopOut;                //Indica si activa la correccion de densidad RhopOut o no.						//-Indicates whether the RhopOut density correction is active or not.
  float RhopOutMin,RhopOutMax; //Limites para la correccion de RhopOut.											//-Limits for Rhopout correction.

  double TimeMax;
  double TimePart;

  double DtIni;       //-Dt inicial																				//-Initial Dt
  double DtMin;       //-Dt minimo permitido (si el calculado es menor se sustituye por DtMin).					//-Minimum allowed Dt (if the calculated is lower is replaced by DTmin).
  float CoefDtMin;    //-Coefficient to calculate minimum time step. dtmin=coefdtmin*h/speedsound (def=0.03)
  bool DtAllParticles; //-Velocity of particles used to calculate DT. 1:All, 0:Only fluid/floating (def=0)
  JSphDtFixed* DtFixed;
  JSaveDt* SaveDt;

  float PartsOutMax;  //-Porcentaje maximo de particulas excluidas permitidas.									//-Allowed percentage of fluid particles out of the domain.
  unsigned NpMinimum; //-Numero minimo de particulas permitidas.												//-Minimum number of particles allowed.

  //-Salida de resultados.
  //-Result output.
  byte SvData;        //-Combinacion de valores TpSaveDat.														//-Combination of the TpSaveDat values.
  bool SvRes;         //-Graba fichero con resumen de ejecucion.												//-Creates file with execution summary.
  bool SvTimers;      //-Obtiene tiempo para cada proceso.														//-Computes the time for each process.
  bool SvDomainVtk;   //-Graba fichero vtk con el dominio de las particulas en cada Part.						//-Stores VTK file with the domain of particles of each PART file.

  //-Constantes para calculo.
  //-Computation constants.
  float H,CteB,Gamma,CFLnumber,RhopZero;
  double Dp;
  double Cs0;
  float Delta2H;     ///<Constant for DeltaSPH. Delta2H=DeltaSph*H*2
  float MassFluid,MassBound;  
  tfloat3 Gravity;
  tdouble3 GravityDbl;
  float Dosh,H2,Fourh2,Eta2;

  //-Informacion general del caso.
  tdouble3 CasePosMin,CasePosMax;  //-Limites de particulas del caso en instante inicial.		//-Particle limits of the case in the initial instant.
  unsigned CaseNp;                 ///<Number of total particles of initial PART.  
  unsigned CaseNfixed;             ///<Number of fixed boundary particles. 
  unsigned CaseNmoving;            ///<Number of moving boundary particles. 
  unsigned CaseNfloat;             ///<Number of floating boundary particles. 
  unsigned CaseNfluid;             ///<Number of fluid particles (including the excluded ones). 
  unsigned CaseNbound;             ///<Number of boundary particles ( \ref Nfixed + \ref Nmoving + \ref Nfloat ).
  unsigned CaseNpb;                ///<Number of particles of the boundary block ( \ref Nbound - \ref Nfloat ) or ( \ref Nfixed + \ref Nmoving).

  //-Almacena informacion de Mk de las particulas.
  //-Stores information for the Mk of the particles.
  StMkInfo *MkList;    //-Almacena informacion de cada bloque Mk.											//-Stores information for each Mk block.
  unsigned MkListSize; //-Num total de bloques Mk.															//-Total number of Mk blocks.
  unsigned MkListFixed,MkListMoving,MkListFloat,MkListBound,MkListFluid; //-Num de bloques Mk de cada tipo. //-Number of Mk blocks of each type.

  //-Vars para condiciones periodicas.
  //-Variables for periodic conditions.
  StPeriodic PeriodicConfig; //-Almacena la configuracion de condiciones periodicas antes de aplicar CellOrder.	//-Stores configuration of periodic conditions before applying CellOrder
  byte PeriActive;
  bool PeriX;        //-Condiciones periodicas en X.															//-Periodic conditions in X.
  bool PeriY;        //-Condiciones periodicas en Y.															//-Periodic conditions in Y.
  bool PeriZ;        //-Condiciones periodicas en Z.															//-Periodic conditions in Z.
  bool PeriXY;       //-Condiciones periodicas en X-Y.															//-Periodic conditions in X-Y.
  bool PeriXZ;       //-Condiciones periodicas en X-Z.															//-Periodic conditions in X-Z.
  bool PeriYZ;       //-Condiciones periodicas en Y-Z.															//-Periodic conditions in Y-Z.
  tdouble3 PeriXinc; //-Valor que se suma al extremo final para modificar coordenadas.							//-Value that is added at the outer limit to modify coordinates.
  tdouble3 PeriYinc; //-Valor que se suma al extremo final para modificar coordenadas.							//-Value that is added at the outer limit to modify coordinates.
  tdouble3 PeriZinc; //-Valor que se suma al extremo final para modificar coordenadas.							//-Value that is added at the outer limit to modify coordinates.

  //-Vars para reanudacion de simulacion.
  std::string PartBeginDir;   //-Directorio donde busca el PART de arranque.									//-Searches directory for starting PART.
  unsigned PartBegin;         //-Indica el PART de arranque (0:Sin reanudacion).								//-Indicates the start (0: no resumption).
  unsigned PartBeginFirst;    //-Indica el numero del primer PART a generar.									//-Indicates the number of the first PART to be generated.
  double PartBeginTimeStep;   ///<Instante de inicio de la simulación.											//-initial instant of the simulation
  ullong PartBeginTotalNp;    ///<Total number of simulated particles.

  //-Vars para movimiento predefinido.
  //-Variables for predefined movement.
  JSphMotion *Motion;
  double MotionTimeMod; ///<Modificador del TimeStep para Motion.												///<Modifies the timestep for motion
  unsigned MotionObjCount,MotionObjBegin[256];

  //-Vars para floating bodies.
  //-Variables for floating bodies.
  StFloatingData *FtObjs;    ///<Data of floating object. [ftcount]
  unsigned FtCount;          ///<Number of floating objects.
  float FtPause;             ///<Time to start floating bodies movement.

  //-Vars para DEM. (DEM)
  //-Variabless for DEM (DEM)
  bool UseDEM;
  double DemDtForce;  //-Dt for tangencial acceleration.
  static const unsigned DemObjsSize=CODE_TYPE_FLUID;
  StDemData DemObjs[DemObjsSize];    ///<Data of floating object.   

  JWaveGen *WaveGen;      //-Objecto para generacion de oleaje.													//-Object for wave generation

  JSphVarAcc *VarAcc;     ///<Object for variable acceleration functionality.

  TpCellOrder CellOrder;  //-Orden de ejes en ordenacion de particulas en celdas.								//-Defines axes' ordination of particles in cells.

  //-Division en celdas.
  //-Division in cells.
  TpCellMode CellMode;    //-Modo de division en celdas.														//-Cell division mode.
  unsigned Hdiv;          //-Valor por el que se divide a DosH													//-Value to divide 2H
  float Scell;            //-Tamaño de celda: 2h o h.															//-Cell size: 2h or h.
  float MovLimit;         //-Distancia maxima que se permite recorrer a una particula en un paso (Scell*0.9).	//-Maximum distance a particle is allowed to move in one step (Sce;;*0.9)

  //-Dominio global de la simulacion.
  //-Global domain of the simulation.
  tdouble3 Map_PosMin,Map_PosMax,Map_Size; //-Limites de simulacion + borde 2h si hay condiciones periodicas.				//-Limits of simulation + edge 2h if periodic conditions.
  tuint3 Map_Cells;                        //-Numero de celdas maximo segun los limites del caso.							//-Maximum number of cells within case limits.
  tdouble3 MapRealPosMin,MapRealPosMax,MapRealSize; //-Limites de simulacion reales (sin bordes de condiciones periodicas).	//-Real limits of the simulation (without the periodic condition borders)

  //-Dominio local de la simulacion.
  //-Local domain of the simualtion.
  tuint3 DomCelIni,DomCelFin; //-Celda inicial y final dentro de Map que define el area de simulacion local.						//-First and last cell within the Map defining local simulation area.
  tuint3 DomCells;            //-Numero de celdas en cada direccion.																//-Number of cells in each direction.
  tdouble3 DomPosMin,DomPosMax,DomSize; //-Limites de simulacion + borde 2h si hay condiciones periodicas.							//-Limits of simulation + edge 2h if periodic conditions.
  tdouble3 DomRealPosMin,DomRealPosMax; //-Limites reales de simulacion segun DomCelIni/Fin (sin bordes de condiciones periodicas).	//-Real limits of the simulation according to DomCelIni/Fin (without periodic condition borders)
  unsigned DomCellCode;       //-Clave para la codificacion de la celda de posicion dentro de Domain.								//-Key for encoding cell position within the Domain.

  //-Control del numero de particulas.
  //-Controls particle number.
  bool NpDynamic;   ///<CaseNp can increase.
  bool ReuseIds;    ///<Id of particles excluded values ​​are reused.
  ullong TotalNp;   ///<Total number of simulated particles (no cuenta las particulas inlet no validas).
  unsigned IdMax;   ///<It is the maximum Id used.

  //-Monitorizacion del dt.
  //-Monitors dt.
  unsigned DtModif;    //-Numero de modificaciones del dt calculado por ser demasiado bajo.			//-Number of modifications on  dt computed when it is too low.
  double PartDtMin,PartDtMax; //-Valores minimo y maximo de dt en el PART actual.					//-Minimum and maximum values of dt in the current PART.

  //-Valores maximos (o casi) alcanzados durante la simulacion.
  //-Maximum values (or almost) achieved during the simulation.
  llong MaxMemoryCpu;  //-Cantidad de memoria Cpu reservada.			//-Amount of reserved CPU memory.
  llong MaxMemoryGpu;  //-Cantidad de memoria Gpu reservada.			//-Amount of reserved GPU memory.
  unsigned MaxParticles;  //-Numero maximo de particulas.				//-Maximum number of particles.
  unsigned MaxCells;      //-Numero maximo de celdas.					//-Maximum number of cells.

  //-Vars para simulacion de PARTs.
  //-Variables for simulation of PARTs.
  int PartIni;        //-Primer PART generado.												//-First generated PART.
  int Part;           //-Siguiente PART a guardar.											//-Saves subsequent PART.
  int Nstep;
  int PartNstep;
  unsigned PartOut;   ///<Numero total de particulas excluidas al grabar el ultimo PART.	//-Total number of excluded particles to be recorded to the last PART.
  double TimeStepIni; ///<Instante inicial de la simulación.								//-Initial instant of the simulation
  double TimeStep;    ///<Instante actual de la simulación.									//-Current instant of the simulation
  double TimeStepM1;  ///<Instante de la simulación en que se grabo el último PART.			//-Instant of the simulation when the last PART was stored.

  //-Control de tiempos de ejecucion.
  //-Control of the execution times.
  JTimer TimerTot,TimerSim,TimerPart;
  JTimersStep* TimersStep;


  void AllocMemoryFloating(unsigned ftcount);
  llong GetAllocMemoryCpu()const;

  void LoadConfig(const JCfgRun *cfg);
  void LoadCaseConfig();

  void ResetMkInfo();
  void LoadMkInfo(const JSpaceParts *parts);
  inline unsigned GetMkBlockById(unsigned id)const;
  unsigned GetMkBlockByMk(word mk)const;

  word CodeSetType(word code,TpParticle type,unsigned value)const;
  void LoadCodeParticles(unsigned np,const unsigned *idp,word *code)const;
  void ResizeMapLimits();

  void ConfigConstants(bool simulate2d);
  void VisuConfig()const;
  void LoadDcellParticles(unsigned n,const word *code,const tdouble3 *pos,unsigned *dcell)const;

  void ConfigCellOrder(TpCellOrder order,unsigned np,tdouble3* pos,tfloat4* velrhop);
  void DecodeCellOrder(unsigned np,tdouble3 *pos,tfloat3 *vel)const;
  tuint3 OrderCode(const tuint3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tfloat3 OrderCode(const tfloat3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tfloat3 OrderDecode(const tfloat3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  tdouble3 OrderCode(const tdouble3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tdouble3 OrderDecode(const tdouble3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  tuint3 OrderDecode(const tuint3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  tmatrix4d OrderCode(const tmatrix4d &v)const{ return(OrderCodeValue(CellOrder,v)); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tfloat3 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tdouble3 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tdouble3 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tfloat4 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tfloat4 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }
  void ConfigCellDivision();
  void SelecDomain(tuint3 celini,tuint3 celfin);
  static unsigned CalcCellCode(tuint3 ncells);
  void CalcFloatingRadius(unsigned np,const tdouble3 *pos,const unsigned *idp);
  tdouble3 UpdatePeriodicPos(tdouble3 ps)const;

  void PrintSizeNp(unsigned np,llong size)const;
  void PrintHeadPart();

  void ConfigSaveData(unsigned piece,unsigned pieces,std::string div);
  void AddParticlesOut(unsigned nout,const unsigned *idp,const tdouble3* pos,const tfloat3 *vel,const float *rhop,unsigned noutrhop,unsigned noutmove);

  tfloat3* GetPointerDataFloat3(unsigned n,const tdouble3* v)const;
  void SavePartData(unsigned npok,unsigned nout,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus);
  void SaveData(unsigned npok,const unsigned *idp,const tdouble3 *pos,const tfloat3 *vel,const float *rhop,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus);
  void SaveDomainVtk(unsigned ndom,const tdouble3 *vdom)const;
  void SaveMapCellsVtk(float scell)const;
  void SaveTimersStep(unsigned np,unsigned npb,unsigned npbok,unsigned nct);

  void GetResInfo(float tsim,float ttot,const std::string &headplus,const std::string &detplus,std::string &hinfo,std::string &dinfo);
  void SaveRes(float tsim,float ttot,const std::string &headplus="",const std::string &detplus="");
  void ShowResume(bool stop,float tsim,float ttot,bool all,std::string infoplus);

  unsigned GetOutPosCount()const{ return(OutPosCount); }
  unsigned GetOutRhopCount()const{ return(OutRhopCount); }
  unsigned GetOutMoveCount()const{ return(OutMoveCount); }
  void solveVienna();
public:
  JSph(bool cpu,bool withmpi);
  ~JSph();

  static std::string GetStepName(TpStep tstep);
  static std::string GetKernelName(TpKernel tkernel);
  static std::string GetViscoName(TpVisco tvisco);
  static std::string GetDeltaSphName(TpDeltaSph tdelta);
  static std::string GetShiftingName(TpShifting tshift);

  static std::string TimerToText(const std::string &name,float value);
};

/*
ES:
Consideraciones sobre condiciones periodicas:
- Para cada eje periodico se define un valor tfloat3 para sumar a las particulas
  que se salgan por el extremo superior del dominio.
- En MapPosMin/Max se el añade una holgura de H*BORDER_MAP, pero en el caso de
  condiciones periodicas esta holgura solo se aplica a MapPosMax.
- El ajuste de tamaño de dominio realizado por ResizeMapLimits() no afecta a los
  ejes periodicos.
- El CellOrder se aplica a la configuracion de condiciones periodicas.
- El halo periodico tendrá una unica celda de grosor 2h aunque en los otros ejes
  se use celdas de tamaño h.
- En la interaccion, una celda de tamaño 2h o dos celdas de tamaño h del extremo 
  inferior interaccionan con el halo periodico. En el caso del extremo superior
  deben ser 2 celdas de 2h o 3 celdas de h.

EN:
Considerations for periodic conditions:
- For each periodic edge a tfloat3 value is defined to be added to the particles
   that they get out at the limits of the domain.
- In MapPosMin/Max there is the added space of H*BORDER_MAP, but in the case of
   periodic conditions this space only applies to MapPosMax.
- The adjustment of the domain size by ResizeMapLimits() does not affect the
   periodic edges.
- The CellOrder applies to the configuration of periodic conditions.
- The periodic halo will have a single cell thick 2h although in the other axes
   h cell size is used.
- In the interaction, a cell of size 2h or two cells of size h in the
   lower end interact with the periodic halo. For the upper limit
   there must be either 2 2h cells or 3 h cells.
*/


#endif


