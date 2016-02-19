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

#ifndef _Types_
#define _Types_

#include "TypesDef.h"
#include <algorithm>


#define HIDE_AWAS      //-Mantiene compatibilidad sin AWAS.//-Maintain compatibility without AWAS

//#define DISABLE_TIMERS           //-Compilado sin timers. //-Compiles without timers

#define CELLDIV_OVERMEMORYNP 0.05f  //-Mermoria que se reserva de mas para la gestion de particulas en JCellDivGpu. //-Memory that is reserved for the particle management in JCellDivGpu.
#define CELLDIV_OVERMEMORYCELLS 1   //-Numero celdas que se incrementa en cada dimension al reservar memoria para celdas en JCellDivGpu. //-Number of cells in each dimension is increased to allocate memory for JCellDivGpu cells.
#define PERIODIC_OVERMEMORYNP 0.05f //-Mermoria que se reserva de mas para la creacion de particulas periodicas en JSphGpuSingle::RunPeriodic(). //-Memory reserved for the creation of periodic particles in JSphGpuSingle::RunPeriodic().

#define _WITHOMP //-Activar/desactivar tb en Props config -> C/C++ -> Lenguaje -> OpenMp //-Enable/Disable tb in Props config-> C/C++ -> Language -> OpenMP

//#define _WITHGPU 1 //<-Esta definida en las propiedades del proyecto. //<-Is defined in the project properties.

#define MAXTHREADS_OMP 64
#define STRIDE_OMP 200
#define LIMIT_COMPUTESTEP_OMP 25000
#define LIMIT_COMPUTEMEDIUM_OMP 10000
#define LIMIT_COMPUTELIGHT_OMP 100000
//#define LIMIT_COMPUTEHEAVY_OMP 20000
#define LIMIT_PREINTERACTION_OMP 100000

#define BORDER_MAP 0.001

#define ALMOSTZERO 1e-18f


//-Codigos para particulas:
#define CODE_MASKSPECIAL 0xe000   //-Bits de special:     1110 0000 0000 0000							//-Special bits: 1110 0000 0000 0000
#define CODE_NORMAL   0x0         //-0  Particulas normales no excluidas.								//-0 Normal particles (not excluded)
#define CODE_PERIODIC 0x2000      //-1  Particulas duplicadas por periodicas.							//-1 Duplicate particles for periodic
#define CODE_OUTIGNORE 0x4000     //-2  Marca particulas que se van a ignorar en el siguiente divide.	//-2 Brands particles to be ignored in the next division.
#define CODE_OUTMOVE 0x6000       //-3  Particulas normales excluidas por movimiento.					//-3 Normal particles excluded for motion
#define CODE_OUTPOS 0x8000        //-4  Particulas normales excluidas por posicion.						//-4 Normal particles excluded for position
#define CODE_OUTRHOP 0xA000       //-5  Particulas normales excluidas por densidad.						//-5 Normal particles excluded for density
//#define CODE_SPECIAL1 0xC000   //-6  Por ejemplo, CODE_DOMAINPREV pertenece a Proceso-1				//-6 For example, CODE_DOMAINPREV belongs to Process-1
//#define CODE_SPECIAL2 0xE000   //-7  Por ejemplo, CODE_DOMAINNEXT pertenece a Proceso+1				//-7 For example, CODE_DOMAINNEXT belongs to Process+1

#define CODE_MASKTYPEVALUE 0x1fff //-Bits de type:    0001 1111 1111 1111								//-Bits for type:    0001 1111 1111 1111
#define CODE_MASKTYPE 0x1800      //-Bits de type:    0001 1000 0000 0000								//-Bits for type:    0001 1000 0000 0000
#define CODE_TYPE_FIXED 0x0       //---Particles fixed:  0-2047											
#define CODE_TYPE_MOVING 0x800    //---Particles moving: 2048-4095										
#define CODE_TYPE_FLOATING 0x1000 //---Particles float:  4096-6143										
#define CODE_TYPE_FLUID 0x1800    //---Particles fluid:  6144-8191										
#define CODE_MASKVALUE 0x7ff      //-Bits type-value: 0000 0111 1111 1111  Range:0-2047

#define CODE_SetNormal(code)    (code&(~CODE_MASKSPECIAL))
#define CODE_SetPeriodic(code)  (CODE_SetNormal(code)|CODE_PERIODIC)
#define CODE_SetOutIgnore(code) (CODE_SetNormal(code)|CODE_OUTIGNORE)
#define CODE_SetOutPos(code)    (CODE_SetNormal(code)|CODE_OUTPOS)
#define CODE_SetOutMove(code)   (CODE_SetNormal(code)|CODE_OUTMOVE)
#define CODE_SetOutRhop(code)   (CODE_SetNormal(code)|CODE_OUTRHOP)
#define CODE_GetSpecialValue(code) (code&CODE_MASKSPECIAL)

#define CODE_GetType(code) (code&CODE_MASKTYPE)
#define CODE_GetTypeValue(code) (code&CODE_MASKVALUE)
#define CODE_GetTypeAndValue(code) (code&CODE_MASKTYPEVALUE)

/// Structure with the information of the floating object.
typedef struct{
  word mkbound;     ///<MkBound of the floating object.
  unsigned begin;   ///<First particle of the floating object.
  unsigned count;   ///<Number of objects.
  float mass;       ///<Mass of the object.
  float massp;      ///<Mass of the particle of the floating object.
  float radius;     ///<Maximum distance between particles and center.
  tdouble3 center;  ///<Center of the object.
  tfloat3 fvel;     ///<Linear velocity of the object.
  tfloat3 fomega;   ///<Angular velocity of the object
}StFloatingData;

/// Structure with the information of the floating object in forces calculation.
typedef struct{
  tfloat3 face;      //-Sumatorio de ace de particulas. //-Sum of particle acceleration.
  tfloat3 fomegavel; //-Sumatorio de ace de particulas combinado con la distancia al centro. //-Sum of particle acceleration combined with distance from the center.
}StFtoForces;

/// Structure with the information of the floating object for DEM interaction.
typedef struct{ //(DEM)
  float mass;          ///<Mass of the object.
  float massp;         ///<Mass of the particle of the floating object.
  float young;         ///<Young Modulus of the floating object.
  float poisson;       ///<Poisson coefficient of the floating object.
  float kfric;         ///<Kinetic friction coefficient of the floating object.
  float tau;           ///<Value of (1-poisson^2)/young
  float restitu;       ///<Restitution Coefficient.
}StDemData;

///Controls the output of information on the screen and/or log.
typedef enum{ 
    MOUT_ScrFile=3,          ///<Output on the screen and log.
    MOUT_File=2,             ///<Output in the log.
    MOUT_Screen=1,           ///<Output on the screen.
    MOUT_None=0              ///<No output.
}TpModeOut;   
 
typedef enum{ SDAT_Binx=1,SDAT_Vtk=2,SDAT_Csv=4,SDAT_Info=8,SDAT_None=0 }TpSaveDat; //-Opciones de salida de datos. //-Data output options.

///Types of step algorithm.
typedef enum{ 
  STEP_Symplectic=2,         ///<Symplectic algorithm.
  STEP_Verlet=1,             ///<Verlet algorithm.
  STEP_None=0 
}TpStep;                    

///Types of kernel function.
typedef enum{ 
  KERNEL_Wendland=1,         ///<Wendland kernel.
  KERNEL_None=0 
}TpKernel;                  

///Types of viscosity treatment.
typedef enum{ 
  VISCO_LaminarSPS=2,        ///<Laminar viscosity and Sub-Partice Scale Turbulence.
  VISCO_Artificial=1,        ///<Artificial viscosity.
  VISCO_None=0 
}TpVisco;            

///Types of interaction.
typedef enum{ 
  INTER_ForcesCorr=2,        ///<Interaction to compute forces using the corrector step of Symplectic algorithm.
  INTER_Forces=1             ///<Interaction to compute forces using the Verlet algorithm and the predictor step of Symplectic algorithm. 
}TpInter;   

typedef enum{ DELTA_DynamicExt=3,DELTA_Dynamic=2,DELTA_None=0 }TpDeltaSph; //-Tipos de Delta-SPH. //-Types of Delta-SPH

typedef enum{
  SHIFT_Full=3,
  SHIFT_NoFixed=2,
  SHIFT_NoBound=1,
  SHIFT_None=0
}TpShifting; //-Modos de Shifting. //-Shifting modes.

///Types of particles.
typedef enum{ 
    PART_BoundFx=1,          ///<Fixed boundary particles.
    PART_BoundMv=2,          ///<Moving boundary particles.
    PART_BoundFx_BoundMv=3,  ///<Both fixed and moving boundary particles.
    PART_BoundFt=4,          ///<Floating boundary particles.
    PART_Fluid=8,            ///<Fluid particles.
    PART_BoundFt_Fluid=12    ///<Both floating and fluid particles.
}TpParticle;

typedef enum{ FTMODE_None=0,FTMODE_Sph=1,FTMODE_Dem=2 }TpFtMode;  //-Modo de interaccion para Floatings. //-Interaction mode for floatings.
#define USE_FLOATING (ftmode!=FTMODE_None)
#define USE_NOFLOATING (ftmode==FTMODE_None)
#define USE_DEM (ftmode==FTMODE_Dem)

///Order of the axis to reorder particles in cells.
typedef enum{ 
    ORDER_None=0,
    ORDER_XYZ=1,
    ORDER_XZY=2,
    ORDER_YXZ=3,
    ORDER_YZX=4,
    ORDER_ZXY=5,
    ORDER_ZYX=6 
}TpCellOrder;  

///Devuelve el nombre de CellOrder en texto. 
///Returns the name of CellOrder in text.
inline const char* GetNameCellOrder(TpCellOrder cellorder){
  switch(cellorder){
    case ORDER_XYZ:   return("XYZ");
    case ORDER_XZY:   return("XZY");
    case ORDER_YXZ:   return("YXZ");
    case ORDER_YZX:   return("YZX");
    case ORDER_ZXY:   return("ZXY");
    case ORDER_ZYX:   return("ZYX");
  }
  return("???");
}

///Devuelve el nombre de CellOrder en texto.
///Returns the name of CellOrder in text.
inline tuint3 GetCodeCellOrder(TpCellOrder cellorder){
  switch(cellorder){
    case ORDER_XYZ:   return(TUint3(1,2,3));
    case ORDER_XZY:   return(TUint3(1,3,2));
    case ORDER_YXZ:   return(TUint3(2,1,3));
    case ORDER_YZX:   return(TUint3(2,3,1));
    case ORDER_ZXY:   return(TUint3(3,1,2));
    case ORDER_ZYX:   return(TUint3(3,2,1));
  }
  return(TUint3(1,2,3));
}

///Devuelve el nombre de CellOrder en texto.
///Returns the name of CellOrder in text.
inline TpCellOrder GetDecodeOrder(TpCellOrder order){
  switch(order){
    case ORDER_XYZ:   return(ORDER_XYZ);
    case ORDER_XZY:   return(ORDER_XZY);
    case ORDER_YXZ:   return(ORDER_YXZ);
    case ORDER_YZX:   return(ORDER_ZXY);
    case ORDER_ZXY:   return(ORDER_YZX);
    case ORDER_ZYX:   return(ORDER_ZYX);
  }
  return(ORDER_None);
}

///Devuelve valor tfloat3 reordenado.
///Returns reordered tfloat3 value.
inline tfloat3 ReOrderXZY(const tfloat3 &v){ return(TFloat3(v.x,v.z,v.y)); }
inline tfloat3 ReOrderYXZ(const tfloat3 &v){ return(TFloat3(v.y,v.x,v.z)); }
inline tfloat3 ReOrderYZX(const tfloat3 &v){ return(TFloat3(v.y,v.z,v.x)); }
inline tfloat3 ReOrderZXY(const tfloat3 &v){ return(TFloat3(v.z,v.x,v.y)); }
inline tfloat3 ReOrderZYX(const tfloat3 &v){ return(TFloat3(v.z,v.y,v.x)); }

///Devuelve valor tdouble3 reordenado.
///Returns reordered tdouble3 value.
inline tdouble3 ReOrderXZY(const tdouble3 &v){ return(TDouble3(v.x,v.z,v.y)); }
inline tdouble3 ReOrderYXZ(const tdouble3 &v){ return(TDouble3(v.y,v.x,v.z)); }
inline tdouble3 ReOrderYZX(const tdouble3 &v){ return(TDouble3(v.y,v.z,v.x)); }
inline tdouble3 ReOrderZXY(const tdouble3 &v){ return(TDouble3(v.z,v.x,v.y)); }
inline tdouble3 ReOrderZYX(const tdouble3 &v){ return(TDouble3(v.z,v.y,v.x)); }

///Devuelve valor tuint3 reordenado.
///Returns reordered tuint3 value.
inline tuint3 ReOrderXZY(const tuint3 &v){ return(TUint3(v.x,v.z,v.y)); }
inline tuint3 ReOrderYXZ(const tuint3 &v){ return(TUint3(v.y,v.x,v.z)); }
inline tuint3 ReOrderYZX(const tuint3 &v){ return(TUint3(v.y,v.z,v.x)); }
inline tuint3 ReOrderZXY(const tuint3 &v){ return(TUint3(v.z,v.x,v.y)); }
inline tuint3 ReOrderZYX(const tuint3 &v){ return(TUint3(v.z,v.y,v.x)); }

///Devuelve valor tfloat4 reordenado.
///Returns reordered tfloat4 value.
inline tfloat4 ReOrderXZY(const tfloat4 &v){ return(TFloat4(v.x,v.z,v.y,v.w)); }
inline tfloat4 ReOrderYXZ(const tfloat4 &v){ return(TFloat4(v.y,v.x,v.z,v.w)); }
inline tfloat4 ReOrderYZX(const tfloat4 &v){ return(TFloat4(v.y,v.z,v.x,v.w)); }
inline tfloat4 ReOrderZXY(const tfloat4 &v){ return(TFloat4(v.z,v.x,v.y,v.w)); }
inline tfloat4 ReOrderZYX(const tfloat4 &v){ return(TFloat4(v.z,v.y,v.x,v.w)); }

///Reordena matriz tmatrix4f.
///Reorders tmatrix4f matrix.
inline void ReOrderXZY(tmatrix4d &x){ std::swap(x.a12,x.a13); std::swap(x.a21,x.a31); std::swap(x.a22,x.a33); std::swap(x.a23,x.a32); std::swap(x.a24,x.a34); }
inline void ReOrderYXZ(tmatrix4d &x){ std::swap(x.a11,x.a21); std::swap(x.a12,x.a22); std::swap(x.a13,x.a23); std::swap(x.a14,x.a24); std::swap(x.a11,x.a12); std::swap(x.a21,x.a22); std::swap(x.a31,x.a32); }
inline void ReOrderYZX(tmatrix4d &x){ ReOrderYXZ(x); ReOrderXZY(x); }
inline void ReOrderZXY(tmatrix4d &x){ ReOrderXZY(x); ReOrderYXZ(x); }
inline void ReOrderZYX(tmatrix4d &x){ std::swap(x.a11,x.a31); std::swap(x.a12,x.a32); std::swap(x.a13,x.a33); std::swap(x.a14,x.a34); std::swap(x.a11,x.a13); std::swap(x.a21,x.a23); std::swap(x.a31,x.a33); }

///Devuelve valor tfloat3 reordenado.
///Returns reordered tfloat3 value.
inline tfloat3 OrderCodeValue(TpCellOrder order,const tfloat3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Devuelve valor tfloat3 en el orden original.
///Returns tfloat3 value in the original order.
inline tfloat3 OrderDecodeValue(TpCellOrder order,const tfloat3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Devuelve valor tdouble3 reordenado.
///Returns reordered tdouble3 value.
inline tdouble3 OrderCodeValue(TpCellOrder order,const tdouble3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Devuelve valor tdouble3 en el orden original.
///Returns tdouble3 value in the original order.
inline tdouble3 OrderDecodeValue(TpCellOrder order,const tdouble3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Devuelve valor tuint3 reordenado.
///Returns reordered tuint3 value.
inline tuint3 OrderCodeValue(TpCellOrder order,const tuint3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Devuelve valor tuint3 en el orden original.
///Returns tuint3 value in the original order.
inline tuint3 OrderDecodeValue(TpCellOrder order,const tuint3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Devuelve matriz tmatrix4d reordenada.
///Returns reordered matrix tmatrix4d.
inline tmatrix4d OrderCodeValue(TpCellOrder order,tmatrix4d x){
  switch(order){
    case ORDER_XZY:   ReOrderXZY(x);   break;
    case ORDER_YXZ:   ReOrderYXZ(x);   break;
    case ORDER_YZX:   ReOrderYZX(x);   break;
    case ORDER_ZXY:   ReOrderZXY(x);   break;
    case ORDER_ZYX:   ReOrderZYX(x);   break;
  }
  return(x);
} 

//-Modo de division en celdas.
//-Cell division mode.
typedef enum{ 
   CELLMODE_None=0
  ,CELLMODE_2H=1      //-Divide en celdas de tamaño 2h. //-Cell division of size 2h.
  ,CELLMODE_H=2       //-Divide en celdas de tamaño h.	//-Cell division of size h.
}TpCellMode; 

///Devuelve el nombre de CellMode en texto.
///returns the name of CellMode in text.
inline const char* GetNameCellMode(TpCellMode cellmode){
  switch(cellmode){
    case CELLMODE_2H:      return("2H");
    case CELLMODE_H:       return("H");
  }
  return("???");
}

///Codificacion de celdas para posicion.
///Codification of cells for position.
#define PC__CodeOut 0xffffffff
#define PC__GetCode(sx,sy,sz) ((sx<<25)|(sy<<20)|(sz<<15)|((sy+sz)<<10)|((sx+sz)<<5)|(sx+sy))  //-Clave de codificacion (orden de valores: sx,sy,sz,sy+sz,sx+sz,sx+sy). //-Encryption key (order of values: sx,sy,sz,sy+sz,sz+sx,sx+sy).
#define PC__GetSx(cc) (cc>>25)       //-Numero de bits para coordenada X de celda. //-Number of bits for X coordinate cell.
#define PC__GetSy(cc) ((cc>>20)&31)  //-Numero de bits para coordenada Y de celda. //-Number of bits for Y coordinate cell.
#define PC__GetSz(cc) ((cc>>15)&31)  //-Numero de bits para coordenada Z de celda. //-Number of bits for Z coordinate cell.
#define PC__Cellx(cc,cel) ((*((unsigned*)&cel))>>((cc>>10)&31))             //-Coordenada X de celda. //-X coordinate of the cell.
#define PC__Celly(cc,cel) (((*((unsigned*)&cel))<<(cc>>25))>>((cc>>5)&31))  //-Coordenada Y de celda. //-Y coordinate of the cell.
#define PC__Cellz(cc,cel) (((*((unsigned*)&cel))<<(cc&31))>>(cc&31))        //-Coordenada Z de celda. //-Z coordinate of the cell.
#define PC__Cell(cc,cx,cy,cz) ((cx<<((cc>>10)&31))|(cy<<((cc>>15)&31))|cz)  //-Valor de celda para cx, cy y cz. //-Cell value for cx,cy and cz.
#define PC__MaxCellx(cc) (0xffffffff>>((cc>>10)&31))                //-Coordenada X de celda maxima. //-Maximum X coordinate of the cell.
#define PC__MaxCelly(cc) ((0xffffffff<<(cc>>25))>>((cc>>5)&31))     //-Coordenada Y de celda maxima. //-Maximum Y coordinate of the cell.
#define PC__MaxCellz(cc) ((0xffffffff<<(cc&31))>>(cc&31))           //-Coordenada Z de celda maxima. //-Maximum Z coordinate of the cell.

#endif


