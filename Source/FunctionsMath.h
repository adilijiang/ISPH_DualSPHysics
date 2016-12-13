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


/// \file FunctionsMath.h \brief Declares basic/general math functions.

#ifndef _FunctionsMath_
#define _FunctionsMath_

#include "TypesDef.h"
#include <cstdlib>
#include <cmath>
#include <cfloat>

/// Implements a set of basic/general math functions.
namespace fmath{

//==============================================================================
/// Devuelve la interpolacion lineal de dos valores.
/// Returns the linear interpolation value.
//==============================================================================
inline double InterpolationLinear(double x,double x0,double x1,double v0,double v1){
  const double fx=(x-x0)/(x1-x0);
  return(fx*(v1-v0)+v0);
}

//==============================================================================
/// Devuelve la interpolacion bilineal de cuatro valores que forman un cuadrado.
/// Returns the bilinear interpolation of four values that form a square.
//==============================================================================
inline double InterpolationBilinear(double x,double y,double px,double py,double dx,double dy,double vxy,double vxyy,double vxxy,double vxxyy){
  double vy0=InterpolationLinear(x,px,px+dx,vxy,vxxy);
  double vy1=InterpolationLinear(x,px,px+dx,vxyy,vxxyy);
  return(InterpolationLinear(y,py,py+dy,vy0,vy1));
}


//==============================================================================
/// Devuelve el producto escalar de 2 vectores.
/// Returns the scalar product of two vectors.
//==============================================================================
inline double ProductScalar(tdouble3 v1,tdouble3 v2){
  return(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

//==============================================================================
/// Devuelve el producto escalar de 2 vectores.
/// Returns the scalar product of two vectors.
//==============================================================================
inline float ProductScalar(tfloat3 v1,tfloat3 v2){
  return(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}


//==============================================================================
/// Devuelve el producto vectorial de 2 vectores.
/// Returns the vectorial product of two vectors.
//==============================================================================
inline tdouble3 ProductVec(const tdouble3 &v1,const tdouble3 &v2){
  tdouble3 r;
  r.x=v1.y*v2.z - v1.z*v2.y;
  r.y=v1.z*v2.x - v1.x*v2.z;
  r.z=v1.x*v2.y - v1.y*v2.x;
  return(r);
}

//==============================================================================
/// Devuelve el producto vectorial de 2 vectores.
/// Returns the vectorial product of two vectors.
//==============================================================================
inline tfloat3 ProductVec(const tfloat3 &v1,const tfloat3 &v2){
  tfloat3 r;
  r.x=v1.y*v2.z - v1.z*v2.y;
  r.y=v1.z*v2.x - v1.x*v2.z;
  r.z=v1.x*v2.y - v1.y*v2.x;
  return(r);
}


//==============================================================================
/// Resuelve punto en el plano.
/// Solves point in the plane.
//==============================================================================
inline double PointPlane(const tdouble4 &pla,const tdouble3 &pt){ 
  return(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w);
}

//==============================================================================
/// Resuelve punto en el plano.
/// Solves point in the plane.
//==============================================================================
inline float PointPlane(const tfloat4 &pla,const tfloat3 &pt){ 
  return(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w);
}


//==============================================================================
/// Devuelve la distancia entre un punto y un plano.
/// Returns the distance between a point and a plane.
//==============================================================================
inline double DistPlane(const tdouble4 &pla,const tdouble3 &pt){ 
  return(fabs(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w)/sqrt(pla.x*pla.x+pla.y*pla.y+pla.z*pla.z));
}

//==============================================================================
/// Devuelve la distancia entre un punto y un plano.
/// Returns the distance between a point and a plane.
//==============================================================================
inline float DistPlane(const tfloat4 &pla,const tfloat3 &pt){ 
  return(fabs(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w)/sqrt(pla.x*pla.x+pla.y*pla.y+pla.z*pla.z));
}


//==============================================================================
/// Devuelve la distancia entre dos puntos.
/// Returns the distance between two points.
//==============================================================================
inline double DistPoints(const tdouble3 &p1,const tdouble3 &p2){
  const tdouble3 v=p1-p2;
  return(sqrt(v.x*v.x+v.y*v.y+v.z*v.z));
}

//==============================================================================
/// Devuelve la distancia entre dos puntos.
/// Returns the distance between two points.
//==============================================================================
inline float DistPoints(const tfloat3 &p1,const tfloat3 &p2){
  const tfloat3 v=p1-p2;
  return(sqrt(v.x*v.x+v.y*v.y+v.z*v.z));
}


//==============================================================================
/// Devuelve la distancia al (0,0,0).
/// Returns the distance from (0,0,0).
//==============================================================================
inline double DistPoint(const tdouble3 &p1){
  return(sqrt(p1.x*p1.x+p1.y*p1.y+p1.z*p1.z));
}

//==============================================================================
/// Devuelve la distancia al (0,0,0).
/// Returns the distance from (0,0,0).
//==============================================================================
inline float DistPoint(const tfloat3 &p1){
  return(sqrt(p1.x*p1.x+p1.y*p1.y+p1.z*p1.z));
}


//==============================================================================
/// Devuelve vector unitario del vector (0,0,0)->p1.
/// Returns a unit vector of the vector (0,0,0)->p1.
//==============================================================================
inline tdouble3 VecUnitary(const tdouble3 &p1){
  return(p1/TDouble3(DistPoint(p1)));
}

//==============================================================================
/// Devuelve vector unitario del vector (0,0,0)->p1.
/// Returns a unit vector of the vector (0,0,0)->p1.
//==============================================================================
inline tfloat3 VecUnitary(const tfloat3 &p1){
  return(p1/TFloat3(DistPoint(p1)));
}

//==============================================================================
/// Devuelve la normal de un triangulo.
/// Returns the normal of a triangle.
//==============================================================================
inline tdouble3 NormalTriangle(const tdouble3& p1,const tdouble3& p2,const tdouble3& p3){
  return(ProductVec(p1-p2,p2-p3));
}

//==============================================================================
/// Devuelve la normal de un triangulo.
/// Returns the normal of a triangle.
//==============================================================================
inline tfloat3 NormalTriangle(const tfloat3& p1,const tfloat3& p2,const tfloat3& p3){
  return(ProductVec(p1-p2,p2-p3));
}


//==============================================================================
/// Calcula el determinante de una matriz de 3x3.
/// Returns the determinant of a 3x3 matrix.
//==============================================================================
inline double Determinant3x3(const tmatrix3d &d){
  return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
}

//==============================================================================
/// Calcula el determinante de una matriz de 3x3.
/// Returns the determinant of a 3x3 matrix.
//==============================================================================
inline float Determinant3x3(const tmatrix3f &d){
  return(d.a11 * d.a22 * d.a33 + d.a12 * d.a23 * d.a31 + d.a13 * d.a21 * d.a32 - d.a31 * d.a22 * d.a13 - d.a32 * d.a23 * d.a11 - d.a33 * d.a21 * d.a12);
}


//==============================================================================
/// Devuelve proyeccion ortogonal del punto en el plano.
/// Returns orthogonal projection of the point in the plane.
//==============================================================================
inline tdouble3 PtOrthogonal(const tdouble3 &pt,const tdouble4 &pla){
  const double t=-(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w)/(pla.x*pla.x+pla.y*pla.y+pla.z*pla.z);
  return(TDouble3(pt.x+pla.x*t,pt.y+pla.y*t,pt.z+pla.z*t));
}

//==============================================================================
/// Devuelve proyeccion ortogonal del punto en el plano.
/// Returns orthogonal projection of the point in the plane.
//==============================================================================
inline tfloat3 PtOrthogonal(const tfloat3 &pt,const tfloat4 &pla){
  const float t=-(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w)/(pla.x*pla.x+pla.y*pla.y+pla.z*pla.z);
  return(TFloat3(pt.x+pla.x*t,pt.y+pla.y*t,pt.z+pla.z*t));
}


//==============================================================================
/// Devuelve el plano formado por 3 puntos.
/// Returns the plane defined by 3 points.
//==============================================================================
tdouble4 Plane3Pt(const tdouble3 &p1,const tdouble3 &p2,const tdouble3 &p3);

//==============================================================================
/// Devuelve el plano formado por 3 puntos.
/// Returns the plane defined by 3 points.
//==============================================================================
tfloat4 Plane3Pt(const tfloat3 &p1,const tfloat3 &p2,const tfloat3 &p3);


//==============================================================================
/// Devuelve el plano formado por un punto y un vector.
/// Returns the plane defined by a point and a vector.
//==============================================================================
inline tdouble4 PlanePtVec(const tdouble3 &pt,const tdouble3 &vec){
  const tdouble3 v=VecUnitary(vec);//-No es necesario pero asi el modulo del vector no afecta al resultado de PointPlane().
  return(TDouble4(v.x,v.y,v.z,-v.x*pt.x-v.y*pt.y-v.z*pt.z));
}

//==============================================================================
/// Devuelve el plano formado por un punto y un vector.
/// Returns the plane defined by a point and a vector.
//==============================================================================
inline tfloat4 PlanePtVec(const tfloat3 &pt,const tfloat3 &vec){
  const tfloat3 v=VecUnitary(vec);//-No es necesario pero asi el modulo del vector no afecta al resultado de PointPlane().
  return(TFloat4(v.x,v.y,v.z,-v.x*pt.x-v.y*pt.y-v.z*pt.z));
}


//==============================================================================
/// Devuelve los tres planos normales que limitan un triangulo formado por 3 puntos.
/// Con openingdist puedes abrir o cerrar los planos normales.
/// Returns the three normal planes which bound a triangle formed by 3 points.
/// With openingdist you can open or close normal planes.
//==============================================================================
void NormalPlanes3Pt(const tdouble3 &p1,const tdouble3 &p2,const tdouble3 &p3,double openingdist,tdouble4 &pla1,tdouble4 &pla2,tdouble4 &pla3);

//==============================================================================
/// Devuelve los tres planos normales que limitan un triangulo formado por 3 puntos.
/// Con openingdist puedes abrir o cerrar los planos normales.
/// Los calculos internos se hacen con double precision.
/// Returns the three normal planes which bound a triangle formed by 3 points.
/// With openingdist you can open or close normal levels.
/// The internal computation is performed with double precision.
//==============================================================================
inline void NormalPlanes3Pt_dbl(const tfloat3 &p1,const tfloat3 &p2,const tfloat3 &p3,float openingdist,tfloat4 &pla1,tfloat4 &pla2,tfloat4 &pla3){
  tdouble4 plad1,plad2,plad3;
  NormalPlanes3Pt(ToTDouble3(p1),ToTDouble3(p2),ToTDouble3(p3),double(openingdist),plad1,plad2,plad3);
  pla1=ToTFloat4(plad1); pla2=ToTFloat4(plad2); pla3=ToTFloat4(plad3);
}

//==============================================================================
/// Devuelve los tres planos normales que limitan un triangulo formado por 3 puntos.
/// Con openingdist puedes abrir o cerrar los planos normales.
/// Returns the three normal planes which bound a triangle formed by 3 points.
/// With openingdist you can open or close normal planes.
//==============================================================================
void NormalPlanes3Pt(const tfloat3 &p1,const tfloat3 &p2,const tfloat3 &p3,float openingdist,tfloat4 &pla1,tfloat4 &pla2,tfloat4 &pla3);


//==============================================================================
/// Devuelve punto de interseccion entre 3 planos no paralelos entre si.
/// Returns intersection of three planes not parallel to each other.
//==============================================================================
tdouble3 Intersec3Planes(const tdouble4 &pla1,const tdouble4 &pla2,const tdouble4 &pla3);

//==============================================================================
/// Devuelve punto de interseccion entre 3 planos no paralelos entre si.
/// Returns intersection of three planes not parallel to each other.
//==============================================================================
tfloat3 Intersec3Planes(const tfloat4 &pla1,const tfloat4 &pla2,const tfloat4 &pla3);


//==============================================================================
/// A partir de un triangulo formado por 3 puntos devuelve los puntos que forman
/// un triangulo mas o menos abierto segun openingdist.
/// Starting from a triangle formed by 3 points returns the points that form
/// a triangle more or less open according to openingdist.
//==============================================================================
void OpenTriangle3Pt(const tdouble3 &p1,const tdouble3 &p2,const tdouble3 &p3,double openingdist,tdouble3 &pt1,tdouble3 &pt2,tdouble3 &pt3);

//==============================================================================
/// A partir de un triangulo formado por 3 puntos devuelve los puntos que forman
/// un triangulo mas o menos abierto segun openingdist.
/// Starting from a triangle formed by 3 points returns the points that form
/// a triangle more or less open according to openingdist.
//==============================================================================
void OpenTriangle3Pt(const tfloat3 &p1,const tfloat3 &p2,const tfloat3 &p3,float openingdist,tfloat3 &pt1,tfloat3 &pt2,tfloat3 &pt3);

//==============================================================================
/// Devuelve el area de un triangulo formado por 3 puntos.
/// Returns the area of a triangle formed by 3 points.
//==============================================================================
double AreaTriangle(const tdouble3 &p1,const tdouble3 &p2,const tdouble3 &p3);

//==============================================================================
/// Devuelve el area de un triangulo formado por 3 puntos.
/// Returns the area of a triangle formed by 3 points.
//==============================================================================
float AreaTriangle(const tfloat3 &p1,const tfloat3 &p2,const tfloat3 &p3);


//==============================================================================
/// Devuelve la distancia entre un punto y una recta entre dos puntos.
/// Returns the distance between a point and a line between two points.
//==============================================================================
inline double DistLine(const tdouble3 &pt,const tdouble3 &pr1,const tdouble3 &pr2){
  double ar=AreaTriangle(pt,pr1,pr2);
  double dis=DistPoints(pr1,pr2);
  return((ar*2)/dis);
}

//==============================================================================
/// Devuelve la distancia entre un punto y una recta entre dos puntos.
/// Returns the distance between a point and a line between two points.
//==============================================================================
inline float DistLine(const tfloat3 &pt,const tfloat3 &pr1,const tfloat3 &pr2){
  float ar=AreaTriangle(pt,pr1,pr2);
  float dis=DistPoints(pr1,pr2);
  return((ar*2)/dis);
}


//==============================================================================
/// Devuelve el angulo en grados que forman dos vectores.
/// Returns angle in degrees between two vectors.
//==============================================================================
inline double AngleVector(const tdouble3 &v1,const tdouble3 &v2){
  return(acos(ProductScalar(v1,v2)/(DistPoint(v1)*DistPoint(v2)))*TODEG);
}

//==============================================================================
/// Devuelve el angulo en grados que forman dos vectores.
/// Returns angle in degrees between two vectors.
//==============================================================================
inline float AngleVector(const tfloat3 &v1,const tfloat3 &v2){
  return(float(acos(ProductScalar(v1,v2)/(DistPoint(v1)*DistPoint(v2)))*TODEG));
}


//==============================================================================
/// Devuelve el angulo en grados que forman dos planos.
/// Returns angle in degrees between two planes.
//==============================================================================
inline double AnglePlanes(tdouble4 v1,tdouble4 v2){
  return(AngleVector(TDouble3(v1.x,v1.y,v1.z),TDouble3(v2.x,v2.y,v2.z)));
}

//==============================================================================
/// Devuelve el angulo en grados que forman dos planos.
/// Returns angle in degrees between two planes.
//==============================================================================
inline float AnglePlanes(tfloat4 v1,tfloat4 v2){
  return(AngleVector(TFloat3(v1.x,v1.y,v1.z),TFloat3(v2.x,v2.y,v2.z)));
}


//==============================================================================
/// Devuelve normal eliminando error de precision en double.
/// Returns normal removing the error of precision in double.
//==============================================================================
inline tdouble3 CorrectNormal(tdouble3 n){
  if(abs(n.x)<DBL_EPSILON*10)n.x=0;
  if(abs(n.y)<DBL_EPSILON*10)n.y=0;
  if(abs(n.z)<DBL_EPSILON*10)n.z=0;
  return(VecUnitary(n));
}

//==============================================================================
/// Devuelve normal eliminando error de precision en float.
/// Returns normal removing the error of precision in float.
//==============================================================================
inline tfloat3 CorrectNormal(tfloat3 n){
  if(abs(n.x)<FLT_EPSILON*10)n.x=0;
  if(abs(n.y)<FLT_EPSILON*10)n.y=0;
  if(abs(n.z)<FLT_EPSILON*10)n.z=0;
  return(VecUnitary(n));
}


//==============================================================================
/// Returns cotangent of angle in radians.
//==============================================================================
inline double cot(double z){ return(1.0 / tan(z)); }

//==============================================================================
/// Returns hyperbolic cotangent of angle in radians.
//==============================================================================
inline double coth(double z){ return(cosh(z) / sinh(z)); }

//==============================================================================
/// Returns secant of angle in radians.
//==============================================================================
inline double sec(double z){ return(1.0 / cos(z)); }

//==============================================================================
/// Returns cosecant of input angle in radians.
//==============================================================================
inline double csc(double z){ return(1.0 / sin(z)); }

}

#endif




