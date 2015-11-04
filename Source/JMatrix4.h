/*
 <DualSPHysics4 codes>  Copyright (c) 2015 by Dr. Jose M. Dominguez, Dr. Alejandro Crespo, Prof. M. Gomez Gesteira, Dr. Anxo Barreiro, Dr. Benedict Rogers
 All rights reserved.

 DualSPHysics is an international collaboration between:
 - EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 - School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
 the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the distribution.
 * Neither the name of the DualSPHysics nor the names of its contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, 
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
 SHALL THE COPYRIGHT OWNER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//#############################################################################
//# Cambios:
//# =========
//# - Error corregido: En el metodo rotate() de la clase JMatrix4 se cambio 
//#   MulPre(MatrixRot(...)) por Mul(MatrixRot(...)) para que funcionase bien 
//#   la combinaci�n de movimiento con rotaci�n.  (19-10-2010)
//# - Traduccion de comentarios al ingles. (10/02/2012)
//# - Metodo GetMotion() para obtener angulos con respecto a ejes y traslacion
//#   a partir de la matriz de transformacion. (04/07/2014)
//#############################################################################

/// \file JMatrix4.h \brief Declares the template \ref JMatrix4
/// and the classes \ref JMatrix4f and \ref JMatrix4d

#ifndef _JMatrix4_
#define _JMatrix4_

#include "TypesDef.h"
#include <cstdio>
#include <cmath>

//==============================================================================
//##############################################################################
//==============================================================================
/// \brief Template for a matrix 4x4
/// used for geometric transformation of points in space.
template <class T,class T3,class TMAT> class JMatrix4
{
private:
  T a11,a12,a13,a14;
  T a21,a22,a23,a24;
  T a31,a32,a33,a34;
  T a41,a42,a43,a44;

public:
//==============================================================================
/// Construtor of objects.
//==============================================================================
  JMatrix4(){ SetIdentity(); }

//==============================================================================
/// Construtor of objects.
//==============================================================================
  //JMatrix4(const JMatrix4 &m){ 
  //  //printf("***copia***\n");
  //  a11=m.a11;  a12=m.a12;  a13=m.a13;  a14=m.a14;
  //  a21=m.a21;  a22=m.a22;  a23=m.a23;  a24=m.a24;
  //  a31=m.a31;  a32=m.a32;  a33=m.a33;  a34=m.a34;
  //  a41=m.a41;  a42=m.a42;  a43=m.a43;  a44=m.a44;
  //}

//==============================================================================
/// Construtor of objects.
/// \param m structure of type matrix 4x4.
//==============================================================================
  JMatrix4(TMAT m){ 
    a11=m.a11;  a12=m.a12;  a13=m.a13;  a14=m.a14;
    a21=m.a21;  a22=m.a22;  a23=m.a23;  a24=m.a24;
    a31=m.a31;  a32=m.a32;  a33=m.a33;  a34=m.a34;
    a41=m.a41;  a42=m.a42;  a43=m.a43;  a44=m.a44;
  }

//==============================================================================
/// Becomes the indentity matrix.
//==============================================================================
  void SetIdentity(){
    a11=1; a12=0; a13=0; a14=0;
    a21=0; a22=1; a23=0; a24=0;
    a31=0; a32=0; a33=1; a34=0;
    a41=0; a42=0; a43=0; a44=1;
  }

//==============================================================================
/// Adds the matrix \a m.
//==============================================================================
  void Sum(const JMatrix4 &m){
    a11+=m.a11; a12+=m.a12; a13+=m.a13; a14+=m.a14;
    a21+=m.a21; a22+=m.a22; a23+=m.a23; a24+=m.a24;
    a31+=m.a31; a32+=m.a32; a33+=m.a33; a34+=m.a34;
    a41+=m.a41; a42+=m.a42; a43+=m.a43; a44+=m.a44;
  }

//==============================================================================
/// Multiplies by the matrix \a m2.
//==============================================================================
  void Mul(const JMatrix4 &m2){
    JMatrix4 m1=*this;
    a11= m1.a11*m2.a11 + m1.a12*m2.a21 + m1.a13*m2.a31 + m1.a14*m2.a41;
    a12= m1.a11*m2.a12 + m1.a12*m2.a22 + m1.a13*m2.a32 + m1.a14*m2.a42;
    a13= m1.a11*m2.a13 + m1.a12*m2.a23 + m1.a13*m2.a33 + m1.a14*m2.a43;
    a14= m1.a11*m2.a14 + m1.a12*m2.a24 + m1.a13*m2.a34 + m1.a14*m2.a44;
    a21= m1.a21*m2.a11 + m1.a22*m2.a21 + m1.a23*m2.a31 + m1.a24*m2.a41;
    a22= m1.a21*m2.a12 + m1.a22*m2.a22 + m1.a23*m2.a32 + m1.a24*m2.a42;
    a23= m1.a21*m2.a13 + m1.a22*m2.a23 + m1.a23*m2.a33 + m1.a24*m2.a43;
    a24= m1.a21*m2.a14 + m1.a22*m2.a24 + m1.a23*m2.a34 + m1.a24*m2.a44;
    a31= m1.a31*m2.a11 + m1.a32*m2.a21 + m1.a33*m2.a31 + m1.a34*m2.a41;
    a32= m1.a31*m2.a12 + m1.a32*m2.a22 + m1.a33*m2.a32 + m1.a34*m2.a42;
    a33= m1.a31*m2.a13 + m1.a32*m2.a23 + m1.a33*m2.a33 + m1.a34*m2.a43;
    a34= m1.a31*m2.a14 + m1.a32*m2.a24 + m1.a33*m2.a34 + m1.a34*m2.a44;
    a41= m1.a41*m2.a11 + m1.a42*m2.a21 + m1.a43*m2.a31 + m1.a44*m2.a41;
    a42= m1.a41*m2.a12 + m1.a42*m2.a22 + m1.a43*m2.a32 + m1.a44*m2.a42;
    a43= m1.a41*m2.a13 + m1.a42*m2.a23 + m1.a43*m2.a33 + m1.a44*m2.a43;
    a44= m1.a41*m2.a14 + m1.a42*m2.a24 + m1.a43*m2.a34 + m1.a44*m2.a44;
  }

//==============================================================================
/// Left multiplies by the matrix \a m1.
//==============================================================================
  void MulPre(const JMatrix4 &m1){
    JMatrix4 m2=*this;
    a11= m1.a11*m2.a11 + m1.a12*m2.a21 + m1.a13*m2.a31 + m1.a14*m2.a41;
    a12= m1.a11*m2.a12 + m1.a12*m2.a22 + m1.a13*m2.a32 + m1.a14*m2.a42;
    a13= m1.a11*m2.a13 + m1.a12*m2.a23 + m1.a13*m2.a33 + m1.a14*m2.a43;
    a14= m1.a11*m2.a14 + m1.a12*m2.a24 + m1.a13*m2.a34 + m1.a14*m2.a44;
    a21= m1.a21*m2.a11 + m1.a22*m2.a21 + m1.a23*m2.a31 + m1.a24*m2.a41;
    a22= m1.a21*m2.a12 + m1.a22*m2.a22 + m1.a23*m2.a32 + m1.a24*m2.a42;
    a23= m1.a21*m2.a13 + m1.a22*m2.a23 + m1.a23*m2.a33 + m1.a24*m2.a43;
    a24= m1.a21*m2.a14 + m1.a22*m2.a24 + m1.a23*m2.a34 + m1.a24*m2.a44;
    a31= m1.a31*m2.a11 + m1.a32*m2.a21 + m1.a33*m2.a31 + m1.a34*m2.a41;
    a32= m1.a31*m2.a12 + m1.a32*m2.a22 + m1.a33*m2.a32 + m1.a34*m2.a42;
    a33= m1.a31*m2.a13 + m1.a32*m2.a23 + m1.a33*m2.a33 + m1.a34*m2.a43;
    a34= m1.a31*m2.a14 + m1.a32*m2.a24 + m1.a33*m2.a34 + m1.a34*m2.a44;
    a41= m1.a41*m2.a11 + m1.a42*m2.a21 + m1.a43*m2.a31 + m1.a44*m2.a41;
    a42= m1.a41*m2.a12 + m1.a42*m2.a22 + m1.a43*m2.a32 + m1.a44*m2.a42;
    a43= m1.a41*m2.a13 + m1.a42*m2.a23 + m1.a43*m2.a33 + m1.a44*m2.a43;
    a44= m1.a41*m2.a14 + m1.a42*m2.a24 + m1.a43*m2.a34 + m1.a44*m2.a44;
  }

//==============================================================================
/// Returns the product of the matrix by the vector \a p.
//==============================================================================
  T3 MulPoint(const T3 &p)const{
    T3 r;
    r.x= a11*p.x + a12*p.y + a13*p.z + a14;
    r.y= a21*p.x + a22*p.y + a23*p.z + a24;
    r.z= a31*p.x + a32*p.y + a33*p.z + a34;
    return(r);
  }

//==============================================================================
/// Returns the product of the matrix by the array of points.
//==============================================================================
  void MulArray(unsigned np,T3 *vp)const{
    for(unsigned c=0;c<np;c++){
      const T3 p=vp[c];
      T3 r;
      r.x= a11*p.x + a12*p.y + a13*p.z + a14;
      r.y= a21*p.x + a22*p.y + a23*p.z + a24;
      r.z= a31*p.x + a32*p.y + a33*p.z + a34;
      vp[c]=r;
    }
  }

//==============================================================================
/// Returns the product of the matrix by the array of points.
//==============================================================================
  void MulArray(unsigned np,const T3 *vp,T3 *vr)const{
    for(unsigned c=0;c<np;c++){
      const T3 p=vp[c];
      T3 r;
      r.x= a11*p.x + a12*p.y + a13*p.z + a14;
      r.y= a21*p.x + a22*p.y + a23*p.z + a24;
      r.z= a31*p.x + a32*p.y + a33*p.z + a34;
      vr[c]=r;
    }
  }

//==============================================================================
/// Linear motion is applied.
/// \param p Array with displacement in every axis.
//==============================================================================
  void Move(const T3 &p){
    Mul(MatrixMov(p));
  }

//==============================================================================
/// Rotational motion is applied over an arbitrary axis.
/// \param ang Angle of roation (in degrees).
/// \param axisp1 Initial point of the array that defines the axis of rotation.
/// \param axisp2 Final point of the array that defines the axis of rotation.
//==============================================================================
  void Rotate(T ang,const T3 &axisp1,const T3 &axisp2){
    Mul(MatrixRot(ang,axisp1,axisp2));
  }

//==============================================================================
/// Scaling is aplied.
/// \param p Array with the scale in every axis.
//==============================================================================
  void Scale(const T3 &p){
    Mul(MatrixScale(p));
  }

//==============================================================================
/// Returns a transformation matrix for a linear movement.
/// \param p Array with displacement in every axis.
//==============================================================================
  static JMatrix4 MatrixMov(const T3 &p){
    JMatrix4 m;
    m.a14=p.x; m.a24=p.y; m.a34=p.z;
    return(m);
  }

//==============================================================================
/// Returns a transformation matrix for a scaling.
/// \param p Array with displacement in every axis.
//==============================================================================
  static JMatrix4 MatrixScale(const T3 &p){
    JMatrix4 m;
    m.a11=p.x; m.a22=p.y; m.a33=p.z;
    return(m);
  }

//==============================================================================
/// Returns a transformation matrix for a rotation over an arbitrary axis.
/// \param ang Angle of roation (in degrees).
/// \param axisp1 Initial point of the array that defines the axis of rotation.
/// \param axisp2 Final point of the array that defines the axis of rotation.
//==============================================================================
  static JMatrix4 MatrixRot(T ang,const T3 &axisp1,const T3 &axisp2){
    //fflush(stdout);    printf("MatrixRot\n");

    T rad=T(ang*TORAD);   //float(ang*PI/180);

    T3 v; v.x=axisp2.x-axisp1.x; v.y=axisp2.y-axisp1.y; v.z=axisp2.z-axisp1.z;
    T L=sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
    T L1=sqrt(v.y*v.y+v.z*v.z);

    JMatrix4 t;    t.a14=axisp1.x; t.a24=axisp1.y;  t.a34=axisp1.z;
    JMatrix4 tm1;  tm1.a14=-axisp1.x; tm1.a24=-axisp1.y; tm1.a34=-axisp1.z;
    JMatrix4 rx;
    JMatrix4 rxm1;
    if(L1==0){
      rx.a22=0;    rx.a23=1;    rx.a32=-1;    rx.a33=0; 
      rxm1.a22=0;  rxm1.a23=-1; rxm1.a32=1;     rxm1.a33=0;
    }
    else{
      rx.a22=v.z/L1;    rx.a23=v.y/L1;    rx.a32=-v.y/L1;    rx.a33=v.z/L1; 
      rxm1.a22=v.z/L1;  rxm1.a23=-v.y/L1; rxm1.a32=v.y/L1;  rxm1.a33=v.z/L1;
    }
    JMatrix4 ry;   ry.a11=L1/L;      ry.a13=v.x/L;     ry.a31=-v.x/L;   ry.a33=L1/L;    
    JMatrix4 rym1; rym1.a11=L1/L;    rym1.a13=-v.x/L;  rym1.a31=v.x/L;    rym1.a33=L1/L;
    T cs=cos(rad),sn=sin(rad);
    JMatrix4 rz;   rz.a11=cs;  rz.a12=sn;  rz.a21=-sn;  rz.a22=cs;  
//printf("---> L1:%f L:%f\n",L1,L);
    t.Mul(rx);
    t.Mul(ry);
    t.Mul(rz);
    t.Mul(rym1);
    t.Mul(rxm1);
    t.Mul(tm1);
    //t.SetIdentity();
    return(t);
  }

//==============================================================================
/// Returns axis rotation and translation.
/// \param rot Angles of roation (in degrees).
/// \param mov Translation.
//==============================================================================
  void GetMotion(T3 &rot,T3 &mov)const{
    T3 pt[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    T3 pr[3]; MulArray(3,pt,pr);
    mov=pr[0];
    T3 imov={-mov.x,-mov.y,-mov.z};
    JMatrix4::MatrixMov(imov).MulArray(3,pr);

    double ppy=fabs(pr[1].y/sqrt(pr[1].x*pr[1].x+pr[1].y*pr[1].y));
    ppy=(ppy>1? 1.: ppy);
    double angz1=asin(ppy)*TODEG;
    if(pr[1].x<0)angz1=(pr[1].y>=0? angz1: -angz1)-180;
    else if(pr[1].y>=0)angz1=-angz1;
    //printf("\n ppy:%f  angz1:%f \n",ppy,angz1); // fflush(stdout);
    JMatrix4::MatrixRot(angz1,pt[3],pt[0]).MulArray(3,pr);

    double ppz=fabs(pr[1].z/sqrt(pr[1].x*pr[1].x+pr[1].z*pr[1].z));
    ppz=(ppz>1? 1.: ppz);
    double angy1=asin(ppz)*TODEG;
    if(pr[1].x<0)angy1=(pr[1].z>=0? -angy1: angy1)-180;
    else if(pr[1].z<0)angy1=-angy1;
    //printf("\n ppz:%f  angy1:%f \n",ppz,angy1); // fflush(stdout);
    JMatrix4::MatrixRot(angy1,pt[2],pt[0]).MulArray(3,pr);

    double ppz2=fabs(pr[2].z/sqrt(pr[2].y*pr[2].y+pr[2].z*pr[2].z));
    ppz2=(ppz2>1? 1.: ppz2);
    double angx1=asin(ppz2)*TODEG;
    if(pr[2].y<0)angx1=(pr[2].z>=0? angx1: -angx1)-180;
    else if(pr[2].z>=0)angx1=-angx1;
    //printf("\n ppz2:%f  angx1:%f \n",ppz2,angx1); // fflush(stdout);

    rot.x=angx1; rot.y=angy1; rot.z=angz1;
  }

//==============================================================================
/// Visualises the content of the matrix in console.
//==============================================================================
  void Print(const char* text,const char* fmt="[%8.5f,%8.5f,%8.5f,%8.5f]\n")const{
    printf("%s\n",text);
    printf(fmt,a11,a12,a13,a14);
    printf(fmt,a21,a22,a23,a24);
    printf(fmt,a31,a32,a33,a34);
    printf(fmt,a41,a42,a43,a44);
  }

//==============================================================================
/// Returns values of the matrix as a structure.
//==============================================================================
  tmatrix4f GetMatrix4f()const{ return(TMatrix4f(float(a11),float(a12),float(a13),float(a14),float(a21),float(a22),float(a23),float(a24),float(a31),float(a32),float(a33),float(a34),float(a41),float(a42),float(a43),float(a44))); }

//==============================================================================
/// Returns values of the matrix as a structure.
//==============================================================================
  tmatrix4d GetMatrix4d()const{ return(TMatrix4d(double(a11),double(a12),double(a13),double(a14),double(a21),double(a22),double(a23),double(a24),double(a31),double(a32),double(a33),double(a34),double(a41),double(a42),double(a43),double(a44))); }

//==============================================================================
/// Returns values of the matrix as a structure.
//==============================================================================
  TMAT GetMatrix()const{
    TMAT m;
    m.a11=a11; m.a12=a12; m.a13=a13; m.a14=a14;
    m.a21=a21; m.a22=a22; m.a23=a23; m.a24=a24;
    m.a31=a31; m.a32=a32; m.a33=a33; m.a34=a34;
    m.a41=a41; m.a42=a42; m.a43=a43; m.a44=a44;
    return(m);
  }

};

typedef JMatrix4<float,tfloat3,tmatrix4f> JMatrix4f;   ///<Matrix of 4x4 for values of type float.
typedef JMatrix4<double,tdouble3,tmatrix4d> JMatrix4d; ///<Matrix of 4x4 for values of type double.


#endif

