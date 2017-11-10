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

/// \file JObject.cpp \brief Implements the class \ref JObject.

#include "JObject.h"
#include "JException.h"

#include <cstdlib>
#include <ctime>

//==============================================================================
/// Throws simple exception.
/// \param method Name of the method that throws an exception.
/// \param msg Text of the exception.
/// \throw JException 
//==============================================================================
void JObject::RunException(const std::string &method,const std::string &msg)const{
  throw JException(ClassName,method,msg,"");
}

//==============================================================================
/// Throws exception related to a file.
/// \param method Name of the method that throws an exception.
/// \param msg Text of the exception.
/// \param file Name of the file.
/// \throw JException 
//==============================================================================
void JObject::RunException(const std::string &method,const std::string &msg,const std::string &file)const{
  throw JException(ClassName,method,msg,file);
}

//==============================================================================
/// Returns thetext of the simple exception.
/// \param method Name of the method that throws an exception.
/// \param msg Text of the exception.
//==============================================================================
std::string JObject::GetExceptionText(const std::string &method,const std::string &msg)const{
  return(JException(ClassName,method,msg,"").ToStr());
}

//==============================================================================
/// Returns thetext of the exception related to a file.
/// \param method Name of the method that throws an exception.
/// \param msg Text of the exception.
/// \param file Name of the file.
//==============================================================================
std::string JObject::GetExceptionText(const std::string &method,const std::string &msg,const std::string &file)const{
  return(JException(ClassName,method,msg,file).ToStr());
}




