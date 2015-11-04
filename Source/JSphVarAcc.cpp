/*
<DUALSPHYSICS>  Copyright (C) 2013 by Jose M. Dominguez, Dr Alejandro Crespo, Prof. M. Gomez Gesteira, Anxo Barreiro, Ricardo Canelas
                                      Dr Benedict Rogers, Dr Stephen Longshaw, Dr Renato Vacondio

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics. 

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphVarAcc.cpp \brief Implements the class \ref JSphVarAcc.

#include "JSphVarAcc.h"
#include "Functions.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <float.h>

using std::string;
using std::ios;
using std::ifstream;
using std::stringstream;

//==============================================================================
/// Constructor.
//==============================================================================
JSphVarAccFile::JSphVarAccFile(){
  ClassName="JSphVarAccFile";
  AccTime=NULL;
  AccLin=NULL;
  AccAng=NULL;
  VelLin=NULL; //SL: New linear velocity variable
  VelAng=NULL; //SL: New angular velocity variable
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphVarAccFile::~JSphVarAccFile(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphVarAccFile::Reset(){
  MkFluid=0;
  AccCoG=TFloat3(0);
  AccSize=AccCount=0;
  delete[] AccTime;  AccTime=NULL;
  delete[] AccLin;   AccLin=NULL;
  delete[] AccAng;   AccAng=NULL;
  delete[] VelLin;   VelLin=NULL; //SL: New linear velocity variable
  delete[] VelAng;   VelAng=NULL; //SL: New angular velocity variable
  CurrAccLin=CurrAccAng=TDouble3(0);
  AccIndex=0;
}

//==============================================================================
/// Resizes memory space for values.
//==============================================================================
void JSphVarAccFile::Resize(unsigned size){
  if(size>SIZEMAX)size=SIZEMAX;
  if(size==AccSize)RunException("Resize","It has reached the maximum size allowed.");
  AccTime=fun::ResizeAlloc(AccTime,AccCount,size);
  AccLin=fun::ResizeAlloc(AccLin,AccCount,size);
  AccAng=fun::ResizeAlloc(AccAng,AccCount,size);
  VelLin=fun::ResizeAlloc(VelLin,AccCount,size); //SL: New linear velocity variable
  VelAng=fun::ResizeAlloc(VelAng,AccCount,size); //SL: New angular velocity variable
  AccSize=size;
}

//==============================================================================
/// Returns the allocated memory.
//==============================================================================
long long JSphVarAccFile::GetAllocMemory()const{
  long long s=0;
  if(AccTime)s+=sizeof(float)*AccSize;
  if(AccLin)s+=sizeof(tfloat3)*AccSize;
  if(AccAng)s+=sizeof(tfloat3)*AccSize;
  if(VelLin)s+=sizeof(tfloat3)*AccSize; //SL: New linear velocity variable
  if(VelAng)s+=sizeof(tfloat3)*AccSize; //SL: New angular velocity variable
  return(s);
}

//==============================================================================
/// Reads data from an external file to enable time-dependent acceleration of specific particles.
//==============================================================================
void JSphVarAccFile::LoadFile(std::string file,double tmax){
  const char met[]="LoadFile";
  Reset();
  float timeMax=-1.0f;
  //printf("---> LoadFile> [%s]\n",file.c_str());
  ifstream pf;
  pf.open(file.c_str());
  if(pf){                              //-Code of JSph::ReadVarAcc().
    pf.seekg(0,ios::end);
    unsigned len=(unsigned)pf.tellg(); //printf("FileSize: %u\n",len);
    pf.seekg(0,ios::beg);
    Resize(SIZEINITIAL);
    AccCount=0;                        //Zero the count for this input file number.
    std::string s,item;
    bool mkHeaderRead=false;           //Flag to determine whether the header values (mk/CoG) have been read in.
    bool cogHeaderRead=false;          //Flag to determine whether the centre of gravity values have been read in.
    bool flagHeaderRead=false;         //SL: New flag to determine whether header that contains whether gravity is enabled has been read in.
    int lineCount=1;                   //Line counter.
    while(getline(pf, s)){             //Get the line from the file.
      if(s.find('#')==string::npos){   //Read any line that does not contain a hash and is therefore a comment.
        if(!mkHeaderRead){             //We haven't yet read the MK value in.
          //Iterate through all the items in the stream, checking for space separated values.
          std::istringstream linestream(s); //Get the current line as an istringstream.
          getline(linestream, item);        //Grab the value.
          std::stringstream tmpItem(item);  //Convert current value to a stringstream.
          tmpItem >> MkFluid;               //Push the value to the correct MkValues array location.
          if(MkFluid<=MKFLUIDMAX) //Quick sanity check that the value read was zero or greater.
            mkHeaderRead=true;         //it was so set the flag.
          else               //Value entered wasn't sensible.
            RunException(met,"The MK value in variable acceleration file was out of possible range.",file);
        }
        else if(!cogHeaderRead){
          //Iterate through all the items in the stream, checking for comma separated values.
          std::istringstream linestream(s); //Get the current line as an istringstream.
          //Initialise the current centre of gravity variable to -FLT_MIN so a sanity check can be performed.
          AccCoG=TFloat3(-FLT_MIN);
          unsigned count=0;  //Switch counter.
          //Iterate through all the items in the stream, checking for space separated values.
          while(getline(linestream, item, ',')){
            stringstream tmpItem(item);     //Convert current value to a stringstream.
            //Switch over the 3 possible cases.
            switch (count){
              case 0:  tmpItem >> AccCoG.x;  break;   //X Value.
              case 1:  tmpItem >> AccCoG.y;  break;   //Y Value.
              case 2:  tmpItem >> AccCoG.z;  break;   //Z Value.
            }
            count++;
          }
          //Quick check the three values were actually read in.
          if(AccCoG.x!=-FLT_MIN&&AccCoG.y!=-FLT_MIN&&AccCoG.z!=-FLT_MIN)
            cogHeaderRead=true;
          else               //One or more values were not read in.
            RunException(met,"At least one value for the centre of gravity was missing in variable acceleration file.",file);
        }
        else if(!flagHeaderRead){ //SL: New header reading
          //Iterate through all the items in the stream, checking for space separated values.
          std::istringstream linestream(s); //Get the current line as an istringstream.
          getline(linestream, item);        //Grab the value.
          std::stringstream tmpItem(item);  //Convert current value to a stringstream.
          if(tmpItem.str().compare("true")==0||tmpItem.str().compare("True")==0||tmpItem.str().compare("TRUE")==0){ //True entered, using global gravity
            GravityEnabled=true;
            flagHeaderRead=true;
          }
          else if(tmpItem.str().compare("false")==0||tmpItem.str().compare("False")==0||tmpItem.str().compare("FALSE")==0){ //False entered, not using global gravity
            GravityEnabled=false;
            flagHeaderRead=true;
          }
          else //Something incorrect entered, inform and quit
            RunException(met,"The flag to set global gravity was invalid.",file);
        }
        else{                //We've read the Mk value and CoG values in so anything left must be data.
          //Temporary variables to hold time and linear/angular acceleration values.
          float time=-FLT_MIN;
          tfloat3 LinAcc=TFloat3(-FLT_MIN);
          tfloat3 AngAcc=TFloat3(-FLT_MIN);
          std::istringstream linestream(s); //Get the current line as an istringstream.
          unsigned count=0;                 //Switch counter.
          //Iterate through all the items in the stream, checking for comma separated values.
          while(getline(linestream, item, ',')){
            std::stringstream tmpItem(item);//Convert current value to a stringstream.
            //Switch over the 7 possible cases.
            switch (count){
              case 0:  tmpItem >> time;      break;   //Time.
              case 1:  tmpItem >> LinAcc.x;  break;   //Linear X.
              case 2:  tmpItem >> LinAcc.y;  break;   //Linear Y.
              case 3:  tmpItem >> LinAcc.z;  break;   //Linear Z.
              case 4:  tmpItem >> AngAcc.x;  break;   //Angular X.
              case 5:  tmpItem >> AngAcc.y;  break;   //Angular Y.
              case 6:  tmpItem >> AngAcc.z;  break;   //Angular Z.
            }
            count++;
          }
          //Quick sanity check that we at least read in the values.
          if(time!=-FLT_MIN&&LinAcc.x!=-FLT_MIN&&LinAcc.y!=-FLT_MIN&&LinAcc.z!=-FLT_MIN&&AngAcc.x!=-FLT_MIN&&AngAcc.y!=-FLT_MIN&&AngAcc.z!=-FLT_MIN){
            //Resize all the appropriate arrays to accept a new value.
            if(AccCount>=AccSize){
              unsigned newsize=unsigned(float(len)/float(pf.tellg())*1.05f*(AccCount+1))+100;
              //printf("---> Size: %u -> %u   tellg: %u / %u\n",Size,newsize,unsigned(pf.tellg()),len);
              Resize(newsize);
            } 
            //Save the loaded time value.
            AccTime[AccCount]=time;
            //If the read value for time is greater than the currently stored value then store the read value.
            if(time>timeMax)timeMax=time;
            //Save the loaded gravity vector.
            AccLin[AccCount]=LinAcc;
            //SL: Calculate angular velocity vector based on acceleration and time data loaded
            tfloat3 CurrVelLin=TFloat3(0.0f); //SL: New linear velocity variable
            if(AccCount!=0){ //SL: Angular velocity is always zero at time zero
              CurrVelLin.x=VelLin[AccCount-1].x+(AccLin[AccCount].x*(AccTime[AccCount]-AccTime[AccCount-1]));
              CurrVelLin.y=VelLin[AccCount-1].y+(AccLin[AccCount].y*(AccTime[AccCount]-AccTime[AccCount-1]));
              CurrVelLin.z=VelLin[AccCount-1].z+(AccLin[AccCount].z*(AccTime[AccCount]-AccTime[AccCount-1]));
            }
            //SL: Save the calculated angular velocity vector
            VelLin[AccCount]=CurrVelLin;
            //Save the loaded angular velocity vector (may be zero).
            AccAng[AccCount]=AngAcc;
            //SL: Calculate angular velocity vector based on acceleration and time data loaded
            tfloat3 CurrVelAng=TFloat3(0.0f); //SL: New angular velocity variable
            if(AccCount!=0){ //SL: Angular velocity is always zero at time zero
              CurrVelAng.x=VelAng[AccCount-1].x+(AccAng[AccCount].x*(AccTime[AccCount]-AccTime[AccCount-1]));
              CurrVelAng.y=VelAng[AccCount-1].y+(AccAng[AccCount].y*(AccTime[AccCount]-AccTime[AccCount-1]));
              CurrVelAng.z=VelAng[AccCount-1].z+(AccAng[AccCount].z*(AccTime[AccCount]-AccTime[AccCount-1]));
            }
            //SL: Save the calculated angular velocity vector
            VelAng[AccCount]=CurrVelAng;
            AccCount++;      //Increment the global line counter.
          }
          else{ //There was a problem
            stringstream tmpErrorFile;
            tmpErrorFile << "At least one value of line [" << lineCount << "] of variable acceleration file was not read correctly.";
            RunException(met,tmpErrorFile.str(),file);
          }
        }
      }
      lineCount++;           //Increment the line counter.
    }
    pf.close();
  }
  else RunException(met,"Cannot open the file.",file);
  //Check that at least 2 values were given or interpolation will be impossible.
  if(AccCount<2)RunException(met,"Cannot be less than two positions in variable acceleration file.",file);
  //Check that the final value for time is not smaller than the final simulation time.
  if(double(timeMax)<tmax){
    stringstream tmpErrorFile;
    tmpErrorFile << "Final time [" << timeMax << "] is less than total simulation time in variable acceleration file.";
    RunException(met,tmpErrorFile.str(),file);
  } 
}

//=================================================================================================================
/// Returns interpolation variable acceleration values. SL: Added angular and linear velocity and set gravity flag
//=================================================================================================================
void JSphVarAccFile::GetAccValues(double timestep,unsigned &mkfluid,tdouble3 &acclin,tdouble3 &accang,tdouble3 &centre,tdouble3 &velang,tdouble3 &vellin,bool &setgravity){
  double currtime=AccTime[AccIndex];
  //Find the next nearest time value compared to the current simulation time (current value used if still appropriate).
  while((AccIndex<(AccCount-1))&&(timestep>=currtime)){
    AccIndex++;                   //Increment the index.
    currtime=AccTime[AccIndex];   //Get the next value for time.
  }
  //Not yet reached the final time value, so interpolate new values.
  if(AccIndex>0 && AccIndex<AccCount){
    const double prevtime=AccTime[AccIndex-1];    //Get the previous value for time.
    //Calculate a scaling factor for time.
    const double tfactor=(timestep-prevtime)/(currtime-prevtime);
    //Interpolate and store new value for linear accelerations. (SL: changed variable names to make sense for angular velocity use)
    tdouble3 currval=ToTDouble3(AccLin[AccIndex]);
    tdouble3 prevval=ToTDouble3(AccLin[AccIndex-1]);
    CurrAccLin.x=prevval.x+(tfactor*(currval.x-prevval.x));
    CurrAccLin.y=prevval.y+(tfactor*(currval.y-prevval.y));
    CurrAccLin.z=prevval.z+(tfactor*(currval.z-prevval.z));
    //Interpolate and store new value for angular accelerations.
    currval=ToTDouble3(AccAng[AccIndex]);
    prevval=ToTDouble3(AccAng[AccIndex-1]);
    CurrAccAng.x=prevval.x+(tfactor*(currval.x-prevval.x));
    CurrAccAng.y=prevval.y+(tfactor*(currval.y-prevval.y));
    CurrAccAng.z=prevval.z+(tfactor*(currval.z-prevval.z));
    //SL: Interpolate and store new value for linear velocity.
    currval=ToTDouble3(VelLin[AccIndex]);
    prevval=ToTDouble3(VelLin[AccIndex-1]);
    CurrVelLin.x=prevval.x+(tfactor*(currval.x-prevval.x));
    CurrVelLin.y=prevval.y+(tfactor*(currval.y-prevval.y));
    CurrVelLin.z=prevval.z+(tfactor*(currval.z-prevval.z));
    //SL: Interpolate and store new value for angular velocity.
    currval=ToTDouble3(VelAng[AccIndex]);
    prevval=ToTDouble3(VelAng[AccIndex-1]);
    CurrVelAng.x=prevval.x+(tfactor*(currval.x-prevval.x));
    CurrVelAng.y=prevval.y+(tfactor*(currval.y-prevval.y));
    CurrVelAng.z=prevval.z+(tfactor*(currval.z-prevval.z));
  }
  else{ //Reached the final time value, truncate to that value.
    const unsigned index=(AccIndex>0? AccIndex-1: 0);
    CurrAccLin=ToTDouble3(AccLin[index]);     //Get the last position for linear acceleration.
    CurrAccAng=ToTDouble3(AccAng[index]);     //Get the last position for angular acceleration.
    CurrVelLin=ToTDouble3(VelLin[index]);     //SL: Get the last position for angular velocity.
    CurrVelAng=ToTDouble3(VelAng[index]);     //SL: Get the last position for angular velocity.
  }
  //Return values.
  mkfluid=MkFluid;
  acclin=CurrAccLin;
  accang=CurrAccAng;
  centre=ToTDouble3(AccCoG);
  vellin=CurrVelLin; //SL: Added linear velocity
  velang=CurrVelAng; //SL: Added angular velocity
  setgravity=GravityEnabled; //SL: Added set gravity flag
}

//##############################################################################
//##############################################################################
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JSphVarAcc::JSphVarAcc(){
  ClassName="JSphVarAcc";
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphVarAcc::~JSphVarAcc(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphVarAcc::Reset(){
  BaseFile="";
  for(unsigned c=0;c<Files.size();c++)delete Files[c];
  Files.clear();
  MemSize=0;
}

//==============================================================================
/// Method to load data from input files for variable acceleration.
//==============================================================================
void JSphVarAcc::Config(std::string basefile,unsigned files,double tmax){
  Reset();
  BaseFile=basefile;
  for(unsigned cf=0;cf<files;cf++){
    JSphVarAccFile *accfile=new JSphVarAccFile();
    Files.push_back(accfile);
    
    //Construct a file name (assuming sequential numbering).
    std::stringstream filename;
    filename << BaseFile << "_" << cf;

    //File did not exit so add .csv.
    if(!fun::FileExists(filename.str()))filename << ".csv";

    accfile->LoadFile(filename.str(),tmax);
    MemSize+=Files[cf]->GetAllocMemory();
  }
}

//=====================================================================================================================================================
/// Returns interpolation variable acceleration values. SL: Corrected spelling mistake in exception and added angular velocity and global gravity flag
//=====================================================================================================================================================
void JSphVarAcc::GetAccValues(unsigned cfile,double timestep,unsigned &mkfluid,tdouble3 &acclin,tdouble3 &accang,tdouble3 &centre,tdouble3 &velang,tdouble3 &vellin,bool &setgravity){
  if(cfile>=GetCount())RunException("GetAccValues","The number of input file for variable acceleration is invalid.");
  Files[cfile]->GetAccValues(timestep,mkfluid,acclin,accang,centre,velang,vellin,setgravity);
}
