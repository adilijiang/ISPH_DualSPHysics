@echo off

@set tools="..\_Tools_"
@set partvtk4=%tools%\PartVtk4_Release64.exe
@set partvtkout4=%tools%\PartVtkOut4_Release64.exe

@set dirout=particles
if exist %dirout% del /Q %dirout%\*.*
if not exist %dirout% mkdir %dirout%

rem @set vars="#-vars:-all,vel,rhop,id,type"
rem @set onlytype="-onlytype:-all,fluid,floating"

%partvtk4% -dirin . -savevtk %dirout%/All #-onlytype:-all,fluid #-vars:ace#-filexml#WavesBox2d.xml

%partvtkout4% -dirin . -savevtk %dirout%/Out -saveresume %dirout%/_OutResume.csv

copy Run.out %dirout%

rem %partvtk% -dirin . -savevtk %dirout%/Fluid -onlytype:-all,fluid 
rem %partvtk% -dirin . -savevtk %dirout%/Bound -onlytype:-all,bound

rem %partvtk% -dirin . -savevtk %dirout%/AllFluid -onlytype:-bound -vars:-type,mk,-vel -filexml CasePhase_300k.xml

rem %partvtk% -dirin . -savevtk %dirout%/Fluid -onlytype:-all,fluid -savevtk %dirout%/Floating -onlytype:-all,floating -savevtk %dirout%/All -onlytype:+all
