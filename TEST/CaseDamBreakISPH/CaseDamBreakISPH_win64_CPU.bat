@echo off

rem "name" and "dirout" are named according to the testcase

set name=CaseDamBreakISPH
set dirout=%name%_out

rem "executables" are renamed and called from their directory

set gencase="../GenCase2_Release64.exe"
set dualsphysics="../../Run/DualSPHysicsISPH_ReleaseCPU64.exe"
set partvtk4="../PartVTK4_Release64.exe"

rem "dirout" is created to store results or it is removed if it already exists

if exist %dirout% del /Q %dirout%\*.*
if not exist %dirout% mkdir %dirout%

rem CODES are executed according the selected parameters of execution in this tescase

%gencase% %name%_Def %dirout%/%name% -save:vtkall -bi4
if not "%ERRORLEVEL%" == "0" goto fail

%dualsphysics% %dirout%/%name% %dirout% -svres -cpu 
if not "%ERRORLEVEL%" == "0" goto fail

%partvtk4% -dirin %dirout% -savevtk %dirout%/Fluid -onlytype:-all,fluid
%partvtk4% -dirin %dirout% -savevtk %dirout%/Boundary -onlytype:-all,bound

if not "%ERRORLEVEL%" == "0" goto fail


:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
