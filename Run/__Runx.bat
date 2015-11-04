@echo off
rem %1 Modo de ejcucion: (cpu/gpu)    Ej: cpu
rem %2 Version de codigo.             Ej: R47
rem %3 Directorio de entrada.         Ej: Case5x
rem %4 Fichero de entrada.            Ej: Case5x_30k
rem %5 Fichero de log de ejecucion.   Ej: log.csv
rem %6 Marca identificativa.          Ej: _new
rem %7 Opciones de ejecucion.     Ej: "-verlet -tmax:0.2 -shepard:30"

set datacases="_DataCasesBi4_"

echo --- Run:%1_%2_%4%6 ---
echo --- %7 ---
set modo=%1
if %modo% == gpu:0 set modo=gpu
if %modo% == gpu:1 set modo=gpu
if %modo% == gpu:2 set modo=gpu

set dirout=%2_%modo%_%4%6
if exist %dirout% del /Q %dirout%\*.*
if not exist %dirout% mkdir %dirout%
if not exist %dirout% (
  echo Error: No existe el directorio de salida %dirout%
) else (
  copy _partvtk4.bat %dirout%
  rem copy _partvtkout4.bat %dirout%
  copy _isosurface.bat %dirout%
  rem copy _measure.bat %dirout%
  rem copy _boundvtk.bat %dirout%
  rem copy _fmtbi3_info.bat %dirout%
  copy %datacases%\%3\%4*.* %dirout%
  rem copy _datxml\%3\common_*.* %dirout%
  DualSPHysics_%2.exe -name %datacases%/%3/%4 -dirout %dirout% -runname %dirout% -%1 -svres %7  
  if exist %5 call tail %dirout%\Run.csv 1 >> %5
  if not exist %5 type %dirout%\Run.csv > %5
)
echo --- FinOk:%1_%2_%4%6 ---
echo --- %7 ---
rem fin