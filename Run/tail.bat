@echo off
if {%1}=={} @echo Nombre de archivo requerido.&goto :EOF
if not exist %1 @echo %1 NO existe.&goto :EOF
setlocal
set file=%1
set /a number=10
if not {%2}=={} set /a number=%2
for /f %%i in ('find /v /c "" ^< %file%') do set /a lines=%%i
@rem echo %lines% lineas en el archivo: %file%.
if %number% GEQ %lines% set /a start=0&goto console
set /a start=%lines% - %number%
:console
more /e +%start% %file%
endlocal
