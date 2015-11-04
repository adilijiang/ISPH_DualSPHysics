@echo off

@set tools="..\_Tools_"
@set isosurface=%tools%\IsoSurface4_Release64.exe


@set dirout=iso
if exist %dirout% del /Q %dirout%\*.*
if not exist %dirout% mkdir %dirout%

@set vars="#-vars:-all,vel,rhop,id,type"
@set onlytype="#-onlytype:-all,fluid,floating"

%isosurface% -dirin . -saveiso %dirout%/iso %onlytype% %vars%

