#DualSPHysics GPU v4 (28-03-2016)

#=============== Compilation Options ===============
USE_DEBUG=NO
USE_FAST_MATH=YES
USE_NATIVE_CPU_OPTIMIZATIONS=YES

EXECS_DIRECTORY=../../../EXECS

ifeq ($(USE_DEBUG), YES)
  CCFLAGS=-c -O0 -g -Wall -fopenmp -D_WITHGPU
else
  CCFLAGS=-c -O3 -fopenmp -D_WITHGPU
  ifeq ($(USE_FAST_MATH), YES)
    CCFLAGS+= -ffast-math
  endif
  ifeq ($(USE_NATIVE_CPU_OPTIMIZATIONS), YES)
    CCFLAGS+= -march=native
  endif
endif
CC=g++
CCLINKFLAGS=-fopenmp -lgomp

#=============== CUDA toolkit directory (make appropriate for local CUDA installation) ===============
DIRTOOLKIT=/usr/local/cuda
DIRTOOLKIT=/exports/opt/NVIDIA/cuda-7.5

#=============== Files to compile ===============
OBJ_BASIC=main.o Functions.o FunctionsMath.o JArraysCpu.o JBinaryData.o JCellDivCpu.o JCfgRun.o JException.o
OBJ_BASIC:=$(OBJ_BASIC) JLog2.o JObject.o JPartDataBi4.o JPartFloatBi4.o JPartOutBi4Save.o JPartsOut.o 
OBJ_BASIC:=$(OBJ_BASIC) JRadixSort.o JRangeFilter.o JReadDatafile.o JSaveDt.o JSpaceCtes.o JSpaceEParms.o JSpaceParts.o 
OBJ_BASIC:=$(OBJ_BASIC) JSpaceProperties.o JSph.o JSphAccInput.o JSphCpu.o JSphDtFixed.o JSphVisco.o randomc.o
OBJ_BASIC:=$(OBJ_BASIC) JTimeOut.o
OBJ_CPU_SINGLE=JCellDivCpuSingle.o JSphCpuSingle.o JPartsLoad4.o
OBJ_GPU=JArraysGpu.o JCellDivGpu.o JObjectGpu.o JSphGpu.o JBlockSizeAuto.o JMeanValues.o
OBJ_GPU_SINGLE=JCellDivGpuSingle.o JSphGpuSingle.o
OBJ_CUDA=JCellDivGpu_ker.o JSphGpu_ker.o
OBJ_CUDA_SINGLE=JCellDivGpuSingle_ker.o
OBJECTS=$(OBJ_BASIC) $(OBJ_CPU_SINGLE) $(OBJ_GPU) $(OBJ_CUDA) $(OBJ_GPU_SINGLE) $(OBJ_CUDA_SINGLE)

#=============== Select GPU architectures ===============
GENCODE:=$(GENCODE) -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE:=$(GENCODE) -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE:=$(GENCODE) -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE:=$(GENCODE) -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE:=$(GENCODE) -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE:=$(GENCODE) -gencode=arch=compute_52,code=\"sm_52,compute_52\"

#=============== DualSPHysics libs to be included ===============
JLIBS=-L./ -ljxml_64 -ljformatfiles2_64 -ljsphmotion_64 -ljwavegen_64

#=============== GPU Code Compilation ===============
CCFLAGS := $(CCFLAGS) -I./ -I$(DIRTOOLKIT)/include
CCLINKFLAGS := $(CCLINKFLAGS) -L$(DIRTOOLKIT)/lib64 -lcudart
NCC=nvcc
ifeq ($(USE_DEBUG), NO)
  NCCFLAGS=-c $(GENCODE) -O3
else
  NCCFLAGS=-c $(GENCODE) -O0 -g
endif
ifeq ($(USE_FAST_MATH), YES)
  NCCFLAGS+= -use_fast_math
endif

all:$(EXECS_DIRECTORY)/DualSPHysics4_linux64 
	rm -rf *.o
ifeq ($(USE_DEBUG), NO)
	@echo "  --- Compiled Release GPU version ---"
else
	@echo "  --- Compiled Debug GPU version ---"
	mv DualSPHysics4_linux64 DualSPHysics4_linux64_debug
endif

$(EXECS_DIRECTORY)/DualSPHysics4_linux64:  $(OBJECTS)
	$(CC) $(OBJECTS) $(CCLINKFLAGS) -o $@ $(JLIBS)

.cpp.o: 
	$(CC) $(CCFLAGS) $< 

JSphGpu_ker.o: JSphGpu_ker.cu
	$(NCC) $(NCCFLAGS) JSphGpu_ker.cu

JCellDivGpu_ker.o: JCellDivGpu_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpu_ker.cu

JCellDivGpuSingle_ker.o: JCellDivGpuSingle_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpuSingle_ker.cu

JPeriodicGpu_ker.o: JPeriodicGpu_ker.cu
	$(NCC) $(NCCFLAGS) JPeriodicGpu_ker.cu

clean:
	rm -rf *.o DualSPHysics4_linux64 DualSPHysics4_linux64_debug

