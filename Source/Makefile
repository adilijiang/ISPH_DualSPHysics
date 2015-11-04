#DualSPHysics v4 Re (03-10-2015)

#=============== Cluster selection ===============
CLUSTER=EPH
### EPH -> Uses g++ compiler
### EPH -> To use Cuda 4.0: module load cuda/4.0
### CSF -> Uses g++ compiler
### CSF -> To use Cuda 4.0: module load libs/cuda/4.0.17
### BSC -> Uses icpc compiler 
### BSC -> To use Cuda 4.0: module unload cuda/4.1 && module load cuda/4.0
### EME -> To use icpc: module load intel/12.1
### EME -> To use Cuda 4.0: module load cuda/4.0.17

#=============== Config other flags to compile ===============
#CCMOREFLAGS= -D_CODE_FAST
#CCMOREFLAGS= -D_CODE_FASTBALANCING

#=============== Files to compile ===============
OBJ_BASIC=main.o Functions.o FunctionsMath.o JArraysCpu.o JBinaryData.o JCellDivCpu.o JCfgRun.o JException.o
OBJ_BASIC:=$(OBJ_BASIC) JLinearValue.o JLog2.o JObject.o JPartDataBi4.o JPartFloatBi4.o JPartOutBi4Save.o JPartsOut.o 
OBJ_BASIC:=$(OBJ_BASIC) JRadixSort.o JRangeFilter.o JSaveDt.o JSpaceCtes.o JSpaceEParms.o JSpaceParts.o 
OBJ_BASIC:=$(OBJ_BASIC) JSpaceProperties.o JSph.o JSphCpu.o JSphDtFixed.o JSphVarAcc.o JSphVisco.o randomc.o
OBJ_BASIC:=$(OBJ_BASIC) JTimersStep.o
OBJ_CPU_SINGLE=JCellDivCpuSingle.o JSphCpuSingle.o JPartsLoad4.o
OBJ_GPU=JArraysGpu.o JCellDivGpu.o JObjectGpu.o JPtxasInfo.o JSphGpu.o
OBJ_GPU_SINGLE=JCellDivGpuSingle.o JSphGpuSingle.o
OBJ_CUDA=JCellDivGpu_ker.o JSphGpu_ker.o
OBJ_CUDA_SINGLE=JCellDivGpuSingle_ker.o
OBJECTS=$(OBJ_BASIC) $(OBJ_CPU_SINGLE) $(OBJ_GPU) $(OBJ_CUDA) $(OBJ_GPU_SINGLE) $(OBJ_CUDA_SINGLE)

#=============== To select GPU architecture ===============
#GENCODE:=$(GENCODE) -gencode=arch=compute_10,code=\"sm_10,compute_10\"
#GENCODE:=$(GENCODE) -gencode=arch=compute_12,code=\"sm_12,compute_12\"
#GENCODE:=$(GENCODE) -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE:=$(GENCODE) -gencode=arch=compute_20,code=\"sm_20,compute_20\"

#=============== DualSPHysics libs to include ===============
JLIBS=-L./ -ljxml_64 -ljformatfiles2_64 -ljsphmotion_64 -ljwavegen_64

#=============== To compile CPU code ===============
CC=g++
CCFLAGS=-c -O3 -fopenmp -D_WITHGPU
CCLINKFLAGS=-fopenmp -lgomp
ifneq (,$(filter $(CLUSTER),BSC EME))
  CC=icpc
  CCFLAGS=-c -O3 -openmp -D_WITHGPU
  CCLINKFLAGS=-openmp
endif
CCFLAGS := $(CCFLAGS) $(CCMOREFLAGS)

#=============== To compile GPU code ===============
DIRTOOLKIT=
ifeq ($(CLUSTER),EPH)
  DIRTOOLKIT=/exports/opt/NVIDIA/cuda-4.0
endif
ifeq ($(CLUSTER),CSF)
  DIRTOOLKIT=/opt/gridware/libs/nvidia-cuda/toolkit/4.0.17
endif
ifeq ($(CLUSTER),BSC)
  DIRTOOLKIT=/opt/cuda/4.0
endif
ifeq ($(CLUSTER),EME)
  DIRTOOLKIT=/apps/cuda/4.0.17/cuda
endif
CCFLAGS := $(CCFLAGS) -I./ -I$(DIRTOOLKIT)/include
CCLINKFLAGS := $(CCLINKFLAGS) -L$(DIRTOOLKIT)/lib64 -lcudart
NCC=nvcc
NCCFLAGS=-c $(GENCODE) -use_fast_math -O3


all:dualsphysics4gpu 
	rm -rf *.o
	@echo "  --- Compiled GPU version for the cluster $(CLUSTER) ---"

dualsphysics4gpu:  $(OBJECTS)
	$(CC) $(CCLINKFLAGS) $(OBJECTS) -o $@ $(JLIBS)

.cpp.o: 
	$(CC) $(CCFLAGS) $< 

JSphGpu_ker.o: JSphGpu_ker.cu
	$(NCC) $(NCCFLAGS) --ptxas-options -v JSphGpu_ker.cu 2>dualsphysics4gpu_ptxasinfo

JCellDivGpu_ker.o: JCellDivGpu_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpu_ker.cu

JCellDivGpuSingle_ker.o: JCellDivGpuSingle_ker.cu
	$(NCC) $(NCCFLAGS) JCellDivGpuSingle_ker.cu

clean:
	rm -rf *.o dualsphysics4gpu dualsphysics4gpu_ptxasinfo

help:
	@echo "  make CLUSTER=[EPH,CSF,BSC,EME]"
	@echo "    EPH: Ephyslab - Universidad de Vigo"
	@echo "    CSF: University of Manchester"
	@echo "    BSC: Barcelona Supercomputing Center"
	@echo "    EME: Emerald - e-Infrastructure South"

