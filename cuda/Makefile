
INC	:= -I$(CUDA_HOME)/include -I. -I /usr/local/include/opencv4/
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo --ptxas-options=-v --use_fast_math

all: 		ES

ES:		ES.cu ES_kernel.h Makefile
		nvcc ES.cu -o ES \
		$(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f ES
