# Makefile

EXE=d2q9-bgk

CC=mpiicc
CFLAGS= -std=c99 -Wall -O3 -no-prec-div -vec-threshold=0 -fno-alias -xHost
LIBGPUS = -lm -D DEVICE=CL_DEVICE_TYPE_CPU -lOpenCL
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: GPU-lbm.c CPU-lbm.c
	make cpu
	make gpu

gpu: GPU-lbm.c
	$(CC) $(CFLAGS) $^ $(LIBGPUS) -o GPU-lbm

cpu: CPU-lbm.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o CPU-lbm

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

profile:
	scorep --opencl $(CC) $(CFLAGS) d2q9-bgk.c $(LIBS) -o d2q9-bgk-scorep

clean:
	rm -f $(EXE)
