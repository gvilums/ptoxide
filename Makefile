NVCC=nvcc -ptx -arch sm_89

SOURCES=$(wildcard kernels/*.cu)
PTX=$(SOURCES:.cu=.ptx)

all: $(PTX)

%.ptx: %.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(PTX)