# ================================================================
# VaultWatch Makefile
# ================================================================

HEADERS = math256.h ec_jacobian.h targets.h patoshi_targets.h

all: cuda cpu integrated-cpu

# ---- GPU verifier (pipe input) ----
cuda: vaultwatch-cuda.cu $(HEADERS)
	nvcc -arch=sm_61 -O3 -std=c++14 -lineinfo -o vaultwatch-cuda vaultwatch-cuda.cu -lcudart -lcuda

# ---- CPU verifier (pipe input) ----
cpu: vaultwatch-cuda.cu $(HEADERS)
	g++ -O3 -o vaultwatch-cpu vaultwatch-cuda.cu -I.

# ---- GPU integrated verifier ----
integrated-cuda: vaultwatch-integrated.cu $(HEADERS)
	nvcc -arch=sm_61 -O3 -std=c++14 -lineinfo -o vaultwatch-integrated vaultwatch-integrated.cu -lcudart -lcuda -lcudadevrt

# ---- CPU integrated verifier ----
integrated-cpu: vaultwatch-integrated.cu $(HEADERS)
	g++ -O3 -o vaultwatch-integrated vaultwatch-integrated.cu -I.

clean:
	rm -f vaultwatch-cpu vaultwatch-cuda vaultwatch-integrated *.o *.obj
