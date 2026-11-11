# Build the two native libraries used by the SHM PP transport.
#
#   libppshm.so  : POSIX shared-memory SPSC ring (control + payload pages),
#                  release/acquire publication, cross-process futexes.
#   libppdma.so  : batched/fused D2H+H2D copy submission on the caller's
#                  CUDA/HIP stream, bound to libppshm.so slot bookkeeping.
#
# Both are Linux/glibc only. CUDA and HIP versions of libppdma.so are the
# same binary: the GPU runtime (libcudart.so / libamdhip64.so) is resolved at
# dlopen() time from GAN_SHM_GPU_RUNTIME or from the loaded process image.

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS  ?= -lrt -lpthread

PREFIX  ?= .
BUILD   ?= .

all: $(BUILD)/libppshm.so $(BUILD)/libppdma.so

$(BUILD)/libppshm.so: ppshm.c ppshm.h
	$(CC) $(CFLAGS) -shared -o $@ ppshm.c $(LDLIBS)

$(BUILD)/libppdma.so: gpu_runtime_hook.c
	$(CC) $(CFLAGS) -shared -o $@ gpu_runtime_hook.c -ldl -lpthread

clean:
	rm -f $(BUILD)/libppshm.so $(BUILD)/libppdma.so

# Quick ABI smoke test on a Linux host (no GPU required).
check: all
	@python3 -c "import ctypes;l=ctypes.CDLL('$(BUILD)/libppshm.so');\
	l.pp_create.argtypes=[ctypes.c_char_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32];\
	l.pp_create.restype=ctypes.c_int;\
	l.pp_open.argtypes=[ctypes.c_char_p];l.pp_open.restype=ctypes.c_void_p;\
	l.pp_close.argtypes=[ctypes.c_void_p];\
	l.pp_unlink.argtypes=[ctypes.c_char_p];\
	print('ppshm ABI ok, create=',l.pp_create(b'abi-smoke',2,65536,1024),\
	'unlink=',l.pp_unlink(b'abi-smoke'))"

.PHONY: all clean check
