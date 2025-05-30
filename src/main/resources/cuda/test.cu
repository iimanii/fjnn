#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
#include <curand_kernel.h>

#include "util.h"
/* from wikipedia page, for machine epsilon calculation */
/* assumes mantissa in final bits */
__device__ double machine_eps_dbl() {
    typedef union {
        long long i64;
        double d64;
    } dbl_64;

    dbl_64 s;

    s.d64 = 1.;
    s.i64++;
    return (s.d64 - 1.);
}

__device__ float machine_eps_flt() {
    typedef union {
        int i32;
        float f32;
    } flt_32;

    flt_32 s;

    s.f32 = 1.;
    s.i32++;
    return (s.f32 - 1.);
}

#define EPS 0
#define MIN 1
#define MAX 2

extern "C"
__global__ void calc_consts(float *fvals, double *dvals) {

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i==0) {
        fvals[EPS] = machine_eps_flt();
        dvals[EPS]= machine_eps_dbl();

        float xf, oldxf;
        double xd, oldxd; 

        xf = 2.; oldxf = 1.;
        xd = 2.; oldxd = 1.;

        /* double until overflow */
        /* Note that real fmax is somewhere between xf and oldxf */
        while (!isinf(xf))  {
            oldxf *= 2.;
            xf *= 2.;
        }

        while (!isinf(xd))  {
            oldxd *= 2.;
            xd *= 2.;
        }

        dvals[MAX] = oldxd;
        fvals[MAX] = oldxf;

        /* half until overflow */
        /* Note that real fmin is somewhere between xf and oldxf */
        xf = 1.; oldxf = 2.;
        xd = 1.; oldxd = 2.;

        while (xf != 0.)  {
            oldxf /= 2.;
            xf /= 2.;
        }

        while (xd != 0.)  {
            oldxd /= 2.;
            xd /= 2.;
        }

        dvals[MIN] = oldxd;
        fvals[MIN] = oldxf;

    }
    return;
}


__forceinline__ 
__device__ int next(curandState_t *state) {
    return curand(state);
}

__forceinline__ 
__device__ float nextGaussian(curandState_t *state, float min, float max) {
    float range = max - min;
    
    float result = curand_normal(state);
    
    return ((result + 5) / 10) * range + min;
}

__forceinline__ 
__device__ float nextGaussian(curandState_t *state) {
    return curand_normal(state);
}

__forceinline__ 
__device__ float nextFloat(curandState_t *state) {
    return curand_uniform(state);
}

extern "C"
__global__ void initState(unsigned long long seed, curandState_t *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, 0, 0, &state[i]);
}

extern "C"
__global__ void empty() {

}

extern "C"
__global__ void test_gauss(unsigned long long seed, float* r, curandState_t* state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    curandState_t state1; 
  
//    if(threadIdx.x == 0)
//        printf("%d\n", sizeof(state));
//    curand_init(seed, i, 0, &state[i]);
    
    r[i] = nextGaussian(&state[i], -1, 1);
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
}

extern "C"
__global__ void test_gauss2(unsigned long long seed, float* r) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandState_t state;
//    curand_init(seed, i, 0, &state);
    
    r[i] = nextGaussian(&state);
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
}

extern "C"
__global__ void test_float(unsigned long long seed, float* r) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandState_t state;
//    curand_init(seed, i, 0, &state);
    
    r[i] = nextFloat(&state);
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
//    printf("%5d %f %f\n", i, nextGaussian(&state, -0.5, 0.5), nextGaussian(&state, -1, 1));
}

extern "C"
__global__ void slow(float* r, long size) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int base = blockIdx.x * blockDim.x;
    
    for(int x=0; x < 777; x++) {
        int i = threadIdx.x % 57;
        int j = i % 2 == 1 ? -1 : 1;
        int index = (threadIdx.x + i + j) % THREADS_PER_BLOCK;
        if(base + index < size)
            cache[threadIdx.x] = cache[index] + r[base + index];
    }
    
    if(base + threadIdx.x < size)
        r[base + threadIdx.x] = base + threadIdx.x;
}

extern "C"
__global__ void slowLoop(float* data) {
    int idx = threadIdx.x;
    float value = data[idx];
    // Simulate a long computation by running a large loop
    for (int i = 0; i < 10000000; i++) {
        value = value + 0.0001f;
    }
    data[idx] = value;
}


// 1D Simple - no shared memory
extern "C"
__global__ void add_stride_1d_simple(float* a, float* b, float* c, long stride_size, long total_elements) {
    long global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_id < total_elements) {
        long stride_id = global_id / stride_size;
        long element_pos = global_id % stride_size;
        
        long index = stride_id * stride_size + element_pos;
        c[index] = a[index] + b[element_pos];
    }
}

// 1D Shared - loads b into shared memory, 1D indexing
extern "C" 
__global__ void add_stride_1d_shared(float* a, float* b, float* c, long stride_size, long total_elements) {
    extern __shared__ float shared_b[];
    
    // Load b into shared memory cooperatively
    for (long i = threadIdx.x; i < stride_size; i += blockDim.x) {
        shared_b[i] = b[i];
    }
    __syncthreads();
    
    // Process elements using shared memory
    long global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_id < total_elements) {
        long stride_id = global_id / stride_size;
        long element_pos = global_id % stride_size;
        
        long index = stride_id * stride_size + element_pos;
        c[index] = a[index] + shared_b[element_pos];
    }
}

// 2D Simple - no shared memory, similar to original implementation
extern "C"
__global__ void add_stride_2d_simple(float* a, float* b, float* c, long stride_size, long total_size) {
    int element_pos = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (element_pos < stride_size) {
        int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
        long index = element_pos + stride_id * stride_size;
        
        if (index < total_size) {
            c[index] = a[index] + b[element_pos];
        }
    }
}

// 2D Shared - loads b into shared memory, 2D indexing
extern "C"
__global__ void add_stride_2d_shared(float* a, float* b, float* c, long stride_size, long count) {
    extern __shared__ float shared_b[];
    
    // Load b into shared memory cooperatively
    for (long i = threadIdx.x; i < stride_size; i += blockDim.x) {
        if (threadIdx.y == 0) { // Only first row loads
            shared_b[i] = b[i];
        }
    }
    __syncthreads();
    
    // Process elements using shared memory
    int element_pos = threadIdx.x + blockIdx.x * blockDim.x;
    int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
    
    if (element_pos < stride_size && stride_id < count) {
        long index = element_pos + stride_id * stride_size;
        c[index] = a[index] + shared_b[element_pos];
    }
}