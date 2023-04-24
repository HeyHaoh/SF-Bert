#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "openai_blocksparse.h"

using namespace std;


int main(){
    const uint batch_dim = 32;
    const uint heads_ = 4;
    const uint head_state_ = 128;
    const uint blk_size_= 32;
    const uint blocks_= 55;
    const uint seq_len = 320;
    const uint state = 512;
    const uint m = batch_dim*seq_len;
    const uint n = state;
    
    const uint ctx_blks_a_ = 10;
    const uint ctx_blks_b_ = 10;
    const uint ctx_blks_c_ = 10;
    const uint lut_heads = 1;
    const uint nt_lut_dim = 1;
    const uint nn_lut_dim = 1;
    const uint tn_lut_dim = 1;
    const uint mask_heads= 1;
    const float scale = 1.0/sqrt(head_state_);
    const uint nn_op = 1; 
    const uint nt_op = 0;
    const uint tn_op = 2;
    int count = 0;
    const uint nn_max = 10;
    const uint tn_max = 10;



    CUstream custream;
    cudaStreamCreate(&custream); 

    uint2 *nt_lut = (uint2*)malloc(sizeof(uint2)*blocks_);

    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            if(i >= j){
                nt_lut[count].x = i;
                nt_lut[count].y = j;
                count++;
            }
            
        }
    }

    uint2 *l_ptr;
    cudaMalloc((void**)&l_ptr, sizeof(uint2)*blocks_);
    cudaMemcpy(l_ptr, nt_lut, sizeof(uint2)*blocks_, cudaMemcpyHostToDevice);


    float *a_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    float *b_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    for(int i=0;i<m*n;i++){

        a_ptr_cpu[i] = 1;
        b_ptr_cpu[i] = 1;

    }

    float *a_ptr, *b_ptr;
    cudaMalloc((void**)&a_ptr, sizeof(float)*m*n);
    cudaMalloc((void**)&b_ptr, sizeof(float)*m*n);

    cudaMemcpy(a_ptr, a_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    bhalf *c_ptr;
    cudaMalloc((void**)&c_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    bhalf *c_ptr_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    bst_sgemm_nt(custream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, nt_lut_dim);

    cudaMemcpy(c_ptr_cpu, c_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    return 0;

}