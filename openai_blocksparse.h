#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include "gpu_types.h"

typedef unsigned long long uint64;
typedef unsigned char uchar;


template <bool K64>
__global__ void __launch_bounds__(256,3) bst_sgemm_32x32x64_nt(
    const uint2* __restrict__ Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut, uint loops)
{
    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;


    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // blockid
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head = Lut[idx_H*szLut + bid];

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;
    uint offsetA00 = idx_B*szCtxHeadStateA + (idx_M*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetB00 = idx_B*szCtxHeadStateB + (idx_N*32 + ty)*szHeadState + idx_H*szState + k;
    uint offsetA16 = offsetA00 + szHeadState*16;
    uint offsetB16 = offsetB00 + szHeadState*16;

    uint tid224 = tid & 224; // 256 - 32

    // avoid bank conflicts when writing transpose (+ tx*2)
    uint storAB = (tx*32*4 + ty + tx*2)*4;

    // 32 threads per tile, each tile reads 8 lines, shifted over by 4
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;

    loadA += (tid224 * 32) + (tid224 / 2); // 32*8*4
    loadB += (tid224 * 32) + (tid224 / 2); // 32*8*4

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    uint loop = 0;
    #pragma unroll 1
    do
    {
        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        if (K64 || k < szState)
        {
            a00 = __ldg((const float4*)(add_ptr_u(A, offsetA00)));
            a16 = __ldg((const float4*)(add_ptr_u(A, offsetA16)));
            b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));
        }
        offsetA00 += 64;
        offsetA16 += 64;
        offsetB00 += 64;
        offsetB16 += 64;
        if (!K64)
            k += 64;

        __syncthreads();
        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }


    } while (++loop < loops);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

    //printf("%3d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", tid, regC[0][0], regC[0][1], regC[0][2], regC[0][3], regC[4][0], regC[4][1], regC[4][2], regC[4][3]);

    // if ((tid & 31) == 0)
    //     printf("%3d %.0f\n", tid, regC[0][0]);

    C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

    // Arrange 8 tiles horizontally in the X direction: ((tid & 224) >> 1)
    // Add some spacing  to avoid write bank conflicts: (ty << 2)
    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];

    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = ew_add(c2[i], c2[i+j]);

    store((bhalf2*)C, c2[0]);



    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = ew_add(c2[i], c2[i+j]);

    store((bhalf2*)(C + 16*32), c2[0]);
}


bool bst_sgemm_nt(CUstream stream,
    const uint2* lut,
    const float* a,
    const float* b,
          bhalf* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;

    dim3 grid(blocks, batch_dim, head_dim);
    if (block_size == 32)
    {
        if (k64)
            bst_sgemm_32x32x64_nt< true><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
        else
            bst_sgemm_32x32x64_nt<false><<<grid,256,0,stream>>>(lut, a, b, c, szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, loops);
    }
    return true;
};
