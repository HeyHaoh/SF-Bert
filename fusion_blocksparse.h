#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include "ew_op_gpu.h"
#include "gpu_hmma.h"
#include "gpu_types.h"

typedef unsigned long long uint64;
#define LOG2e 1.4426950408889634f
typedef unsigned char uchar;


template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax1(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
                        uint2* rblk_lut,
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale)
{
    
    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head  = NT_Lut[idx_H*szLut + bid];
    uint rblk_idx      = rblk_lut[bid].y;
    uint lut_size_rblk = NN_Lut[idx_H*szNN + rblk_idx].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + rblk_idx].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;
    uint inf = 0xff80;

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

    // C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

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

    float2 c1[8]; bhalf2 c1b;
    float2 c2[8]; bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

    // store((bhalf2*)C, c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    // store((bhalf2*)(C + 16*32), c2[0]);

    if(tid == 0)
        atomicAdd(rblk_flag+(idx_B*szBatchRblk)+(idx_H*szHeadRblk)+rblk_idx, 1);

    // uint ty = tid/16;
    // uint tx = tid%16;
    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    
    
    float Xmax[2]; 
    float2 Xval[2];
    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 

    do{
        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
        
            
        if(tx == 0){
            store(Max_in_block+offset_in_ms, Xmax[0]);
            store(Max_in_block+offset_in_ms+(1*16), Xmax[1]);
        }

        __threadfence();

    }while(rblk_flag[(idx_B*szBatchRblk)+(idx_H*szHeadRblk)+(rblk_idx)] != lut_size_rblk);


    float max0 = -inf, max16 = 0;
    float Max0 = -inf, Max16 = 0;
    float Sum0 = -inf, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);
        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);

    }


    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    xval[0] = to_bhalf(Xval[0]); 

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    xval[1] = to_bhalf(Xval[1]); 

    __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tid == 0)
        atomicAdd(rblk_flag1+(idx_B*szBatchRblk)+(idx_H*szHeadRblk)+rblk_idx, 1);

    // if(tx == 0){

    //     store(Sum_in_rblock+offset_in_ms, Sum0);
    //     store(Sum_in_rblock+offset_in_ms+16, Sum16);

    // }
    
    do{
        
        // Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
        // Sum0  = ew_sum(Xval[0]);
        // xval[0] = to_bhalf(Xval[0]); 

        // Sum0 += shfl_xor(Sum0, 8);
        // Sum0 += shfl_xor(Sum0, 4);
        // Sum0 += shfl_xor(Sum0, 2);
        // Sum0 += shfl_xor(Sum0, 1);

        // Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
        // Sum16  = ew_sum(Xval[1]);
        // xval[1] = to_bhalf(Xval[1]); 


        // Sum16 += shfl_xor(Sum16, 8);
        // Sum16 += shfl_xor(Sum16, 4);
        // Sum16 += shfl_xor(Sum16, 2);
        // Sum16 += shfl_xor(Sum16, 1);

        if(tx == 0){

            store(Sum_in_rblock+offset_in_ms, Sum0);
            store(Sum_in_rblock+offset_in_ms+16, Sum16);

        }
        
        __threadfence();

    }while(rblk_flag1[(idx_B*szBatchRblk)+(idx_H*szHeadRblk)+rblk_idx] != lut_size_rblk);

    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul(to_float(xval[0]), rcp_exp_sum0);


    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul(to_float(xval[1]), rcp_exp_sum16);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);


}


template <typename T, typename V>
bool fusion_attention1(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
          uint2* rblk_lut,
    const  uint* Mask,
          float* Max,
          float* Sum,
              T* Y,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;

    dim3 grid(blocks, batch_dim, head_dim);
    
    if (block_size == 32)
    {
        if(k64)

            fusion_nt_softmax1< true, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, rblk_lut, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        else

            fusion_nt_softmax1<false, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, rblk_lut, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
    } 
    return true;
};


template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax2(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale)
{   

    // printf("%u %u %u %u \n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);

    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;
    
    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head  = NT_Lut[idx_H*szLut + bid];
    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint lut_size_rblk = NN_Lut[idx_H*szNN + idx_M].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + idx_M].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

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
    
    // if(tid == 223 && idx_B == 0 && idx_H == 0 && bid == 0){
    //     printf("%u\n",tid224);
    // }


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

    // C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

    // Arrange 8 tiles horizontally in the X direction: ((tid & 224) >> 1)
    // Add some spacing  to avoid write bank conflicts: (ty << 2)
    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    // if(tid == 3 && bid == 0 && idx_B ==0 && idx_H == 0){
    //     printf("ty, tx, storeC:%u %u %u\n", ty, tx, storC);
    // }

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // if(tid == 3 && bid == 0 && idx_B ==0 && idx_H == 0){
    //     printf("readC:%u \n", readC);
    // }

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];

    __syncthreads();

    float2 c1[8]; 
    bhalf2 c1b;
    float2 c2[8]; 
    bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

    // store((bhalf2*)C, c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    // store((bhalf2*)(C + 16*32), c2[0]);

    uint offset_flag = (idx_B*szBatchRblk)+(idx_H*szHeadRblk)+idx_M;

    // uint ty = tid/16;
    // uint tx = tid%16;

    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    uint inf = 0xff80;
    
    float Xmax[2]; 
    float2 Xval[2];
    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 

    if(tid == 0)
        atomicAdd(rblk_flag+offset_flag, 1);
    
    do{
        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
            
        if(tx == 0){
            store(Max_in_block+offset_in_ms, Xmax[0]);
            store(Max_in_block+offset_in_ms+(16), Xmax[1]);
        }

        __threadfence();

    }while(rblk_flag[offset_flag] != lut_size_rblk);

    float max0 = 0, max16 = 0;
    float Max0 = 0, Max16 = 0;
    float Sum0 = 0, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);

        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);

    }


    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    // xval[0] = to_bhalf(Xval[0]); 

    // __syncthreads();

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    // if(tid == 0 && bid == 0 && idx_B == 0 && idx_H == 0){
    //     printf("%f \n", Sum0);
    // }

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    // xval[1] = to_bhalf(Xval[1]); 

    // __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tx == 0){
        store(Sum_in_rblock+offset_in_ms, Sum0);
        store(Sum_in_rblock+offset_in_ms+16, Sum16);
    }

    if(tid == 0)
        atomicAdd(rblk_flag1+offset_flag, 1);

    do{
        __threadfence();
    }while(rblk_flag1[offset_flag] != lut_size_rblk);


    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul((Xval[0]), rcp_exp_sum0);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul((Xval[1]), rcp_exp_sum16);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);

}

template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax2_warp_level(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale)
{   

    // printf("%u %u %u %u \n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);
    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;
    
    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head  = NT_Lut[idx_H*szLut + bid];
    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint lut_size_rblk = NN_Lut[idx_H*szNN + idx_M].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + idx_M].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

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
    
    // if(tid == 223 && idx_B == 0 && idx_H == 0 && bid == 0){
    //     printf("%u\n",tid224);
    // }


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

    // C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

    // Arrange 8 tiles horizontally in the X direction: ((tid & 224) >> 1)
    // Add some spacing  to avoid write bank conflicts: (ty << 2)
    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    // if(tid == 3 && bid == 0 && idx_B ==0 && idx_H == 0){
    //     printf("ty, tx, storeC:%u %u %u\n", ty, tx, storC);
    // }

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // if(tid == 3 && bid == 0 && idx_B ==0 && idx_H == 0){
    //     printf("readC:%u \n", readC);
    // }

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];

    __syncthreads();

    float2 c1[8]; 
    bhalf2 c1b;
    float2 c2[8]; 
    bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

    // store((bhalf2*)C, c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    // store((bhalf2*)(C + 16*32), c2[0]);

    uint offset_flag = (idx_B*szBatchRblk)+(idx_H*szHeadRblk)+idx_M;

    // uint ty = tid/16;
    // uint tx = tid%16;

    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    uint inf = 0xff80;
    
    float Xmax[2]; 
    float2 Xval[2];
    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 

    if(tid == 0)
        atomicAdd(rblk_flag+offset_flag, 1);
    
    do{
        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
            
        if(tx == 0){
            store(Max_in_block+offset_in_ms, Xmax[0]);
            store(Max_in_block+offset_in_ms+(16), Xmax[1]);
        }

        __threadfence();

    }while(rblk_flag[offset_flag] != lut_size_rblk);

    float max0 = 0, max16 = 0;
    float Max0 = 0, Max16 = 0;
    float Sum0 = 0, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);

        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);

    }


    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    // xval[0] = to_bhalf(Xval[0]); 

    // __syncthreads();

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    // if(tid == 0 && bid == 0 && idx_B == 0 && idx_H == 0){
    //     printf("%f \n", Sum0);
    // }

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    // xval[1] = to_bhalf(Xval[1]); 

    // __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tx == 0){
        store(Sum_in_rblock+offset_in_ms, Sum0);
        store(Sum_in_rblock+offset_in_ms+16, Sum16);
    }

    if(tid == 0)
        atomicAdd(rblk_flag1+offset_flag, 1);

    do{
        __threadfence();
    }while(rblk_flag1[offset_flag] != lut_size_rblk);


    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul((Xval[0]), rcp_exp_sum0);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul((Xval[1]), rcp_exp_sum16);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);

}


template <typename T, typename V>
bool fusion_attention2(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
    const  uint* Mask,
          float* Max,
          float* Sum,
              T* Y,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;
    dim3 grid(blocks, batch_dim, head_dim);
    
    if (block_size == 32)
    {
        if(k64){

            fusion_nt_softmax2< true, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        }else{

            fusion_nt_softmax2<false, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        }

            
    } 
    return true;
};



template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax_full(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale, uint blocks, int *d_id_extractor, uint ctx_blks_a)
{

    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x;
    uint idx_B = blockIdx.y;
    uint idx_H = blockIdx.z; 

    // uint seg_id = 0;
    // uint seg_tid = 0;

    // if(tid == 0){
    //     seg_id = atomicAdd(d_id_extractor, 1);
    //     fShare[0] = seg_id;
    // }
    // __syncthreads();

    // seg_tid = (uint)fShare[0];

    // uint bid = blocks - seg_tid%(gridDim.x) -1;
    // // uint bid = seg_tid%gridDim.x;
    // uint idx_B = (seg_tid/gridDim.x)%(gridDim.y);
    // uint idx_H = (seg_tid/(gridDim.x*gridDim.y));

    // uint idx_B = (seg_tid)%gridDim.y;
    // uint idx_H = (seg_tid/gridDim.y)%(gridDim.z);
    // uint bid = (seg_tid/(gridDim.y*gridDim.z));

    // each head can optionally have its own lut
    uint2 lut_head = NT_Lut[idx_H*szLut + bid];

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint lut_size_rblk = NN_Lut[idx_H*szNN + idx_M].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + idx_M].x;
    
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

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


    }while (++loop < loops);

    // asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    // asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bid)   :);
    // asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B) :);
    // asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H) :);

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

    float2 c1[8]; 
    float2 c2[8]; 
    bhalf2 c1b;
    bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    tx = tid % 16; //id_in_group
    ty = tid / 16; //id_group

    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    uint inf = 0xff80;
    
    float Xmax[2]; 
    float2 Xval[2];

    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 
    // uint offset_of_flag = idx_B*szBatchRblk+idx_H*szHeadRblk+idx_M;

    if(tid == 0){
        atomicAdd((rblk_flag+idx_B*szBatchRblk+idx_H*szHeadRblk+idx_M), 1);
    }


    do{

        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));

        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));

        store(Max_in_block+offset_in_ms, Xmax[0]);
        store(Max_in_block+offset_in_ms+16, Xmax[1]);  
    
        __threadfence();

    }while(rblk_flag[idx_B*szBatchRblk+idx_H*szHeadRblk+idx_M] != lut_size_rblk);

    float max0 = 0, max16 = 0;
    float Max0 = 0, Max16 = 0;
    float Sum0 = 0, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);
        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);
    }

    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    xval[0] = to_bhalf(Xval[0]); 

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    xval[1] = to_bhalf(Xval[1]); 

    __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tid == 0)
        atomicAdd((rblk_flag1+idx_B*szBatchRblk+idx_H*szHeadRblk+idx_M), 1);
    
    do{
        
        if(tx == 0){
            store(Sum_in_rblock+offset_in_ms, Sum0);
            store(Sum_in_rblock+offset_in_ms+16, Sum16);
        }
        __threadfence();

    }while(rblk_flag1[idx_B*szBatchRblk+idx_H*szHeadRblk+idx_M] != lut_size_rblk);


    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul(to_float(xval[0]), rcp_exp_sum0);
    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul(to_float(xval[1]), rcp_exp_sum16);
    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);

}

template <typename T, typename V>
bool fusion_attention_full(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
    const  uint* Mask,
          float* Max,
          float* Sum,
              T* Y,
            int* d_id_extractor,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;

    dim3 grid(blocks, batch_dim, head_dim);
    // printf("gird dim : %u %u %u\n", blocks, batch_dim, head_dim);

  
    if (block_size == 32)
    {
        if(k64)

            fusion_nt_softmax_full< true, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale, blocks, d_id_extractor, ctx_blks_a);
        else

            fusion_nt_softmax_full<false, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale, blocks, d_id_extractor, ctx_blks_a);
    } 

    return true;
};


template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax_local(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale)
{
    
    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim


    uint lut_size_rblk = NN_Lut[idx_H*szNN + bid].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + bid].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;
    uint block_offset = fstblock_in_rblk;

    bhalf2 xval[2];
    float max[4];
    uint mask0[2];
    float Xmax[2]; 
    float2 Xval[2];

    uint inf = 0xff80;

    for(int cal_loop = 0; cal_loop < lut_size_rblk; cal_loop++){

    //     // each head can optionally have its own lut
        uint2 lut_head  = NT_Lut[idx_H*szLut + block_offset];

        uint tx = tid % 16;
        uint ty = tid / 16;
        uint k  = tx  * 4;
        uint tid224 = tid & 224;
        uint storAB = (tx*32*4 + ty + tx*2)*4;
        
        uint idx_M = lut_head.x;
        uint idx_N = lut_head.y;

        uint offsetA00 = idx_B*szCtxHeadStateA + (idx_M*32 + ty)*szHeadState + idx_H*szState + k;
        uint offsetB00 = idx_B*szCtxHeadStateB + (idx_N*32 + ty)*szHeadState + idx_H*szState + k;
        uint offsetA16 = offsetA00 + szHeadState*16;
        uint offsetB16 = offsetB00 + szHeadState*16;

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

        float regA[8], regB[4];

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

        float2 c1[8]; 
        bhalf2 c1b;
        float2 c2[8]; 
        bhalf2 c2b;

        for (int i = 0; i < 8; i++)
            c1[i] = *(float2*)&fShare[readC + i*32];

        // Tree reduce
        for (int j = 4; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                c1[i] = ew_add(c1[i], c1[i+j]);
        
        c1b = to_bhalf(c1[0]);

        // if(tid == 0 && bid == 0 && idx_B == 0 && idx_H == 0){
        //     printf("%f %f ", c1[0].x, c1[0].y);
        // }

        // store((bhalf2*)C, c1[0]);

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

        c2b = to_bhalf(c2[0]);

        // if(tid == 0 && bid == 0 && idx_B == 0 && idx_H == 0){
        //     printf("%f %f ", c2[0].x, c2[0].y);
        // }

        xval[0] = c1b;
        xval[1] = c2b;

        Mask += (idx_H * szMask + block_offset * BSIZE + ty);
        mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

        Mask += 16;
        mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

        uint bit0  = 1 << ((tx)*2); 
        uint bit1  = bit0 << 1;
        uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+block_offset*BSIZE+ty;
        uint mask; 

        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
    
        max[cal_loop*2] = Xmax[0];
        max[cal_loop*2+1] = Xmax[1];

        block_offset++;

        // if(tid == 0 && bid == 0 && idx_H == 0 && idx_B == 0){
        //     printf("%f %f \n", Xmax[0], Xmax[1]);
        // }

    }
    
    float max0 = -inf, max16 = -inf;
    float Max0 = -inf, Max16 = -inf;
    float Sum0 = 0, Sum16 = 0;
    float sum[4];

    Max0 = fmaxf(max[0], max[2]);
    Max16 = fmaxf(max[1], max[3]);

    for(int j=0;j<lut_size_rblk;j++)
    {

        Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
        Sum0  = ew_sum(Xval[0]);
        xval[0] = to_bhalf(Xval[0]);

        Sum0 += shfl_xor(Sum0, 8);
        Sum0 += shfl_xor(Sum0, 4);
        Sum0 += shfl_xor(Sum0, 2);
        Sum0 += shfl_xor(Sum0, 1);

        Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
        Sum16  = ew_sum(Xval[1]);
        xval[1] = to_bhalf(Xval[1]);

        Sum16 += shfl_xor(Sum16, 8);
        Sum16 += shfl_xor(Sum16, 4);
        Sum16 += shfl_xor(Sum16, 2);
        Sum16 += shfl_xor(Sum16, 1);

        sum[j*2] = Sum0;
        sum[j*2+1] = Sum16;
    }

    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;
    uint tx = tid % 16;
    uint ty = tid / 16;

    exp_sum0 = sum[0]+sum[2];
    exp_sum16 = sum[1]+sum[3];

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul(to_float(xval[0]), rcp_exp_sum0);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul(to_float(xval[1]), rcp_exp_sum16);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);

}


template <typename T, typename V>
bool fusion_attention_local(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
    const  uint* Mask,
          float* Max,
          float* Sum,
              T* Y,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;

    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;

    dim3 grid(ctx_blks_a, batch_dim, head_dim);
    // printf("%u %u %u \n", ctx_blks_a, batch_dim, head_dim);
    
    if (block_size == 32)
    {
        if(k64){

            fusion_nt_softmax_local< true, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        }
        else

            fusion_nt_softmax_local<false, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
    } 
    return true;
};



template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_nt_softmax2_64(
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale)
{   

   __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;
    
    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head  = NT_Lut[idx_H*szLut + bid];
    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint lut_size_rblk = NN_Lut[idx_H*szNN + idx_M].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + idx_M].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

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

    // C += idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + bid*32*32 + tid*2;

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

    float2 c1[8]; 
    bhalf2 c1b;
    float2 c2[8]; 
    bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

    // store((bhalf2*)C, c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    // store((bhalf2*)(C + 16*32), c2[0]);

    uint offset_flag = (idx_B*szBatchRblk)+(idx_H*szHeadRblk)+idx_M;

    // uint ty = tid/16;
    // uint tx = tid%16;

    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    uint inf = 0xff80;
    
    float Xmax[2]; 
    float2 Xval[2];
    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 

    if(tid == 0)
        atomicAdd(rblk_flag+offset_flag, 1);
    
    do{
        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
            
        if(tx == 0){
            store(Max_in_block+offset_in_ms, Xmax[0]);
            store(Max_in_block+offset_in_ms+(16), Xmax[1]);
        }

        __threadfence();

    }while(rblk_flag[offset_flag] != lut_size_rblk);

    float max0 = 0, max16 = 0;
    float Max0 = 0, Max16 = 0;
    float Sum0 = 0, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);

        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);

    }


    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    // xval[0] = to_bhalf(Xval[0]); 

    // __syncthreads();

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    // if(tid == 0 && bid == 0 && idx_B == 0 && idx_H == 0){
    //     printf("%f \n", Sum0);
    // }

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    // xval[1] = to_bhalf(Xval[1]); 

    // __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tx == 0){
        store(Sum_in_rblock+offset_in_ms, Sum0);
        store(Sum_in_rblock+offset_in_ms+16, Sum16);
    }

    if(tid == 0)
        atomicAdd(rblk_flag1+offset_flag, 1);

    do{
        __threadfence();
    }while(rblk_flag1[offset_flag] != lut_size_rblk);


    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul((Xval[0]), rcp_exp_sum0);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul((Xval[1]), rcp_exp_sum16);

    store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);


}


template <typename T, typename V>
bool fusion_attention2_64(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
  const  uint64* Mask,
          float* Max,
          float* Sum,
              T* Y,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;
    
    //here need to be checked
    uint loops = CEIL_DIV(state_dim, 64);
    bool k64   = (state_dim & 63) == 0;

    dim3 grid(blocks, batch_dim, head_dim);
    
    if (block_size == 64)
    {
        if(k64){

            fusion_nt_softmax2_64< true, float, float2, 64, uint64><<<grid,1024,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        }else{

            fusion_nt_softmax2_64<false, float, float2, 64, uint64><<<grid,1024,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y,
                                                            szCtxHeadStateA, szCtxHeadStateB, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale);
        }

            
    } 
    return true;
};



template <bool K64, typename T, typename V2, uint BSIZE, typename MASKT>
__global__ void __launch_bounds__(256,3) fusion_kernel( //here dont know if have problem
    const uint2* __restrict__ NT_Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          bhalf*              C,
    const uint2* __restrict__ NN_Lut,
                         int* rblk_flag,
                         int* rblk_flag1, 
    const MASKT* __restrict__ Mask, 
          float*              Max_in_block,
          float*              Sum_in_rblock, 
              T*              Y,    
    const float* __restrict__ E,
          float*              F,
    uint szCtxHeadStateA, uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szHeadBlocks, uint szBatchBlocks, uint szBatchRblk, uint szHeadRblk, uint szHeadRow, uint szBatchRow, 
    uint szLut, uint szNN, uint loops, uint use_mask, uint szMask, float scale, uint ctx_blocks_c, uint gridM, uint gridN)
{   

    __shared__ float fShare[65*32*2];
    char* bShare = (char*)fShare;

    uint tid   = threadIdx.x;
    uint bid   = blockIdx.x; // block id
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    uint2 lut_head  = NT_Lut[idx_H*szLut + bid];
    uint idx_M = lut_head.x;
    uint idx_N = lut_head.y;

    uint idx_M_4_xn = gridM - idx_M;
    uint idx_N_4_xn = tid / 128;

    uint tx = tid % 16;
    uint ty = tid / 16;
    uint k  = tx  * 4;

    uint lut_size_rblk = NN_Lut[idx_H*szNN + idx_M].y;
    uint offset_of_nn  = NN_Lut[idx_H*szNN + idx_M].x;
    uint fstblock_in_rblk = NN_Lut[idx_H*szNN + offset_of_nn].x;

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

    float2 c1[8]; 
    bhalf2 c1b;
    float2 c2[8]; 
    bhalf2 c2b;

    for (int i = 0; i < 8; i++)
        c1[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c1[i] = ew_add(c1[i], c1[i+j]);
    
    c1b = to_bhalf(c1[0]);

    // store((bhalf2*)C, c1[0]);

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

    c2b = to_bhalf(c2[0]);

    bhalf2 xval[2];
    xval[0] = c1b;
    xval[1] = c2b;

    uint offset_flag = (idx_B*szBatchRblk)+(idx_H*szHeadRblk)+idx_M;

    uint mask0[2];

    Mask += (idx_H * szMask + bid * BSIZE + ty);
    mask0[0] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    Mask += 16;
    mask0[1] = use_mask ? (uint)__ldg(Mask) : 0xffffffff;

    uint bit0  = 1 << ((tx)*2); 
    uint bit1  = bit0 << 1;
    uint inf = 0xff80;
    
    float Xmax[2]; 
    float2 Xval[2];
    uint offset_in_ms = idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+bid*BSIZE+ty;
    uint mask; 

    if(tid == 0)
        atomicAdd(rblk_flag+offset_flag, 1);
    
    do{
        mask = mask0[0];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[0].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[0] = ew_max(to_float(xval[0]));

        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 8));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 4));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 2));  
        Xmax[0] = fmaxf(Xmax[0], shfl_xor(Xmax[0], 1));  

        mask = mask0[1];
        asm("{                               \n\t"
            ".reg .pred p0, p1;              \n\t"
            "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
            "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
            "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
            "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
            "}" : "+r"(xval[1].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));


        Xmax[1] = ew_max(to_float(xval[1]));
        
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 8));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 4));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 2));  
        Xmax[1] = fmaxf(Xmax[1], shfl_xor(Xmax[1], 1));  
            
        if(tx == 0){
            store(Max_in_block+offset_in_ms, Xmax[0]);
            store(Max_in_block+offset_in_ms+(16), Xmax[1]);
        }

        __threadfence();

    }while(rblk_flag[offset_flag] != lut_size_rblk);

    float max0 = 0, max16 = 0;
    float Max0 = 0, Max16 = 0;
    float Sum0 = 0, Sum16 = 0;
        
    for(int j=0;j<lut_size_rblk;j++)
    {
        max0 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        Max0 = fmaxf(Max0, max0);

        max16 = Max_in_block[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        Max16 = fmaxf(Max16, max16);

    }


    Xval[0] = ew_ex2(ew_mul(ew_sub(to_float(xval[0]), Max0), scale));
    Sum0  = ew_sum(Xval[0]);
    // xval[0] = to_bhalf(Xval[0]); 

    // __syncthreads();

    Sum0 += shfl_xor(Sum0, 8);
    Sum0 += shfl_xor(Sum0, 4);
    Sum0 += shfl_xor(Sum0, 2);
    Sum0 += shfl_xor(Sum0, 1);

    Xval[1] = ew_ex2(ew_mul(ew_sub(to_float(xval[1]), Max16), scale));
    Sum16  = ew_sum(Xval[1]);
    // xval[1] = to_bhalf(Xval[1]); 

    // __syncthreads();

    Sum16 += shfl_xor(Sum16, 8);
    Sum16 += shfl_xor(Sum16, 4);
    Sum16 += shfl_xor(Sum16, 2);
    Sum16 += shfl_xor(Sum16, 1);

    if(tx == 0){
        store(Sum_in_rblock+offset_in_ms, Sum0);
        store(Sum_in_rblock+offset_in_ms+16, Sum16);
    }

    if(tid == 0)
        atomicAdd(rblk_flag1+offset_flag, 1);

    do{
        __threadfence();

    }while(rblk_flag1[offset_flag] != lut_size_rblk);


    float exp_sum0 = 0, exp_sum16 = 0;
    float rcp_exp_sum0 = 0, rcp_exp_sum16 = 0;
    float sum0 = 0, sum16 = 0;
    float2 y20, y216;

    for(int j=0;j<lut_size_rblk;j++)
    {   
        
        sum0 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty];
        exp_sum0 += sum0; 

        sum16 = Sum_in_rblock[idx_B*szBatchBlocks*BSIZE+idx_H*szHeadBlocks*BSIZE+(fstblock_in_rblk+j)*BSIZE+ty+16];
        exp_sum16 += sum16; 

    }

    rcp_exp_sum0 = ew_rcp(exp_sum0);
    rcp_exp_sum16 = ew_rcp(exp_sum16);

    y20 = ew_mul((Xval[0]), rcp_exp_sum0);

    // store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty)*BSIZE+tx*2)), y20);

    y216 = ew_mul((Xval[1]), rcp_exp_sum16);

    // store((V2*)(Y+(idx_B*szHeadBlocksBlk+idx_H*szBlocksBlk+bid*(BSIZE*BSIZE)+(ty+16)*BSIZE+tx*2)), y216);

    
    uint2 entry = NN_Lut[idx_H*szLut+ctx_blocks_c+bid]; 
    entry.y *= szHeadState * 32;
    entry.x *= 32*32;
    
    uint tid_xn = tid % 128;

    uint txa = tid_xn % 8;
    uint tya = tid_xn / 8;

    uint txb = tid_xn % 16;
    uint tyb = tid_xn / 16;

    uint tid16 = tid_xn & 16;
    uint tid96 = tid_xn & 96;

    uint loadA_xn =     (tid % 2)      * 4*4;
    uint loadB_xn = ((tid_xn / 2) % 8) * 4*4;

    loadA_xn += tid96;

    loadB_xn += tid16 * 64*4;
    loadA_xn += tid16 * 32*4;

    uint storB_xn = (tyb*64 + txb*4) * 4;
    uint storA_xn;

    storA_xn = (txa*32*2 + tya + txa*4) * 4;
    
    uint b = idx_N_4_xn*64 + txb*4;
    uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

    bool inB = K64 || b < szState;

    // zero accumulation registers
    float regC_xn[4][8];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            regC_xn[i][j] = 0.0f;

    // Force compiler to fully compute these prior to loop
    asm("mov.b32 %0, %0;" : "+r"(loadB)   : );
    asm("mov.b32 %0, %0;" : "+r"(storA_xn)   : );
    asm("mov.b32 %0, %0;" : "+r"(storB_xn)   : );
    asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

    float4 b00 = {0.0f}, b08 = {0.0f}, b16 = {0.0f}, b24 = {0.0f};
    entry.y += offsetB;
    
    if (inB)
    {
        b00 = __ldg((const float4*)(E + (entry.y +  0*szHeadState)));
        b08 = __ldg((const float4*)(E + (entry.y +  8*szHeadState)));
        b16 = __ldg((const float4*)(E + (entry.y + 16*szHeadState)));
        b24 = __ldg((const float4*)(E + (entry.y + 24*szHeadState)));
    }
    __syncthreads();

    *(float*)&bShare[storA_xn + (0*32 + 0*16 + 64*32)*4] = y20.x;
    *(float*)&bShare[storA_xn + (1*32 + 0*16 + 64*32)*4] = y20.y;
    // *(float*)&bShare[storA_xn + (2*32 + 0*16 + 64*32)*4] = fa00.z;
    // *(float*)&bShare[storA_xn + (3*32 + 0*16 + 64*32)*4] = fa00.w;

    *(float*)&bShare[storA_xn + (0*32 + 1*16 + 64*32)*4] = y216.x;
    *(float*)&bShare[storA_xn + (1*32 + 1*16 + 64*32)*4] = y216.y;
    // *(float*)&bShare[storA_xn + (2*32 + 1*16 + 64*32)*4] = fa16.z;
    // *(float*)&bShare[storA_xn + (3*32 + 1*16 + 64*32)*4] = fa16.w;

    *(float4*)&bShare[storB_xn +  0*64*4] = b00;
    *(float4*)&bShare[storB_xn +  8*64*4] = b08;
    *(float4*)&bShare[storB_xn + 16*64*4] = b16;
    *(float4*)&bShare[storB_xn + 24*64*4] = b24;
    __syncthreads();

    float regA_xn[4];
    float regB_xn[8];
    #pragma unroll
    for (int j = 0; j < 16; j++)
    {
        // fetch outer product data
        *(float4*)&regA_xn[0] = *(float4*)&bShare[loadA + (32*j + 64*32 + (j/4)*4)*4]; // shift over 4 floats every 4 rows
        *(float4*)&regB_xn[0] = *(float4*)&bShare[loadB + (64*j +  0)*4];
        *(float4*)&regB_xn[4] = *(float4*)&bShare[loadB + (64*j + 32)*4];

        // accumulate outer product
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                regC_xn[i][j] += regA_xn[i] * regB_xn[j];
    }
    
    tid16 = tid_xn & 16;
    tid96 = tid_xn & 96;

    uint tn = (tid_xn / 2) % 8;
    uint tm = ((tid_xn % 2) + (tid96 / 16))*4 + (tid16 / 16);

    bool t16 = tid16 != 0;

    float outC[2][8];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 8; j++)
        {
            float swap = t16 ? regC[2*i + 0][j] : regC[2*i + 1][j];
            outC[i][j] = t16 ? regC[2*i + 1][j] : regC[2*i + 0][j];
            outC[i][j] += shfl_xor(swap, 16);
        }

        
    uint n = idx_N_4_xn*64 + tn*4;
    bool bn00 = K64 || n +  0 < szState;
    bool bn32 = K64 || n + 32 < szState;

    uint offsetC = idx_B*szCtxHeadStateC + (idx_M_4_xn*32 + tm)*szHeadState + idx_H*szState + n;

    store((float4*)(F + (offsetC + szHeadState*0 +  0)), *(float4*)&outC[0][0], 0, bn00);
    store((float4*)(F + (offsetC + szHeadState*0 + 32)), *(float4*)&outC[0][4], 0, bn32);
    store((float4*)(F + (offsetC + szHeadState*2 +  0)), *(float4*)&outC[1][0], 0, bn00);
    store((float4*)(F + (offsetC + szHeadState*2 + 32)), *(float4*)&outC[1][4], 0, bn32);
}


template <typename T, typename V>
bool fusion_attention(CUstream stream,
    const uint2* nt_lut,
    const float* a,
    const float* b,
          bhalf* c,
    const uint2* nn_lut,
            int* rblk_flag,
            int* rblk_flag1,
    const  uint* Mask,
          float* Max,
          float* Sum,
              T* Y,
          float* e,
          float* f,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint ctx_blks_c, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint mask_heads, float scale, uint max_lut)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateA = ctx_blks_a * block_size * szHeadState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;
    uint szCtxHeadStateC = ctx_blks_c * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;
    uint szHeadBlocks    = blocks;
    uint szBatchBlocks   = head_dim * blocks;
    uint szBatchRblk     = head_dim * ctx_blks_a;
    uint szHeadRblk      = ctx_blks_a;
    uint szHeadRow       = block_size * ctx_blks_b;
    uint szBatchRow      = head_dim * szHeadRow;
    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;
    uint szMask = mask_heads > 1 ? blocks * block_size : 0;
    uint szNN = lut_heads > 1 ? lut_dim+ctx_blks_a : 0;
    uint shared = ((max_lut+1)/2)*2*8;
    uint gridM =  ctx_blks_c -1;
    uint gridN = CEIL_DIV(state_dim, 64);
    uint loops = CEIL_DIV(state_dim, 64); 
    bool k64   = (state_dim & 63) == 0;
    dim3 grid(blocks, batch_dim, head_dim);
    
    if (block_size == 32)
    {
        if(k64){
            fusion_kernel< true, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y, e, f,
                                                            szCtxHeadStateA, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale, ctx_blks_c, gridM, gridN);
        }else{

            fusion_kernel<false, float, float2, 32, uint><<<grid,256,0,stream>>>(nt_lut, a, b, c, nn_lut, rblk_flag, rblk_flag1, (const uint*)Mask, Max, Sum, Y, e, f, 
                                                            szCtxHeadStateA, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szHeadBlocks, szBatchBlocks, szBatchRblk, szHeadRblk, szHeadRow, szBatchRow,
                                                            szLut, szNN, loops, Mask != NULL, szMask, scale, ctx_blks_c, gridM, gridN);
        }

            
    } 
    return true;
};



