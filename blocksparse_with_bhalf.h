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

    if(threadIdx.x == 0)
        printf("%f ", c2[0]);

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
}


template <uint UNROLL, uint BLOCKS, uint BSIZE, typename T, typename V2, typename MASKT>
__global__ void __launch_bounds__(1024,BLOCKS) bst_masked_softmax(
    const uint2* __restrict__ Lut,
    const MASKT* __restrict__ Mask,
    const bhalf* __restrict__ X,
              T*              Y,
    uint blocks, uint szLut, uint szMask, uint szHead, uint szBatch, float scale, uint shfl_init, uint max_lut, uint use_mask)

{
    __shared__ float Max[32];
    __shared__ float Sum[32];
    uint64* LutMask64 = (uint64*)&Sum[32];
    uint*   LutMask32 = (uint*)&Sum[32];
    uint*   LutOffset = BSIZE == 64 ? (uint*)&LutMask64[max_lut] : &LutMask32[max_lut];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / BSIZE; // Q dim
    uint idx_q = blockIdx.x % BSIZE; // Q dim
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    Lut  += idx_H * szLut;
    Mask += idx_H * szMask + idx_q * blocks;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
    {
        // Allows non-power of 2 threads to work
        Max[tid] = -FLT_MAX;
        Sum[tid] = 0.0f;
    }

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;
    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < max_lut; i += blockDim.x)
    {
        if (BSIZE == 64)
        {
            uint64 mask = 0;
            if (i < lut_size)
            {
                uint2 entry  = Lut[i];
                uint blk_id  = entry.x;
                LutOffset[i] = blk_id * BSIZE*BSIZE;
                mask = use_mask ? __ldg(Mask + blk_id) : 0xffffffffffffffff;
            }
            LutMask64[i] = mask;
        }
        else
        {
            uint mask = 0;
            if (i < lut_size)
            {
                uint2 entry  = Lut[i];
                uint blk_id  = entry.x;
                LutOffset[i] = blk_id * BSIZE*BSIZE;
                mask = use_mask ? (uint)__ldg(Mask + blk_id) : 0xffffffff;
            }
            LutMask32[i] = mask;
        }
    }
    __syncthreads();

    // trim warps that we know are out of lut range
    if ((tid & (1024-32))*2*UNROLL < lut_size*BSIZE)
    {
        uint lut_idx  = (tid & (1024 - BSIZE/2))*2*UNROLL/BSIZE;
        uint tidx     = (tid % (BSIZE/2))*2;
        uint offset   = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx + LutOffset[lut_idx];
        X += offset;
        asm("mov.b64 %0, %0;" : "+l"(X) : );

        bhalf2 xval[UNROLL];
        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            ew_set(xval[i], 0xff80ff80); //-inf, -inf

            if (lut_idx + i < lut_size)
                xval[i] = __ldg((const bhalf2*)(X + i*BSIZE*BSIZE));
        }

        // split the 64 bit mask by half warp
        uint tid16 = BSIZE == 64 ? (tid & 16)/16 : 0;
        uint bit0  = 1 << (tidx - tid16*32);
        uint bit1  = bit0 << 1;
        uint inf   = 0xff80;
        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
            uint mask = LutMask32[(lut_idx + i)*(BSIZE == 64 ? 2 : 1) + tid16];
            asm("{                               \n\t"
                ".reg .pred p0, p1;              \n\t"
                "setp.eq.u32 p0, %2, 0;          \n\t" // if ((mask & bit0) == 0)
                "setp.eq.u32 p1, %3, 0;          \n\t" // if ((mask & bit1) == 0)
                "@p0 prmt.b32 %0, %0, %1, 0x3254;\n\t" // set -inf to lo bits
                "@p1 prmt.b32 %0, %0, %1, 0x5410;\n\t" // set -inf to hi bits
                "}" : "+r"(xval[i].x) : "r"(inf), "r"(mask & bit0), "r"(mask & bit1));
        }

        // reduce within thread
        float Xmax[UNROLL];
        for (int i = 0; i < UNROLL; i++)
            Xmax[i] = ew_max(to_float(xval[i]));

        float xmax = Xmax[0];
        for (int i = 1; i < UNROLL; i++)
            xmax = fmaxf(Xmax[i], xmax);

        // reduce within warp
        for (int i = 16; i > 0; i >>= 1)
            xmax = fmaxf(xmax, shfl_xor(xmax, i));

        if (blockDim.x > 32)
        {
            // first thread of each warp store to shared
            if ((tid & 31) == 0)
                Max[tid/32] = xmax;
            __syncthreads();
            if (tid < 32)
            {
                // first warp loads all prior reductions
                xmax = Max[tid];
                // reduce within this last warp
                #pragma unroll 1
                for (uint i = shfl_init; i > 0; i >>= 1)
                    xmax = fmaxf(xmax, shfl_xor(xmax, i));
                // final reduction to shared
                Max[tid] = xmax;
            }
            __syncthreads();
            xmax = Max[0];
        }

        // subtract xmax and compute exponent
        float exp_sum = 0;
        for (int i = 0; i < UNROLL; i++)
        {
            // use fast approx math: e**x == 2**(x * log2(e))
            // log2(e) is included in scale factor
            float2 Xval = ew_ex2(ew_mul(ew_sub(to_float(xval[i]), xmax), scale));
            exp_sum    += ew_sum(Xval);
            xval[i]     = to_bhalf(Xval);
        }

        // reduce within warp
        for (int i = 16; i > 0; i >>= 1)
            exp_sum += shfl_xor(exp_sum, i);

        if (blockDim.x > 32)
        {
            // first thread of each warp store to shared
            if ((tid & 31) == 0)
                Sum[tid/32] = exp_sum;
            __syncthreads();

            if (tid < 32)
            {
                // first warp loads all prior reductions
                exp_sum = Sum[tid];
                // reduce within this last warp
                #pragma unroll 1
                for (uint i = shfl_init; i > 0; i >>= 1)
                    exp_sum += shfl_xor(exp_sum, i);
                // final reduction to shared
                Sum[tid] = exp_sum;
            }
            __syncthreads();
            exp_sum = Sum[0];
        }
        float rcp_exp_sum = ew_rcp(exp_sum);
        Y += offset;
        asm("mov.b64 %0, %0;" : "+l"(Y) : );

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
            float2 y2 = ew_mul(to_float(xval[i]), rcp_exp_sum);

            store((V2*)Y, y2, i*BSIZE*BSIZE/2, lut_idx + i < lut_size);
        }
    }
}


template <typename T, typename V>
bool BlocksparseMaskedSoftmax(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const bhalf* x,
              T* y,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale)
{
    uint szLut   = lut_heads  > 1 ? lut_dim  : 0;
    uint szMask  = mask_heads > 1 ? blocks * block_size : 0;
    uint gridQ   = ctx_blks * block_size;
    uint szHead  = blocks * block_size * block_size;
    uint szBatch = head_dim * szHead;
    uint maxK    = max_lut * block_size;
    //cuMemsetD16Async((CUdeviceptr)c, 0, szBatch*batch_dim, stream);

    // combine scaling with fast exp(x) compute
    scale *= LOG2e;

    dim3 grid(gridQ, batch_dim, head_dim);

    uint unroll, threads;
         if (maxK > 1024*16) { unroll = 16; threads = CEIL_DIV(maxK, 32*16*2) * 32; }
    else if (maxK > 1024* 8) { unroll =  8; threads = CEIL_DIV(maxK, 32* 8*2) * 32; }
    else                     { unroll =  4; threads = CEIL_DIV(maxK, 32* 4*2) * 32; }
    uint bshift    = block_size == 64 ? 5 : block_size == 32 ? 4 : block_size == 16 ? 3 : 2;
    uint shfl_init = THREAD_POW2(threads) / 64;
    uint lut_max   = (threads * unroll) >> bshift;
    uint shared    = lut_max * 8;

    if (block_size == 64)
    {
        shared = lut_max * 12;
            if (unroll == 16)
            bst_masked_softmax<16,1,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,64,T,V,uint64><<<grid,threads,shared,stream>>>(lut, (const uint64*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else if (block_size == 32)
    {
            if (unroll == 16)
            bst_masked_softmax<16,1,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,32,T,V,  uint><<<grid,threads,shared,stream>>>(lut, (const   uint*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else if (block_size == 16)
    {
            if (unroll == 16)
            bst_masked_softmax<16,1,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2,16,T,V,ushort><<<grid,threads,shared,stream>>>(lut, (const ushort*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    else
    {
            if (unroll == 16)
            bst_masked_softmax<16,1, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else if (unroll == 8)
            bst_masked_softmax< 8,2, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
        else // (unroll == 4)
            bst_masked_softmax< 4,2, 8,T,V, uchar><<<grid,threads,shared,stream>>>(lut, (const  uchar*)mask, x, y, blocks, szLut, szMask, szHead, szBatch, scale, shfl_init, lut_max, mask != NULL);
    }
    return true;
}


template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128,6) bst_sgemm_32x64x32_xn(
    const uint2* __restrict__ Lut,
    const bhalf* __restrict__ A,
    const float* __restrict__ B,
          float*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    __shared__ float fShare[(33 + 64)*32];
    uint2* Lut2s = (uint2*)&fShare[(33 + 64)*32];
    char* bShare = (char*)&fShare;

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    uint txb = tid % 16;
    uint tyb = tid / 16;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 128)
        {
            uint2 entry = Lut[i];
            entry.x *= 32*32;  // 1024 entries of A per block
            entry.y *= szHeadState*32;   // 32 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint txa = tid % 8;
        uint tya = tid / 8;

        uint tid16 = tid & 16;
        uint tid96 = tid & 96;

        uint loadB = ((tid / 2) % 8) * 4*4;
        uint loadA =  (tid % 2)      * 4*4;

        // each warp handles a quarter of the weights
        loadA += tid96;

        // second half of warp starts 16 rows down
        loadB += tid16 * 64*4;
        loadA += tid16 * 32*4;

        uint storB = (tyb*64 + txb*4) * 4;
        uint storA;
        if (OP_A == OP_T)
            storA = tid * 4*4;
        else
        {
            // Transpose weights on store to shared
            // Avoid bank conflicts by shifting writes over by 4 every 4 rows (+txa*4)
            storA = (txa*32*4 + tya + txa*4) * 4;
            loadA += tid16 * 4; // shift over 4 floats every 4 rows, second half of warp starts 16 rows down
        }

        uint b = idx_N*64 + txb*4;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        // zero accumulation registers
        float regC[4][8];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                regC[i][j] = 0.0f;

        // Force compiler to fully compute these prior to loop
        asm("mov.b32 %0, %0;" : "+r"(loadA)   : );
        asm("mov.b32 %0, %0;" : "+r"(loadB)   : );
        asm("mov.b32 %0, %0;" : "+r"(storA)   : );
        asm("mov.b32 %0, %0;" : "+r"(storB)   : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            //asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];

            const bhalf* pA = add_ptr_u(A, entry.x + offsetA);
            bhalf4 a00 = __ldg((const bhalf4*)(pA +  0*32));
            bhalf4 a16 = __ldg((const bhalf4*)(pA + 16*32));
            float4 b00 = {0.0f}, b08 = {0.0f}, b16 = {0.0f}, b24 = {0.0f};
            entry.y += offsetB;
            if (inB)
            {
                b00 = __ldg((const float4*)(B + (entry.y +  0*szHeadState)));
                b08 = __ldg((const float4*)(B + (entry.y +  8*szHeadState)));
                b16 = __ldg((const float4*)(B + (entry.y + 16*szHeadState)));
                b24 = __ldg((const float4*)(B + (entry.y + 24*szHeadState)));
            }
            __syncthreads();

            float4 fa00 = to_float(a00);
            float4 fa16 = to_float(a16);

            if (OP_A == OP_T)
            {
                *(float4*)&bShare[storA + (0*16*32 + 64*32)*4] = fa00;
                *(float4*)&bShare[storA + (1*16*32 + 64*32)*4] = fa16;
            }
            else
            {
                // transpose the shared store of W
                *(float*)&bShare[storA + (0*32 + 0*16 + 64*32)*4] = fa00.x;
                *(float*)&bShare[storA + (1*32 + 0*16 + 64*32)*4] = fa00.y;
                *(float*)&bShare[storA + (2*32 + 0*16 + 64*32)*4] = fa00.z;
                *(float*)&bShare[storA + (3*32 + 0*16 + 64*32)*4] = fa00.w;

                *(float*)&bShare[storA + (0*32 + 1*16 + 64*32)*4] = fa16.x;
                *(float*)&bShare[storA + (1*32 + 1*16 + 64*32)*4] = fa16.y;
                *(float*)&bShare[storA + (2*32 + 1*16 + 64*32)*4] = fa16.z;
                *(float*)&bShare[storA + (3*32 + 1*16 + 64*32)*4] = fa16.w;
            }

            *(float4*)&bShare[storB +  0*64*4] = b00;
            *(float4*)&bShare[storB +  8*64*4] = b08;
            *(float4*)&bShare[storB + 16*64*4] = b16;
            *(float4*)&bShare[storB + 24*64*4] = b24;
            __syncthreads();

            // computes a 32x64x32 gemm tile with 4x8 register blocking
            float regA[4];
            float regB[8];
            #pragma unroll
            for (int j = 0; j < 16; j++)
            {
                // fetch outer product data
                *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j + 64*32 + (OP_A == OP_T ? 0 : (j/4)*4))*4]; // shift over 4 floats every 4 rows
                *(float4*)&regB[0] = *(float4*)&bShare[loadB + (64*j +  0)*4];
                *(float4*)&regB[4] = *(float4*)&bShare[loadB + (64*j + 32)*4];

                // accumulate outer product
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 8; j++)
                        regC[i][j] += regA[i] * regB[j];
            }


        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        // printf("%3d %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f | %.0f %.0f %.0f %.0f\n", tid,
        //     regC[0][0], regC[0][1], regC[0][2], regC[0][3],
        //     regC[1][0], regC[1][1], regC[1][2], regC[1][3],
        //     regC[2][0], regC[2][1], regC[2][2], regC[2][3],
        //     regC[3][0], regC[3][1], regC[3][2], regC[3][3]);

        tid16 = tid & 16;
        tid96 = tid & 96;

        uint tn =  (tid / 2) % 8;
        uint tm = ((tid % 2) + (tid96 / 16))*4 + (tid16 / 16);

        bool t16 = tid16 != 0;

        float outC[2][8];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8; j++)
            {
                float swap = t16 ? regC[2*i + 0][j] : regC[2*i + 1][j];
                outC[i][j] = t16 ? regC[2*i + 1][j] : regC[2*i + 0][j];
                outC[i][j] += shfl_xor(swap, 16);
            }

        uint n = idx_N*64 + tn*4;
        bool bn00 = N64 || n +  0 < szState;
        bool bn32 = N64 || n + 32 < szState;

        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tm)*szHeadState + idx_H*szState + n;

        store((float4*)(C + (offsetC + szHeadState*0 +  0)), *(float4*)&outC[0][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*0 + 32)), *(float4*)&outC[0][4], 0, bn32);
        store((float4*)(C + (offsetC + szHeadState*2 +  0)), *(float4*)&outC[1][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*2 + 32)), *(float4*)&outC[1][4], 0, bn32);
    }
    else
    {
        uint c       = idx_N*64 + txb*4;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tyb)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            float4 zero = {0.0f};
            *(float4*)&C[offsetC + szHeadState* 0] = zero;
            *(float4*)&C[offsetC + szHeadState* 8] = zero;
            *(float4*)&C[offsetC + szHeadState*16] = zero;
            *(float4*)&C[offsetC + szHeadState*24] = zero;
        }
    }
}

bool bst_sgemm_xn(CUstream stream,
    const uint2* lut,
    const bhalf* a,
    const float* b,
          float* c,
    uint block_size, uint blocks,
    uint batch_dim, uint ctx_blks_b, uint ctx_blks_c, uint head_dim, uint state_dim,
    uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_lut)
{
    uint szState         = state_dim;
    uint szHeadState     = head_dim * szState;
    uint szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;
    uint szCtxHeadStateC = ctx_blks_c * block_size * szHeadState;

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = lut_heads > 1 ? lut_dim : 0;

    // compound gridDim.x with m and n coords
    uint gridN  = CEIL_DIV(state_dim, 64);
    uint gridM  = ctx_blks_c - 1;
    uint gridX  = ctx_blks_c * gridN;
    uint shared = ((max_lut+1)/2)*2*8; // round up to nearest even, 8 bytes per entry
    bool n64    = (state_dim & 63) == 0;

    dim3 grid(gridX, batch_dim, head_dim);
    if (block_size == 32)
    {
        if (op == NN_OP) // NN
        {
            if (n64)
                bst_sgemm_32x64x32_xn<OP_N, true><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
            else
                bst_sgemm_32x64x32_xn<OP_N,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
        else // TN
        {
            if (n64)
                bst_sgemm_32x64x32_xn<OP_T, true><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
            else
                bst_sgemm_32x64x32_xn<OP_T,false><<<grid,128,shared,stream>>>(lut, a, b, c, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic, shift);
        }
    }
    return true;
}

void magicu64(uint d, uint &magic, uint &shift)
{
    // common cases
         if (d == 1) { magic = 1; shift = 0; }
    else if (d == 2) { magic = 1; shift = 1; }
    else if (d == 4) { magic = 1; shift = 2; }
    else if (d == 8) { magic = 1; shift = 3; }
    else
    {
        // 3 is a special case that only ends up in the high bits if the nmax is 0xffffffff
        // we can't use 0xffffffff for all cases as some return a 33 bit magic number
        uint   nbits = d == 3 ?   (2*32)+1 :   (2*31)+1;
        uint64 nmax  = d == 3 ? 0xffffffff : 0x7fffffff;
        uint64 d64   = d;
        uint64 nc    = ((nmax + 1ull) / d64) * d64 - 1ull;

        for (uint p = 0; p < nbits; p++)
        {
            if ((1ull << p) > nc * (d64 - 1ull - ((1ull << p) - 1ull) % d64))
            {
                magic = (uint)(((1ull << p) + d64 - 1ull - ((1ull << p) - 1ull) % d64) / d64);
                shift = magic == 1 ? p : p - 32;
                //printf("div:%u magic:%u shift:%u\n", d, magic, shift);
                return;
            }
        }
    }
}

template <uint UNROLL, uint BLOCKS, uint BSIZE, typename T, typename V2>
__global__ void __launch_bounds__(1024,BLOCKS) bst_masked_softmax_grad(
    const uint2* __restrict__ Lut,
    const     T* __restrict__ DY,
    const     T* __restrict__ Y,
              T*              DX,
    uint szLut, uint szHead, uint szBatch, float scale, uint shfl_init)
{
    __shared__ float Sum[32];
    uint* LutOffset = (uint*)&Sum[32];

    uint tid   = threadIdx.x;
    uint idx_Q = blockIdx.x / BSIZE;
    uint idx_q = blockIdx.x % BSIZE;
    uint idx_B = blockIdx.y; // batch dim
    uint idx_H = blockIdx.z; // head dim

    // each head can optionally have its own lut
    Lut  += idx_H * szLut;
    uint2 lut_head = Lut[idx_Q];

    if (tid < 32)
        Sum[tid] = 0.0f;

    // prefetch the lut data into shared
    uint lut_offset = lut_head.x;
    uint lut_size   = lut_head.y;

    Lut += lut_offset;
    #pragma unroll 1
    for (uint i = tid; i < lut_size; i += blockDim.x)
        LutOffset[i] = Lut[i].x * BSIZE*BSIZE;
    __syncthreads();

    // trim warps that we know are out of lut range
    if ((tid & (1024-32))*2*UNROLL < lut_size*BSIZE)
    {
        uint lut_idx = (tid & (1024 - BSIZE/2))*2*UNROLL/BSIZE;
        uint tidx    = (tid % (BSIZE/2))*2;
        uint offset  = idx_B*szBatch + idx_H*szHead + idx_q*BSIZE + tidx + LutOffset[lut_idx];
        DY += offset;
        Y  += offset;
        asm("mov.b64 %0, %0;" : "+l"(DY) : );
        asm("mov.b64 %0, %0;" : "+l"(Y)  : );

        V2 dy[UNROLL], y[UNROLL];
        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            ew_set(dy[i], 0);
            ew_set( y[i], 0);

            if (lut_idx + i < lut_size)
            {
                dy[i] = __ldg((const V2*)(DY + i*BSIZE*BSIZE));
                 y[i] = __ldg((const V2*)( Y + i*BSIZE*BSIZE));
            }
        }

        // compute dy * y and start reduction
        float sum_dyy = 0.0f;
        for (int i = 0; i < UNROLL; i++)
            sum_dyy += ew_sum(ew_mul(to_float(dy[i]), to_float(y[i])));

        // reduce within warp
        for (int i = 16; i > 0; i >>= 1)
            sum_dyy += shfl_xor(sum_dyy, i);

        if (blockDim.x > 32)
        {
            // first thread of each warp store to shared
            if ((tid & 31) == 0)
                Sum[tid/32] = sum_dyy;
            __syncthreads();

            if (tid < 32)
            {
                // first warp loads all prior reductions
                sum_dyy = Sum[tid];
                // reduce within this last warp
                #pragma unroll 1
                for (uint i = shfl_init; i > 0; i >>= 1)
                    sum_dyy += shfl_xor(sum_dyy, i);
                // final reduction to shared
                Sum[tid] = sum_dyy;
            }
            __syncthreads();
            sum_dyy = Sum[0];
        }
        DX += offset;
        //asm("mov.b64 %0, %0;" : "+l"(DX) : );

        #pragma unroll
        for (uint i = 0; i < UNROLL; i++)
        {
            // dx = (dy - sum_dyy) * y * scale
            float2 dx2 = ew_mul(ew_mul(ew_sub(to_float(dy[i]), sum_dyy), to_float(y[i])), scale);

            store((V2*)DX, dx2, i*BSIZE*BSIZE/2, lut_idx + i < lut_size);
            // asm (
            //     "{                             \n\t"
            //     ".reg .pred p;                 \n\t"
            //     ".reg .s64 DX, offset;         \n\t"
            //     "setp.lt.u32 p, %3, %4;        \n\t"
            //     "mov.b64 offset, {%1, 0};      \n\t"
            //     "add.s64 DX, %0, offset;       \n\t"
            //     "@p st.global.wb.u32 [DX], %2; \n\t"
            //     "}" :: "l"(DX), "r"(i*BSIZE*BSIZE*2), "r"(dx.x), "r"(lut_idx + i), "r"(lut_size));
        }
    }
}



template <typename T, typename V>
bool BlocksparseMaskedSoftmaxGrad(CUstream stream,
    const uint2* lut,
    const     T* dy,
    const     T* y,
              T* dx,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    float scale)
{
    uint szLut   = lut_heads  > 1 ? lut_dim  : 0;
    uint gridQ   = ctx_blks * block_size;
    uint szHead  = blocks * block_size * block_size;
    uint szBatch = head_dim * szHead;
    uint maxK    = max_lut * block_size;
    //cuMemsetD16Async((CUdeviceptr)c, 0, szBatch*batch_dim, stream);

    dim3 grid(gridQ, batch_dim, head_dim);

    uint unroll, threads;
         if (maxK > 1024*16) { unroll = 16; threads = CEIL_DIV(maxK, 32*16*2) * 32; }
    else if (maxK > 1024* 8) { unroll =  8; threads = CEIL_DIV(maxK, 32* 8*2) * 32; }
    else                     { unroll =  4; threads = CEIL_DIV(maxK, 32* 4*2) * 32; }
    uint bshift    = block_size == 64 ? 5 : block_size == 32 ? 4 : block_size == 16 ? 3 : 2;
    uint shfl_init = THREAD_POW2(threads) / 64;
    uint lut_max   = (threads * unroll) >> bshift;
    uint shared    = lut_max * 4;

         if (unroll == 16)
    {
             if (block_size == 64)
            bst_masked_softmax_grad<16,1,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad<16,1,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad<16,1,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad<16,1, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    else if (unroll == 8)
    {
             if (block_size == 64)
            bst_masked_softmax_grad< 8,2,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad< 8,2,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad< 8,2,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad< 8,2, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    else // (unroll == 4)
    {
             if (block_size == 64)
            bst_masked_softmax_grad< 4,2,64,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 32)
            bst_masked_softmax_grad< 4,2,32,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else if (block_size == 16)
            bst_masked_softmax_grad< 4,2,16,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
        else
            bst_masked_softmax_grad< 4,2, 8,T,V><<<grid,threads,shared,stream>>>(lut, dy, y, dx, szLut, szHead, szBatch, scale, shfl_init);
    }
    return true;
};