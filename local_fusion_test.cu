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
#include "validate.h"
#include "common.h"
#include "blocksparse.h"
#include "fusion_blocksparse.h"

int main(){

    const uint batch_dim = 32;
    const uint heads_ = 4;
    const uint head_state_ = 128;
    const uint blk_size_= 32;
    const uint seq_len = 1024;
    const uint state = 512;
    const uint m = batch_dim*seq_len;
    const uint n = state;

    const uint nt_op = 0;
    const uint nn_op = 1; 
    const uint tn_op = 2;
    int count = 0;
    const uint ctx_blks_a_ = seq_len/blk_size_;
    const uint ctx_blks_b_ = seq_len/blk_size_;
    const uint ctx_blks_c_ = seq_len/blk_size_;
    const float scale = 1.0/sqrt(head_state_);

    const uint local_lut_heads = 1;
    const uint local_mask_heads = 1;
    uint local_nn_max = 2;
    uint local_tn_max = 2;
    const uint local_attn_ctx = 32;

    uint local_nt_lut_dim = 0;
    uint local_nn_lut_dim = 0;
    uint local_tn_lut_dim = 0;

    uint *layout = (uint*)malloc(sizeof(uint)*ctx_blks_a_*ctx_blks_a_);

    int extra_diag = local_attn_ctx / blk_size_;

    for(int i=0;i<(ctx_blks_a_*ctx_blks_a_);i++){

        layout[i] = 1;
    }

     for(int i=0;i<ctx_blks_a_;i++){
        for(int j=0;j<ctx_blks_a_;j++){
            
            if((extra_diag!=0)&&((j+extra_diag)<i) || (j>i)){
                layout[i*ctx_blks_a_+j] = 0;
            }

        }
    }

    // for(int i=0;i<ctx_blks_a_;i++){
    //     for(int j=0;j<ctx_blks_a_;j++){

    //         printf("%u ", layout[i*ctx_blks_a_+j]);

    //     }

    //     printf("\n");
    // }

    // printf("\n");

    for(int i=0;i<(ctx_blks_a_*ctx_blks_a_);i++){

        if(layout[i] == 1){
            local_nt_lut_dim++;
        }

    }

    local_nn_lut_dim = local_nt_lut_dim+ctx_blks_a_;
    local_tn_lut_dim = local_nn_lut_dim;


    uint2 *local_nt_lut = (uint2*)malloc(sizeof(uint2)*(local_nt_lut_dim));

    for(int i=0;i<(ctx_blks_a_*ctx_blks_a_);i++){

        if(layout[i] == 1){
            local_nt_lut[count].x = i / ctx_blks_a_;
            local_nt_lut[count].y = i % ctx_blks_a_;
            count++;
        }

    }

    uint2 *d_local_nt_lut;
    cudaMalloc((void**)&d_local_nt_lut, sizeof(uint2)*local_nt_lut_dim);
    cudaMemcpy(d_local_nt_lut, local_nt_lut, sizeof(uint2)*local_nt_lut_dim, cudaMemcpyHostToDevice);


    uint2 *local_nn_lut = (uint2*)malloc(sizeof(uint2)*local_nn_lut_dim);

    for(int i=0;i<local_nn_lut_dim;i++){

        local_nn_lut[i].x = 0;
        local_nn_lut[i].y = 0;
    }

    for(int i=0;i<local_nt_lut_dim;i++){
        
        local_nn_lut[local_nt_lut[i].x].y++;
        local_nn_lut[ctx_blks_a_+i].x = i;
        local_nn_lut[ctx_blks_a_+i].y = local_nt_lut[i].y;
        
    }

    local_nn_lut[0].x = ctx_blks_a_;

    for(int i=1;i<ctx_blks_a_;i++){

        local_nn_lut[i].x = local_nn_lut[i-1].x+local_nn_lut[i-1].y;
    }

    for(int i=0;i<ctx_blks_a_;i++){

        local_nn_max = local_nn_max > local_nn_lut[i].y ? local_nn_max : local_nn_lut[i].y;

    }

    // for(int i=0;i<local_nn_lut_dim;i++){
    //     printf("%u %u \n", local_nn_lut[i].x, local_nn_lut[i].y);
    // }

    uint2 *d_local_nn_lut;
    cudaMalloc((void**)&d_local_nn_lut, sizeof(uint2)*local_nn_lut_dim);
    cudaMemcpy(d_local_nn_lut, local_nn_lut, sizeof(uint2)*local_nn_lut_dim, cudaMemcpyHostToDevice);


    uint local_blocks = local_nt_lut_dim;

    bool *mask = (bool*)malloc(sizeof(bool)*local_blocks*blk_size_*blk_size_);

    for(int i=0;i<(local_blocks*blk_size_*blk_size_);i++){

        mask[i] = 1;
    }

    for(int i=0;i<local_blocks;i++){

        uint i_idx = local_nt_lut[i].x;
        uint j_idx = local_nt_lut[i].y;

        for(int j=0;j<(blk_size_);j++){

            for(int l=0;l<blk_size_;l++){

                if(i_idx == j_idx){
                    if(l>j)
                        mask[i*blk_size_*blk_size_+j*blk_size_+l] = 0;
                }

                uint row_idx = (i_idx)*blk_size_+j;
                uint col_idx = (j_idx)*blk_size_+l;

                if((col_idx>row_idx) || ((col_idx+local_attn_ctx) <= row_idx)){
                    mask[i*(blk_size_*blk_size_)+j*blk_size_+l] = 0;
                }
            }
        }
    }

    unsigned int *local_mask_np = (unsigned int*)malloc(sizeof(unsigned int)*local_blocks*blk_size_);
    unsigned int *local_mask = (unsigned int*)malloc(sizeof(unsigned int)*local_blocks*blk_size_);

    for(int i=0;i<(local_blocks*blk_size_);i++){
        local_mask_np[i] = 0;
        for(int j=0;j<32;j++){
            bool keep = mask[i*blk_size_+j];
            unsigned int temp;
            temp = keep << (j);
            local_mask_np[i] = local_mask_np[i] | temp;  
        }
    }

    for(int i=0; i<local_blocks; i++){
        for(int j=0; j<blk_size_; j++){
            local_mask[j*local_blocks+i] = local_mask_np[i*blk_size_+j];
        }
    }

    unsigned int *sm_mask;
    cudaMalloc((void**)&sm_mask, sizeof(unsigned int)*(local_blocks*blk_size_));
    cudaMemcpy(sm_mask, local_mask_np, sizeof(unsigned int)*(local_blocks*blk_size_), cudaMemcpyHostToDevice);

    unsigned *sm_mask1;
    cudaMalloc((void**)&sm_mask1, sizeof(unsigned)*(local_blocks*blk_size_));
    cudaMemcpy(sm_mask1, local_mask, sizeof(unsigned)*(local_blocks*blk_size_), cudaMemcpyHostToDevice);

    float *a_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    float *b_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    // char q_file[] = "/home/songshuhui/Desktop/Transformer-0523/comparison_precision/q";
    // char k_file[] = "/home/songshuhui/Desktop/Transformer-0523/comparison_precision/k";
    // readbinary(q_file, a_ptr_cpu, m*n);  
    // readbinary(k_file, b_ptr_cpu, m*n);  

    for(int i=0;i<m*n;i++){

        a_ptr_cpu[i] = 1.0;
        b_ptr_cpu[i] = 1.0;

    }


    float *a_ptr, *b_ptr;
    cudaMalloc((void**)&a_ptr, sizeof(float)*m*n);
    cudaMalloc((void**)&b_ptr, sizeof(float)*m*n);

    cudaMemcpy(a_ptr, a_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    

    uint2 *rblk_lutc = (uint2*)malloc(sizeof(uint2)*(local_blocks));
    count = 0;
    for(int i=0;i<ctx_blks_a_;i++){
        for(int j=0;j<local_nn_lut[i].y;j++){
            rblk_lutc[count].x = count;
            rblk_lutc[count].y = i;
            count++;
        }
    }

    
    uint2 *rblk_lut;
    cudaMalloc((void**)&rblk_lut, sizeof(uint2)*local_blocks);
    cudaMemcpy(rblk_lut, rblk_lutc, sizeof(uint2)*(local_blocks), cudaMemcpyHostToDevice);

    float *y_ptr;
    cudaMalloc((void**)&y_ptr, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_*blk_size_));

    // int *rblk_flag_c = (int*)malloc(sizeof(int)*ctx_blks_a_*batch_dim*heads_);
    // for(int i=0;i<(batch_dim*heads_);i++){
    //     for(int j=0;j<ctx_blks_a_;j++){
    //         rblk_flag_c[i*ctx_blks_a_+j] = local_nn_lut[ctx_blks_a_-j-1].y;
    //     }
    // }

    int *rblk_flag;
    cudaMalloc((void**)&rblk_flag, sizeof(int)*ctx_blks_a_*batch_dim*heads_);
    // cudaMemcpy(rblk_flag, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);

    int *rblk_flag1;
    cudaMalloc((void**)&rblk_flag1, sizeof(int)*ctx_blks_a_*batch_dim*heads_);
    // cudaMemcpy(rblk_flag1, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);

    float *Max;
    cudaMalloc((void**)&Max, sizeof(float)*batch_dim*heads_*local_blocks*blk_size_);

    float *Sum;
    cudaMalloc((void**)&Sum, sizeof(float)*batch_dim*heads_*local_blocks*blk_size_);

    float *Maxc = (float*)malloc(sizeof(float)*batch_dim*heads_*local_blocks*blk_size_);
    float *Sumc = (float*)malloc(sizeof(float)*batch_dim*heads_*local_blocks*blk_size_);
    float *yc = (float*)malloc(sizeof(float)*blk_size_*blk_size_*batch_dim*heads_*local_blocks); 
    float *yc1 = (float*)malloc(sizeof(float)*blk_size_*blk_size_*batch_dim*heads_*local_blocks); 

    bhalf *c_ptr;
    cudaMalloc((void**)&c_ptr, sizeof(bhalf)*(batch_dim*heads_*local_blocks*blk_size_*blk_size_));
    bhalf *c_ptr_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*local_blocks*blk_size_*blk_size_));

    float *y1_ptr;
    cudaMalloc((void**)&y1_ptr, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_*blk_size_));

    double time=0, ntime=0, time1=0, time2=0;
    double time_avg = 0, ntime_avg = 0; 
    double time_min = 1000, ntime_min = 1000;


    double time_avg1 = 0, time_avg2 = 0;
    double time_min1 = 1000, time_min2 = 1000;

    struct timeval GET_TIME_START, GET_TIME_END, GET_TIME_START1, GET_TIME_END1;

    CUstream custream;
    cudaStreamCreate(&custream); 

    for(int i=0;i<1000;i++){
        
        cudaDeviceSynchronize();
        fusion_attention_local<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y_ptr,
                        blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                        local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        cudaDeviceSynchronize();
    
        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));

    }
    int loops = 1000;

    for(int i=0;i<loops;i++){
        
        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);

        fusion_attention_local<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y_ptr,
                        blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                        local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        double time = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        time_avg+=time;
        time_min = time_min > time ? time : time_min;

        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));

    }
    
    printf("local_time_avg:%f \n", time_avg/loops);
    printf("local_time_min:%f \n", time_min);


    // cudaMemcpy(yc1, y_ptr, sizeof(float)*(local_blocks*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<1;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         for(int l=0;l<blk_size_;l++){
    //             printf("%f ", yc1[i*blk_size_*blk_size_+j*blk_size_+l]);
    //         }
    //         printf("\n");
    //     }

    //     printf("\n");

    // }

    // return 0;

    time_avg = 0;
    time_min = 1000;

    for(int i=0;i<loops;i++){

        fusion_attention1<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, rblk_lut,sm_mask, Max, Sum, y_ptr,
                    blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                    local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        // cudaMemcpy(rblk_flag, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        // cudaMemcpy(rblk_flag1, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));


    }




    for(int i=0;i<loops;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);

        fusion_attention1<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, rblk_lut, sm_mask, Max, Sum, y_ptr,
                        blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                        local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        double time = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        time_avg+=time;
        time_min = time_min > time ? time : time_min;

        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));

    }

    printf("fusion1_time_avg:%f \n", time_avg/loops);
    printf("fusion1_time_min:%f \n", time_min);
    

    time_avg = 0;
    time_min = 1000;


    for(int i=0;i<loops;i++){

        fusion_attention2<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y1_ptr,
                    blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                    local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        // cudaMemcpy(rblk_flag, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        // cudaMemcpy(rblk_flag1, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));


    }


    for(int i=0;i<loops;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);

        fusion_attention2<float, float2>(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, d_local_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y1_ptr,
                        blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                        local_lut_heads, local_nt_lut_dim, local_mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        double time = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        time_avg+=time;
        time_min = time_min > time ? time : time_min;

        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*local_blocks*blk_size_));

    }

    printf("fusion2_time_avg:%f \n", time_avg/loops);
    printf("fusion2_time_min:%f \n", time_min);
  
    // cudaMemcpy(yc, y1_ptr, sizeof(float)*(local_blocks*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<1;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         for(int l=0;l<blk_size_;l++){
    //             printf("%f %f\n", yc[i*blk_size_*blk_size_+j*blk_size_+l], yc1[i*blk_size_*blk_size_+j*blk_size_+l]);
    //         }
    //         printf("\n");
    //     }

    //     printf("\n");

    // }
    // return 0;

    for(int i=0;i<1000;i++){

        bst_sgemm_nt(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, local_lut_heads, local_nt_lut_dim);

        BlocksparseMaskedSoftmax<float,float2>(custream, d_local_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, local_blocks, batch_dim, heads_, ctx_blks_a_, local_lut_heads, local_nn_lut_dim, local_nn_max, local_mask_heads, scale);

    }
    
    for(int i=0;i<1000;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START1), NULL);
        gettimeofday(&(GET_TIME_START), NULL);
    
        bst_sgemm_nt(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, local_lut_heads, local_nt_lut_dim);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);      

        BlocksparseMaskedSoftmax<float,float2>(custream, d_local_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, local_blocks, batch_dim, heads_, ctx_blks_a_, local_lut_heads, local_nn_lut_dim, local_nn_max, local_mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        gettimeofday(&(GET_TIME_END1), NULL);


        time2 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
        ntime = (GET_TIME_END1.tv_sec - GET_TIME_START1.tv_sec) * 1000.0 + (GET_TIME_END1.tv_usec - GET_TIME_START1.tv_usec) / 1000.0;

        ntime_avg+=ntime;
        ntime_min = ntime_min > ntime ? ntime : ntime_min;

        time_avg1+=time1;
        time_min1 = time_min1 > time1 ? time1 : time_min1;

        time_avg2+=time2;
        time_min2 = time_min2 > time2 ? time2 : time_min2;

    }

    printf("sddmm_time_avg:%f \n", time_avg1/1000.0);
    printf("sddmm_time_min:%f \n", time_min1);

    printf("sfmx_time_avg:%f \n", time_avg2/1000.0);
    printf("sfmx_time_min:%f \n", time_min2);

    printf("nofusion_time_avg:%f \n", ntime_avg/1000.0);
    printf("nofusion_time_min:%f \n", ntime_min);

    // ntime_avg = 0;
    // ntime_min = 1000;

    // time_avg1 = 0;
    // time_min1 = 1000;

    // time_avg2 = 0;
    // time_min2 = 1000;

    // for(int i=0;i<1000;i++){

    //     bst_sgemm_nt_nosl(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, local_lut_heads, local_nt_lut_dim);

    //     BlocksparseMaskedSoftmax_nosl<float,float2>(custream, d_local_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, local_blocks, batch_dim, heads_, ctx_blks_a_, local_lut_heads, local_nn_lut_dim, local_nn_max, local_mask_heads, scale);

    // }
    
    // for(int i=0;i<1000;i++){

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START1), NULL);
    //     gettimeofday(&(GET_TIME_START), NULL);
    
    //     bst_sgemm_nt_nosl(custream, d_local_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, local_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, local_lut_heads, local_nt_lut_dim);

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_END), NULL);
    //     time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START), NULL);      

    //     BlocksparseMaskedSoftmax_nosl<float,float2>(custream, d_local_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, local_blocks, batch_dim, heads_, ctx_blks_a_, local_lut_heads, local_nn_lut_dim, local_nn_max, local_mask_heads, scale);

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_END), NULL);
    //     gettimeofday(&(GET_TIME_END1), NULL);


    //     time2 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
    //     ntime = (GET_TIME_END1.tv_sec - GET_TIME_START1.tv_sec) * 1000.0 + (GET_TIME_END1.tv_usec - GET_TIME_START1.tv_usec) / 1000.0;

    //     ntime_avg+=ntime;
    //     ntime_min = ntime_min > ntime ? ntime : ntime_min;

    //     time_avg1+=time1;
    //     time_min1 = time_min1 > time1 ? time1 : time_min1;

    //     time_avg2+=time2;
    //     time_min2 = time_min2 > time2 ? time2 : time_min2;

    // }

    // printf("sddmm_time_avg:%f \n", time_avg1/1000.0);
    // printf("sddmm_time_min:%f \n", time_min1);

    // printf("sfmx_time_avg:%f \n", time_avg2/1000.0);
    // printf("sfmx_time_min:%f \n", time_min2);

    // printf("nofusion_time_avg:%f \n", ntime_avg/1000.0);
    // printf("nofusion_time_min:%f \n", ntime_min);

    return 0;


}