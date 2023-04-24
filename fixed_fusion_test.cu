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

using namespace std;

int main(){

    const uint batch_dim = 32;
    const uint heads_ = 4;
    const uint head_state_ = 128;
    const uint blk_size_= 32;
    const uint seq_len = 320;
    const uint state = 512;
    const uint m = batch_dim*seq_len;
    const uint n = state;
    const uint nn_op = 1; 
    const uint nt_op = 0;
    const uint tn_op = 2;
    int count = 0;
    uint ctx_blks_a_ = seq_len/blk_size_;
    uint ctx_blks_b_ = seq_len/blk_size_;
    uint ctx_blks_c_ = seq_len/blk_size_;
    const float scale = 1.0/sqrt(head_state_);


    const uint fixed_lut_heads = 4;
    const uint fixed_mask_heads= 4;
    uint fixed_nn_max = 0;
    uint fixed_tn_max = 0;
    const uint local_attn_ctx = 128;
    uint stride = local_attn_ctx/blk_size_;


    uint fixed_nt_lut_dim;
    uint fixed_nn_lut_dim;
    uint fixed_tn_lut_dim;

    struct timeval GET_TIME_START, GET_TIME_END, GET_TIME_START1, GET_TIME_END1;
    
    uint *layout = (uint*)malloc(sizeof(uint)*ctx_blks_a_*ctx_blks_a_*heads_);

    for(int i=0;i<(heads_*ctx_blks_a_*ctx_blks_a_);i++){
        layout[i] = 0;
    }

    int *indices = (int*)malloc(sizeof(int)*heads_);
    for(int i=0;i<heads_;i++){
        indices[i] = stride - (i + 1);
    }

    for(int i=0;i<heads_;i++){

        for(int j=0;j<ctx_blks_a_;j++){

            for(int k=indices[i];k<ctx_blks_a_;k+=stride){

                layout[i*(ctx_blks_a_*ctx_blks_a_)+j*ctx_blks_a_+k] = 1;
            }

        }

    }
 

    for(int i=0;i<heads_;i++){

        for(int j=0;j<ctx_blks_a_;j++){

            int row = j / stride;

            int right = ((row+1)*stride) > ctx_blks_a_ ? (ctx_blks_a_) : ((row+1)*stride);

            for(int k=(row*stride);k<right;k++){

                layout[i*(ctx_blks_a_*ctx_blks_a_) + j*ctx_blks_a_ + k] = 1;
                
            }
        }
    }

    for(int i=0;i<heads_;i++){

        for(int j=0;j<ctx_blks_a_;j++){

            for(int k=0;k<ctx_blks_a_;k++){

                if(j < k){

                    layout[i*(ctx_blks_a_*ctx_blks_a_)+j*ctx_blks_a_+k] = 0;
                }
            }

        }
    }


    for(int i=0;i<(ctx_blks_a_*ctx_blks_a_);i++){

        if(layout[i] == 1){
            fixed_nt_lut_dim++;
        }

    }
    printf("test0\n"); 

    fixed_nn_lut_dim = fixed_nt_lut_dim+ctx_blks_a_;
    fixed_tn_lut_dim = fixed_nt_lut_dim+ctx_blks_a_;

    uint2 *fixed_nt_lut = (uint2*)malloc(sizeof(uint2)*fixed_nt_lut_dim*heads_);

    for(int i=0;i<fixed_nt_lut_dim;i++){
        fixed_nt_lut[i].x = 0;
        fixed_nt_lut[i].y = 0;
    }

    printf("test1\n"); 

    for(int i=0;i<(heads_*ctx_blks_a_);i++){

        for(int j=0;j<ctx_blks_a_;j++){

            if(layout[i*ctx_blks_a_+ j] == 1){

                fixed_nt_lut[count].x = i%(ctx_blks_a_);
                fixed_nt_lut[count].y = j;
                count++;

            }
        }
    }

    printf("test2\n"); 
     

    uint2 *d_fixed_nt_lut;
    cudaMalloc((void**)&d_fixed_nt_lut, sizeof(uint2)*(fixed_nt_lut_dim*heads_));
    cudaMemcpy(d_fixed_nt_lut, fixed_nt_lut, sizeof(uint2)*(fixed_nt_lut_dim*heads_), cudaMemcpyHostToDevice);

    uint2 *fixed_nn_lut = (uint2*)malloc(sizeof(uint2)*(fixed_nn_lut_dim*heads_));

    for(int i=0;i<(heads_* fixed_nn_lut_dim);i++){
        
        fixed_nn_lut[i].x = 0;
        fixed_nn_lut[i].y = 0;

    }

    for(int i=0;i<heads_;i++){

        for(int j=0;j<fixed_nt_lut_dim;j++){
            
            fixed_nn_lut[i*fixed_nn_lut_dim+fixed_nt_lut[j].x].y++;

            fixed_nn_lut[i*fixed_nn_lut_dim+ctx_blks_a_+j].x = j;
            fixed_nn_lut[i*fixed_nn_lut_dim+ctx_blks_a_+j].y = fixed_nt_lut[j].y;
            
        }
    }

    for(int i=0;i<heads_;i++){

        fixed_nn_lut[i*fixed_nn_lut_dim].x = ctx_blks_a_;

        for(int j=1;j<ctx_blks_a_;j++){

            fixed_nn_lut[i*fixed_nn_lut_dim+j].x = fixed_nn_lut[i*fixed_nn_lut_dim+(j-1)].y + fixed_nn_lut[i*fixed_nn_lut_dim+(j-1)].x;

        }
    }

    for(int i=0;i<ctx_blks_a_;i++){

        fixed_nn_max = fixed_nn_max > fixed_nn_lut[i].y ? fixed_nn_max : fixed_nn_lut[i].y; 

    }

    uint2 *d_fixed_nn_lut;
    cudaMalloc((void**)&d_fixed_nn_lut, sizeof(uint2)*(fixed_nn_lut_dim*heads_));
    cudaMemcpy(d_fixed_nn_lut, fixed_nn_lut, sizeof(uint2)*(heads_*fixed_nn_lut_dim), cudaMemcpyHostToDevice);

    uint fixed_blocks = fixed_nt_lut_dim;


    bool *mask = (bool*)malloc(sizeof(bool)*fixed_blocks*blk_size_*blk_size_*heads_);

    for(int l = 0;l<(heads_*fixed_blocks);l++){
        
        if((fixed_nt_lut[l].x) == (fixed_nt_lut[l].y)){

            for(int i=0;i<blk_size_;i++){

                for(int j=0;j<blk_size_;j++){

                    if(i>=j){
                        mask[l*(blk_size_*blk_size_)+i*blk_size_+j] = 1;
                    }else{
                        mask[l*(blk_size_*blk_size_)+i*blk_size_+j] = 0;
                    }
                }
            }
        }else{

            for(int i=0;i<(blk_size_*blk_size_);i++){

                mask[(l*blk_size_*blk_size_) + i] = 1;
            }
        }
        
    }



    unsigned int *fixed_mask_np = (unsigned int*)malloc(sizeof(unsigned int)*fixed_blocks*blk_size_*heads_);
    unsigned int *fixed_mask = (unsigned int*)malloc(sizeof(unsigned int)*fixed_blocks*blk_size_*heads_);

    for(int i=0;i<(heads_*fixed_blocks*blk_size_);i++){
        fixed_mask_np[i] = 0;
        for(int j=0;j<32;j++){
            bool keep = mask[i*blk_size_+j];
            unsigned int temp;
            temp = keep << (j);
            fixed_mask_np[i] = fixed_mask_np[i] | temp;  
        }
    }

    for(int i=0;i<heads_;i++){
        for(int j=0; j<fixed_blocks; j++){
            for(int l=0; l<blk_size_; l++){
                fixed_mask[i*(fixed_blocks*blk_size_)+l*fixed_blocks+j] = fixed_mask_np[i*(fixed_blocks*blk_size_)+j*blk_size_+l];
            }
        }
    }

    unsigned *sm_mask;
    cudaMalloc((void**)&sm_mask, sizeof(unsigned)*(heads_*fixed_blocks*blk_size_));
    cudaMemcpy(sm_mask, fixed_mask_np, sizeof(unsigned)*(heads_*fixed_blocks*blk_size_), cudaMemcpyHostToDevice);

    unsigned *sm_mask1;
    cudaMalloc((void**)&sm_mask1, sizeof(unsigned)*(heads_*fixed_blocks*blk_size_));
    cudaMemcpy(sm_mask1, fixed_mask, sizeof(unsigned)*(heads_*fixed_blocks*blk_size_), cudaMemcpyHostToDevice);


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

    uint2 *rblk_lutc = (uint2*)malloc(sizeof(uint2)*(heads_*fixed_blocks));

    count = 0;
    for(int l=0;l<heads_;l++){
        for(int i=0;i<(ctx_blks_a_);i++){
            for(int j=0;j<fixed_nn_lut[l*fixed_nn_lut_dim+i].y;j++){
                rblk_lutc[count].x = count;
                rblk_lutc[count].y = i;
                count++;
            }
        }
        
    }

    uint2 *rblk_lut;
    cudaMalloc((void**)&rblk_lut, sizeof(uint2)*heads_*fixed_blocks);
    cudaMemcpy(rblk_lut, rblk_lutc, sizeof(uint2)*(fixed_blocks*heads_), cudaMemcpyHostToDevice);

    int *rblk_flag_c = (int*)malloc(sizeof(int)*ctx_blks_a_*batch_dim*heads_);

    int *rblk_flag;
    cudaMalloc((void**)&rblk_flag, sizeof(int)*ctx_blks_a_*batch_dim*heads_);
    cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));

    int *rblk_flag1;
    cudaMalloc((void**)&rblk_flag1, sizeof(int)*ctx_blks_a_*batch_dim*heads_);
    cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));


    float *Max;
    cudaMalloc((void**)&Max, sizeof(float)*batch_dim*heads_*fixed_blocks*blk_size_);

    float *Sum;
    cudaMalloc((void**)&Sum, sizeof(float)*batch_dim*heads_*fixed_blocks*blk_size_);

    float *Maxc = (float*)malloc(sizeof(float)*batch_dim*heads_*fixed_blocks*blk_size_);
    float *Sumc = (float*)malloc(sizeof(float)*batch_dim*heads_*fixed_blocks*blk_size_);
    float *yc = (float*)malloc(sizeof(float)*blk_size_*blk_size_*batch_dim*heads_*fixed_blocks); 

    bhalf *c_ptr;
    cudaMalloc((void**)&c_ptr, sizeof(bhalf)*(batch_dim*heads_*fixed_blocks*blk_size_*blk_size_));
    bhalf *c_ptr_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*fixed_blocks*blk_size_*blk_size_));

    float *y_ptr;
    cudaMalloc((void**)&y_ptr, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_*blk_size_));

    float *y1_ptr;
    cudaMalloc((void**)&y1_ptr, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_*blk_size_));

    double time=0, ntime=0, time1=0, time2=0;
    double time_avg = 0, ntime_avg = 0; 
    double time_min = 1000, ntime_min = 1000;


    double time_avg1 = 0, time_avg2 = 0;
    double time_min1 = 1000, time_min2 = 1000;

    CUstream custream;
    cudaStreamCreate(&custream); 
    int loops = 100;

    
    for(int i=0;i<loops;i++){

        fusion_attention1<float, float2>(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, d_fixed_nn_lut, rblk_flag, rblk_flag1, rblk_lut, sm_mask, Max, Sum, y_ptr,
                            blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                            fixed_lut_heads, fixed_nt_lut_dim, fixed_mask_heads, scale);

        // cudaMemcpy(rblk_flag, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        // cudaMemcpy(rblk_flag1, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
        
    }

    

    for(int i=0;i<loops;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);

        fusion_attention1<float, float2>(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, d_fixed_nn_lut, rblk_flag, rblk_flag1, rblk_lut, sm_mask, Max, Sum, y_ptr,
                            blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
                            fixed_lut_heads, fixed_nt_lut_dim, fixed_mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);

        time = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
        time_avg += time;
        time_min = time_min > time ? time : time_min;

        cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
        cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
        cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
    }

    printf("fusion1_time_avg:%f \n", time_avg/loops);
    printf("fusion1_time_min:%f \n", time_min);

    // time_avg = 0;
    // time_min = 1000;

    // for(int i=0;i<loops;i++){

    //     fusion_attention2<float, float2>(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, d_fixed_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y_ptr,
    //                         blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
    //                         fixed_lut_heads, fixed_nt_lut_dim, fixed_mask_heads, scale);

    //     // cudaMemcpy(rblk_flag, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
    //     // cudaMemcpy(rblk_flag1, rblk_flag_c, sizeof(int)*(ctx_blks_a_*batch_dim*heads_), cudaMemcpyHostToDevice);
    //     cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
    //     cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
        
    // }


    // for(int i=0;i<loops;i++){

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START), NULL);

    //     fusion_attention2<float, float2>(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, d_fixed_nn_lut, rblk_flag, rblk_flag1, sm_mask, Max, Sum, y_ptr,
    //                         blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
    //                         fixed_lut_heads, fixed_nt_lut_dim, fixed_mask_heads, scale);

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_END), NULL);
        
    //     time = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
    //     time_avg += time;
    //     time_min = time_min > time ? time : time_min;

        
    //     cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));
    //     cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_));

    // }

    // printf("fusion2_time_avg:%f \n", time_avg/loops);
    // printf("fusion2_time_min:%f \n", time_min);

//    return 0;


    for(int i=0;i<loops;i++){

        bst_sgemm_nt(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, fixed_lut_heads, fixed_nt_lut_dim);

        BlocksparseMaskedSoftmax<float,float2>(custream, d_fixed_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, fixed_blocks, batch_dim, heads_, ctx_blks_a_, fixed_lut_heads, fixed_nn_lut_dim, fixed_nn_max, fixed_mask_heads, scale);


    }
    
    for(int i=0;i<loops;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START1), NULL);
        gettimeofday(&(GET_TIME_START), NULL);
    
        bst_sgemm_nt(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, fixed_lut_heads, fixed_nt_lut_dim);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);      

        BlocksparseMaskedSoftmax<float,float2>(custream, d_fixed_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, fixed_blocks, batch_dim, heads_, ctx_blks_a_, fixed_lut_heads, fixed_nn_lut_dim, fixed_nn_max, fixed_mask_heads, scale);

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


    printf("sddmm_time_avg:%f \n", time_avg1/loops);
    printf("sddmm_time_min:%f \n", time_min1);

    printf("sfmx_time_avg:%f \n", time_avg2/loops);
    printf("sfmx_time_min:%f \n", time_min2);

    printf("nofusion_time_avg:%f \n", ntime_avg/loops);
    printf("nofusion_time_min:%f \n", ntime_min);


    // ntime_avg = 0;
    // ntime_min = 1000;

    // time_avg1 = 0;
    // time_min1 = 1000;

    // time_avg2 = 0;
    // time_min2 = 1000;

    
    // for(int i=0;i<1000;i++){

    //     bst_sgemm_nt_nosl(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, fixed_lut_heads, fixed_nt_lut_dim);

    //     BlocksparseMaskedSoftmax_nosl<float,float2>(custream, d_fixed_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, fixed_blocks, batch_dim, heads_, ctx_blks_a_, fixed_lut_heads, fixed_nn_lut_dim, fixed_nn_max, fixed_mask_heads, scale);


    // }
    
    // for(int i=0;i<1000;i++){

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START1), NULL);
    //     gettimeofday(&(GET_TIME_START), NULL);
    
    //     bst_sgemm_nt_nosl(custream, d_fixed_nt_lut, a_ptr, b_ptr, c_ptr, blk_size_, fixed_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, fixed_lut_heads, fixed_nt_lut_dim);

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_END), NULL);
    //     time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START), NULL);      

    //     BlocksparseMaskedSoftmax_nosl<float,float2>(custream, d_fixed_nn_lut, sm_mask1, c_ptr, y1_ptr, blk_size_, fixed_blocks, batch_dim, heads_, ctx_blks_a_, fixed_lut_heads, fixed_nn_lut_dim, fixed_nn_max, fixed_mask_heads, scale);

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

    // printf("sddmm_time_avg_nosl:%f \n", time_avg1/1000.0);
    // printf("sddmm_time_min_nosl:%f \n", time_min1);

    // printf("sfmx_time_avg_nosl:%f \n", time_avg2/1000.0);
    // printf("sfmx_time_min_nosl:%f \n", time_min2);

    // printf("nofusion_time_avg_nosl:%f \n", ntime_avg/1000.0);
    // printf("nofusion_time_min_nosl:%f \n", ntime_min);


    //check sum/max

    // cudaMemcpy(Maxc, Max, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<(2*fixed_blocks);i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%f ", Maxc[i*blk_size_+j]);
    //     }
    //     printf("\n");
    // }


    // cudaMemcpy(Sumc, Sum, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<1;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%f ", Sumc[i*blk_size_+j]);
    //     }
    //     printf("\n");
    // }

    //check final results
    // cudaMemcpy(yc, y_ptr, sizeof(float)*(batch_dim*heads_*fixed_blocks*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=1;i<12;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%.15f ", yc[i*blk_size_*blk_size_+j*blk_size_]);
    //     }

    //     printf("\n");

    // }


    return 0;
}