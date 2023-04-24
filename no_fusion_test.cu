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
// #include "fusion_blocksparse.h"


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
    const uint ctx_blks_a_ = seq_len/blk_size_;
    const uint ctx_blks_b_ = seq_len/blk_size_;
    const uint ctx_blks_c_ = seq_len/blk_size_;
    const float scale = 1.0/sqrt(head_state_);

    const uint lut_heads = 1;
    uint full_nt_lut_dim = 0;
    uint full_nn_lut_dim = 0;
    uint full_tn_lut_dim = 0;
    const uint mask_heads= 1;

    uint nn_max = 0;
    uint tn_max = 0;

    struct timeval GET_TIME_START, GET_TIME_END, GET_TIME_START1, GET_TIME_END1;

    uint *layout = (uint*)malloc(sizeof(uint)*ctx_blks_a_*ctx_blks_a_);

    for(int i=0;i<(ctx_blks_a_*ctx_blks_a_);i++){
        layout[i] = 0;
    }

    for(int i=0;i<ctx_blks_a_;i++){
        for(int j=0;j<ctx_blks_a_;j++){
            if(i>=j){

                layout[i*ctx_blks_a_+j] = 1;
                full_nt_lut_dim++;

            }
        }
    }

    uint full_blocks = full_nt_lut_dim;

    
    uint2 *nt_lut = (uint2*)malloc(sizeof(uint2)*full_blocks);

    for(int i=0;i<full_blocks;i++){
        nt_lut[i].x =0;
        nt_lut[i].y = 0;
    }

    count = 0;
    for(int i=0;i<ctx_blks_a_;i++){
        for(int j=0;j<ctx_blks_a_;j++){
            if(layout[i*ctx_blks_a_+j] == 1){
                nt_lut[count].x = i;
                nt_lut[count].y = j;
                count++;
            }
        }
    }

    uint2 *l_ptr;
    cudaMalloc((void**)&l_ptr, sizeof(uint2)*full_blocks);
    cudaMemcpy(l_ptr, nt_lut, sizeof(uint2)*full_blocks, cudaMemcpyHostToDevice);

    full_nn_lut_dim = full_nt_lut_dim+ctx_blks_a_;
    full_tn_lut_dim = full_nt_lut_dim+ctx_blks_a_;

    uint2 *nn_lut = (uint2*)malloc(sizeof(uint2)*(full_nn_lut_dim));

    for(int i=0;i<full_nn_lut_dim;i++){
        nn_lut[i].x = 0;
        nn_lut[i].y = 0;
    }

    for(int i=0;i<full_blocks;i++){
        
        nn_lut[i+ctx_blks_a_].x = i;
        nn_lut[i+ctx_blks_a_].y = nt_lut[i].y;
        nn_lut[nt_lut[i].x].y++;        
    }
    
    nn_lut[0].x = ctx_blks_a_;

    for(int i=1;i<ctx_blks_a_;i++){
        nn_lut[i].x = nn_lut[i-1].x + nn_lut[i-1].y;
    }

    for(int i=0; i<ctx_blks_a_+full_blocks; i++){
        printf("%d %d \n", nn_lut[i].x, nn_lut[i].y);   
    }

    for(int i=0;i<ctx_blks_a_;i++){
        nn_max = nn_max > nn_lut[i].y ? nn_max : nn_lut[i].y; 
    }

    uint2 *l2_ptr;
    cudaMalloc((void**)&l2_ptr, sizeof(uint2)*(full_blocks+ctx_blks_a_));
    cudaMemcpy(l2_ptr, nn_lut, sizeof(uint2)*(full_blocks+ctx_blks_a_), cudaMemcpyHostToDevice);

    bool *mask_cpu = (bool*)malloc(sizeof(bool)*full_blocks*blk_size_*blk_size_);
    
    for(int l = 0;l< full_blocks;l++){
        if((nt_lut[l].x) == (nt_lut[l].y)){
            for(int i=0;i<blk_size_;i++){
                for(int j=0;j<blk_size_;j++){
                    if(i>=j){
                        mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 1;
                    }else{
                        mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 0;
                    }
                }
            }
        }else{
            for(int i=0;i<(blk_size_*blk_size_);i++){
                mask_cpu[(l*blk_size_*blk_size_)+ i] = 1;
            }
        }
        
    }

    unsigned int *mask_np = (unsigned int*)malloc(sizeof(unsigned int)*full_blocks*blk_size_);
    unsigned int *mask = (unsigned int*)malloc(sizeof(unsigned int)*full_blocks*blk_size_);

    for(int i=0;i<(full_blocks*blk_size_);i++){
        mask_np[i] = 0;
        for(int j=0;j<32;j++){
            bool keep = mask_cpu[i*blk_size_+j];
            unsigned int temp;
            temp = keep << (j);
            mask_np[i] = mask_np[i] | temp;
            
        }
    }

    for(int i=0; i<full_blocks; i++){
        for(int j=0; j<blk_size_; j++){
            mask[j*full_blocks+i] = mask_np[i*blk_size_+j];
        }
    }

    unsigned int *m_ptr;
    cudaMalloc((void**)&m_ptr, sizeof(unsigned int)*full_blocks*blk_size_);
    cudaMemcpy(m_ptr, mask_np, sizeof(unsigned int)*full_blocks*blk_size_, cudaMemcpyHostToDevice);

    unsigned int *m1_ptr;
    cudaMalloc((void**)&m1_ptr, sizeof(unsigned int)*full_blocks*blk_size_);
    cudaMemcpy(m1_ptr, mask, sizeof(unsigned int)*full_blocks*blk_size_, cudaMemcpyHostToDevice);
    

    float *a_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    float *b_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a_ptr_cpu[i*n+j] = 1;
            b_ptr_cpu[i*n+j] = 1;
        }
    }

    float *a_ptr, *b_ptr;
    cudaMalloc((void**)&a_ptr, sizeof(float)*m*n);
    cudaMalloc((void**)&b_ptr, sizeof(float)*m*n);

    cudaMemcpy(a_ptr, a_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    float *y_ptr;
    cudaMalloc((void**)&y_ptr, sizeof(float)*(batch_dim*heads_*full_blocks*blk_size_*blk_size_));

    double time_avg1 = 0;
    double time_min1 = 10000;

    double time_avg2 = 0;
    double time_min2 = 10000;

    double time_avg = 0;
    double time_min = 10000;

    bhalf *c_ptr;
    cudaMalloc((void**)&c_ptr, sizeof(bhalf)*(batch_dim*heads_*full_blocks*blk_size_*blk_size_));

    bhalf *c_ptr_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*full_blocks*blk_size_*blk_size_));

    float *y1_ptr;
    cudaMalloc((void**)&y1_ptr, sizeof(float)*(batch_dim*heads_*full_blocks*blk_size_*blk_size_));

    bhalf *y2_ptr;
    cudaMalloc((void**)&y2_ptr, sizeof(bhalf)*(batch_dim*heads_*full_blocks*blk_size_*blk_size_));

    CUstream custream;
    cudaStreamCreate(&custream); 

    for(int i=0;i<1000;i++){

        bst_sgemm_nt(custream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, full_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, full_nt_lut_dim);

        BlocksparseMaskedSoftmax<bhalf, bhalf2>(custream, l2_ptr, m1_ptr, c_ptr, y2_ptr, blk_size_, full_blocks, batch_dim, heads_, ctx_blks_a_, lut_heads, full_nn_lut_dim, nn_max, mask_heads, scale);

    }
    
    for(int i=0;i<1000;i++){

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START1), NULL);
        gettimeofday(&(GET_TIME_START), NULL);

        bst_sgemm_nt(custream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, full_blocks, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, full_nt_lut_dim);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        double time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_START), NULL);      

        BlocksparseMaskedSoftmax<bhalf,bhalf2>(custream, l2_ptr, m1_ptr, c_ptr, y2_ptr, blk_size_, full_blocks, batch_dim, heads_, ctx_blks_a_, lut_heads, full_nn_lut_dim, nn_max, mask_heads, scale);

        cudaDeviceSynchronize();
        gettimeofday(&(GET_TIME_END), NULL);
        gettimeofday(&(GET_TIME_END1), NULL);


        double time2 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
        double time = (GET_TIME_END1.tv_sec - GET_TIME_START1.tv_sec) * 1000.0 + (GET_TIME_END1.tv_usec - GET_TIME_START1.tv_usec) / 1000.0;

        time_avg+=time;
        time_min = time_min > time ? time : time_min;

        time_avg1+=time1;
        time_min1 = time_min1 > time1 ? time1 : time_min1;

        time_avg2+=time2;
        time_min2 = time_min2 > time2 ? time2 : time_min2;

    }

    printf("time_avg:%f \n", time_avg1/1000.0);
    printf("time_min:%f \n", time_min1);

    printf("time_avg:%f \n", time_avg2/1000.0);
    printf("time_min:%f \n", time_min2);

    printf("time_avg:%f \n", time_avg/1000.0);
    printf("time_min:%f \n", time_min);

    // uint magic_, shift_;

    // uint div = CEIL_DIV(head_state_, 64);
    // magicu64(div, magic_, shift_);

    // float *e_ptr, *f_ptr;
    // cudaMalloc((void**)&e_ptr, sizeof(float)*m*n);
    // cudaMalloc((void**)&f_ptr, sizeof(float)*m*n);

    // float *e_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    // for(int i=0;i<m*n;i++){
    //     e_ptr_cpu[i] = 1.0;
    // }
    // cudaMemcpy(e_ptr, e_ptr_cpu, sizeof(float)*(m*n), cudaMemcpyHostToDevice);

    // float *f_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    // cudaDeviceSynchronize();s
    // bst_sgemm_xn(custream, l2_ptr, y2_ptr, e_ptr, f_ptr, blk_size_, full_blocks, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, full_nn_lut_dim, nn_op, magic_, shift_, nn_max);
    // cudaDeviceSynchronize();

    // cudaMemcpy(f_ptr_cpu, f_ptr, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    // for(int i=0;i<n;i++){
    //     printf("%f ", f_ptr_cpu[i]);
    // }


    return 0;

    // cudaMemcpy(Maxc, Max, sizeof(int)*(batch_dim*heads_*blocks_*blk_size_), cudaMemcpyDeviceToHost);
    // cudaMemcpy(Sumc, Sum, sizeof(int)*(batch_dim*heads_*blocks_*blk_size_), cudaMemcpyDeviceToHost);
    // cudaMemcpy(yc, y_ptr, sizeof(float)*(blocks_*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<(blocks_);i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%f ", Sumc[i*blk_size_+j]);
    //     }
    //     printf(" ----%d \n", i);
    // }

    // for(int i=0;i<(blocks_);i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%f ", Maxc[i*blk_size_+j]);
    //     }
    //     printf(" ----%d \n", i);
    // }


    // for(int i=0;i<10;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%.15f ", yc[i*blk_size_*blk_size_+j*blk_size_]);
    //     }

    //     printf("\n");

    // }


    // cudaMemcpy(c_ptr_cpu, c_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_), cudaMemcpyDeviceToHost);
    // float *softmax_out = (float*)malloc(sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // cudaMemcpy(softmax_out, y1_ptr, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_), cudaMemcpyDeviceToHost);

    // for(int i=0;i<(55*blk_size_);i++){
    //     for(int j=0;j<blk_size_;j++){
    //         if(fabs(softmax_out[i*blk_size_+j]-yc[i*blk_size_+j]) > 0.00001)
    //             printf("%.15f %.15f\n", softmax_out[i*blk_size_+j], yc[i*blk_size_+j]);
    //     }
    //     printf("%d \n", i);
    // }

    // return 0;

    // float *d_ptr_cpu = (float*)malloc(sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // for(int i=0;i<(batch_dim*heads_*blocks_*blk_size_*blk_size_);i++){
    //     d_ptr_cpu[i] = 1;
    // }

    // count = 0;
    // for(int i=0;i<batch_dim*heads_;i++){
    //     for(int j=0;j<ctx_blks_a_;j++){
    //         for(int k=0;k<ctx_blks_a_;k++){
    //             if(j >= k){
    //                 count++;
    //                 if(j == k){
    //                     for(int l1 = 0;l1<blk_size_;l1++){
    //                         for(int l2=0;l2<blk_size_;l2++){
    //                             if(l1<l2){
    //                                 d_ptr_cpu[i*(blocks_*blk_size_*blk_size_)+count*(blk_size_*blk_size_)+l1*blk_size_+l2] = 0;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // float *d_ptr;
    // cudaMalloc((void**)&d_ptr, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // cudaMemcpy(d_ptr, d_ptr_cpu, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_), cudaMemcpyHostToDevice);


    // float *e_ptr, *f_ptr;
    // cudaMalloc((void**)&e_ptr, sizeof(float)*m*n);
    // cudaMalloc((void**)&f_ptr, sizeof(float)*m*n);

    // float *e_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    // for(int i=0;i<m*n;i++){

    //     e_ptr_cpu[i] = 1.0;

    // }
    // cudaMemcpy(e_ptr, e_ptr_cpu, sizeof(float)*(m*n), cudaMemcpyHostToDevice);

    // float *f_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    // bst_sgemm_xn(custream, l2_ptr, y_ptr, e_ptr, f_ptr, blk_size_, blocks_, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, nn_lut_dim, nn_op, magic_, shift_, nn_max);

    // cudaMemcpy(f_ptr_cpu, f_ptr, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // for(int i=0;i<m*n;i++){
    //     printf("%f ", f_ptr_cpu[i]);
    // }

    // return 0;




    // float *dy_cpu  = (float*)malloc(sizeof(float)*(m*n));
    // float *dy_ptr;
    // cudaMalloc((void**)&dy_ptr, sizeof(float)*(m*n));

    // for(int i=0;i<(m);i++){
    //     for(int j=0;j<n;j++){
    //         dy_cpu[i*n+j] = 1;
    //     }
        
    // }

    // cudaMemcpy(dy_ptr, dy_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    // float *dv_cpu = (float*)malloc(sizeof(float)*m*n);
    // float *dv;
    // cudaMalloc((void**)&dv, sizeof(float)*m*n);

    // bst_sgemm_xn(custream, l1_ptr, y_ptr, dy_ptr, dv, blk_size_, blocks_, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, tn_lut_dim, tn_op, magic_, shift_, tn_max);

    // cudaMemcpy(dv_cpu, dv, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // for(int i=0;i<(32);i++){
    //     for(int j=0;j<n;j++){
    //         printf("%f ", dv_cpu[i*n+j]);
    //     }
    //     printf("%d \n", i);
    // }
    // return 0;

    // bhalf *dw_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // bhalf *dw;
    // cudaMalloc((void**)&dw, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // bst_sgemm_nt(custream, l_ptr, dy_ptr, e_ptr, dw, blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, nt_lut_dim);


    // cudaMemcpy(dw_cpu, dw, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_),cudaMemcpyDeviceToHost);

    // for(int i=0;i<blk_size_*blk_size_;i++){
    //     printf("%f ", dw_cpu[i]);
    // }

    // return 0;


    // bhalf *dx_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // bhalf *dx_ptr;
    // cudaMalloc((void**)&dx_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // bhalf *dw_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // for(int i=0;i<(batch_dim*heads_*blocks_);i++){
    //     for(int j=0;j<blk_size_*blk_size_;j++){
    //         dw_cpu[i*(blk_size_*blk_size_)+j] = j;
    //     }
        
    // }
    // bhalf *dw;
    // cudaMalloc((void**)&dw, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));
    // cudaMemcpy(dw, dw_cpu, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_), cudaMemcpyHostToDevice);

    // BlocksparseMaskedSoftmaxGrad<bhalf,bhalf2>(custream, l2_ptr, dw, y_ptr, dx_ptr, blk_size_, blocks_, batch_dim, heads_, ctx_blks_b_, lut_heads, nn_lut_dim, nn_max, scale);

    // cudaMemcpy(dx_cpu, dx_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_),cudaMemcpyDeviceToHost);

    // for(int i=0;i<blk_size_;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%.15f ", dx_cpu[i*blk_size_+j]);
    //     }
    //     printf("%d \n", i);
        
    // }

    // return 0;

    // float *dk_cpu = (float*)malloc(sizeof(float)*m*n);
    // float *dk;
    // cudaMalloc((void**)&dk, sizeof(float)*m*n);

    // float *dq_cpu = (float*)malloc(sizeof(float)*m*n);
    // float* dq;
    // cudaMalloc((void**)&dq, sizeof(float)*m*n);

    // bst_sgemm_xn(custream, l1_ptr, dx_ptr, a_ptr, dk, blk_size_, blocks_, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, tn_lut_dim, tn_op, magic_, shift_, tn_max);

    // bst_sgemm_xn(custream, l2_ptr, dx_ptr, b_ptr, dq, blk_size_, blocks_, batch_dim, ctx_blks_c_, ctx_blks_b_, heads_, head_state_, lut_heads, nn_lut_dim, nn_op, magic_, shift_, nn_max);

    // cudaMemcpy(dq_cpu, dq, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    // cudaMemcpy(dk_cpu, dk, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // for(int i=0;i<m*n;i++){

    //     printf("%.25f ", dk_cpu[i]);

    // }
    
    // for(int i=0;i<m*n;i++){
    //     if(dq_cpu[i] != 0){
    //         printf("%f ", dk_cpu[i]);
    //     }
    // }

    // return 0;
}