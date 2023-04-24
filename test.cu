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
    const uint blocks_= 55;
    const uint seq_len = 320;
    const uint state = 512;
    const uint m = batch_dim*seq_len;
    const uint n = state;
    const uint nn_op = 1; 
    const uint nt_op = 0;
    const uint tn_op = 2;
    int count = 0;
    const uint ctx_blks_a_ = 10;
    const uint ctx_blks_b_ = 10;
    const uint ctx_blks_c_ = 10;
    const float scale = 1.0/sqrt(head_state_);

    const uint lut_heads = 1;
    const uint nt_lut_dim = 55;
    const uint nn_lut_dim = 65;
    const uint tn_lut_dim = 65;
    const uint mask_heads= 1;
    

    const uint nn_max = 10;
    const uint tn_max = 10;

    // struct timeval GET_TIME_START, GET_TIME_END, GET_TIME_START1, GET_TIME_END1;
    


    // uint2 *nt_lut = (uint2*)malloc(sizeof(uint2)*blocks_);

    // for(int i=0;i<10;i++){
    //     for(int j=0;j<10;j++){
    //         if(i >= j){
    //             nt_lut[count].x = i;
    //             nt_lut[count].y = j;
    //             count++;
    //         }
            
    //     }
    // }

    // uint2 *l_ptr;
    // cudaMalloc((void**)&l_ptr, sizeof(uint2)*blocks_);
    // cudaMemcpy(l_ptr, nt_lut, sizeof(uint2)*blocks_, cudaMemcpyHostToDevice);


    // bool *mask_cpu = (bool*)malloc(sizeof(bool)*blocks_*blk_size_*blk_size_);
    
    // for(int l = 0;l<blocks_;l++){
    //     if((nt_lut[l].x) == (nt_lut[l].y)){
    //         for(int i=0;i<blk_size_;i++){
    //             for(int j=0;j<blk_size_;j++){
    //                 if(i>=j){
    //                     mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 1;
    //                 }else{
    //                     mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 0;
    //                 }
    //             }
    //         }
    //     }else{
    //         for(int i=0;i<(blk_size_*blk_size_);i++){
    //             mask_cpu[(l*blk_size_*blk_size_)+ i] = 1;
    //         }
    //     }
        
    // }

    // unsigned int *mask_np = (unsigned int*)malloc(sizeof(unsigned int)*blocks_*blk_size_);
    // unsigned int *mask = (unsigned int*)malloc(sizeof(unsigned int)*blocks_*blk_size_);

    // for(int i=0;i<(blocks_*blk_size_);i++){
    //     mask_np[i] = 0;
    //     for(int j=0;j<32;j++){
    //         bool keep = mask_cpu[i*blk_size_+j];
    //         unsigned int temp;
    //         temp = keep << (j);
    //         mask_np[i] = mask_np[i] | temp;
            
    //     }
    // }

    // for(int i=0; i<blocks_; i++){
    //     for(int j=0; j<blk_size_; j++){
    //         mask[j*blocks_+i] = mask_np[i*blk_size_+j];
    //     }
    // }

    // char *m_ptr;
    // cudaMalloc((void**)&m_ptr, sizeof(unsigned int)*blocks_*blk_size_);
    // cudaMemcpy(m_ptr, mask_np, sizeof(unsigned int)*blocks_*blk_size_, cudaMemcpyHostToDevice);

    // char *m1_ptr;
    // cudaMalloc((void**)&m1_ptr, sizeof(unsigned int)*blocks_*blk_size_);
    // cudaMemcpy(m1_ptr, mask, sizeof(unsigned int)*blocks_*blk_size_, cudaMemcpyHostToDevice);

    // // for(int i=0;i<blocks_;i++){
    // //     for(int j=0;j<blk_size_;j++){
    // //         printf("%u ", mask_np[i*blk_size_+j]);
            
    // //     }
    // //     printf("\n");
        
    // // }

    // // printf("\n");

    // // for(int i=0;i<blk_size_;i++){
    // //     for(int j=0;j<blocks_;j++){
    // //         printf("%u ", mask[i*blocks_+j]);
    // //     }
    // //     printf("\n");
    // // }

    // // return 0;
    
    // uint2 *tn_lut = (uint2*)malloc(sizeof(uint2)*(blocks_+ctx_blks_b_));
    // for(int i=0;i<(blocks_+ctx_blks_b_);i++){
    //     tn_lut[i].x = 0;
    //     tn_lut[i].y = 0;
    // }

    // int offset = 10;
    // int sum = 10;
    // for(int i=0;i<ctx_blks_b_;i++){
        
    //     tn_lut[i].x = sum;  
    //     tn_lut[i].y = offset;
    //     sum+=offset;
    //     offset--;

    // }

    // count = 0;
    // for(int col = 0;col<ctx_blks_b_;col++){
    //     for(int row = col;row<ctx_blks_b_;row++){

    //         int bid = 0;
    //         for(int k = row; k>0; k--){
    //             bid+=k;
    //         }
    //         tn_lut[ctx_blks_b_+count].x = bid+col;
    //         tn_lut[ctx_blks_b_+count].y = row;
    //         count++;
    //         // printf("count : %d ", count);

    //     }
    // }


    // uint2 *l1_ptr;
    // cudaMalloc((void**)&l1_ptr, sizeof(uint2)*(blocks_+ctx_blks_b_));
    // cudaMemcpy(l1_ptr, tn_lut, sizeof(uint2)*(blocks_)+ctx_blks_b_, cudaMemcpyHostToDevice);


    // uint2 *nn_lut = (uint2*)malloc(sizeof(uint2)*(blocks_+ctx_blks_b_));
    // for(int i=0;i<(blocks_+ctx_blks_b_);i++){
    //     tn_lut[i].x = 0;
    //     tn_lut[i].y = 0;
    // }
    // offset = 1;
    // sum = 10;
    // for(int i=0;i<ctx_blks_b_;i++){
        
    //     nn_lut[i].x = sum;
    //     nn_lut[i].y = offset;
    //     sum+=offset;
    //     offset++;
        
    // }

    // count = 0;
    // for(int row = 0; row<ctx_blks_b_; row++){
    //     for(int col = 0; col<=row; col++){

    //         nn_lut[ctx_blks_b_+count].x = count;
    //         nn_lut[ctx_blks_b_+count].y = col;
    //         count++;
    //     }
    // }

    // // for(int i=0;i<(blocks_+ctx_blks_b_);i++){
    // //     printf("%d %d\n", nn_lut[i].x, nn_lut[i].y);
    // // }

    // uint2 *l2_ptr;
    // cudaMalloc((void**)&l2_ptr, sizeof(uint2)*(blocks_+ctx_blks_b_));
    // cudaMemcpy(l2_ptr, nn_lut, sizeof(uint2)*(blocks_+ctx_blks_b_), cudaMemcpyHostToDevice);

    // //for xn
    // uint div = CEIL_DIV(head_state_, 64);
    // uint magic_, shift_;
    // magicu64(div, magic_, shift_);

    // float *a_ptr_cpu = (float*)malloc(sizeof(float)*m*n);
    // float *b_ptr_cpu = (float*)malloc(sizeof(float)*m*n);

    // for(int i=0;i<m*n;i++){

    //     a_ptr_cpu[i] = 1;
    //     b_ptr_cpu[i] = 1;

    // }

    // float *a_ptr, *b_ptr;
    // cudaMalloc((void**)&a_ptr, sizeof(float)*m*n);
    // cudaMalloc((void**)&b_ptr, sizeof(float)*m*n);

    // cudaMemcpy(a_ptr, a_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    // cudaMemcpy(b_ptr, b_ptr_cpu, sizeof(float)*m*n, cudaMemcpyHostToDevice);



    // uint2 *rblk_lutc = (uint2*)malloc(sizeof(uint2)*(blocks_));
    // count = 0;
    // for(int i=0;i<ctx_blks_a_;i++){
    //     for(int j=0;j<nn_lut[i].y;j++){
    //         rblk_lutc[count].x = count;
    //         rblk_lutc[count].y = i;
    //         count++;
    //     }
    // }

    // uint2 *rblk_lut;
    // cudaMalloc((void**)&rblk_lut, sizeof(uint2)*blocks_);
    // cudaMemcpy(rblk_lut, rblk_lutc, sizeof(uint2)*(blocks_), cudaMemcpyHostToDevice);

    // float *y_ptr;
    // cudaMalloc((void**)&y_ptr, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // int *rblk_flag;
    // cudaMalloc((void**)&rblk_flag, sizeof(int)*ctx_blks_a_*batch_dim*heads_);

    // int *rblk_flag1;
    // cudaMalloc((void**)&rblk_flag1, sizeof(int)*ctx_blks_a_*batch_dim*heads_);

    // int *rblk_flagc = (int*)malloc(sizeof(int)*ctx_blks_a_*batch_dim*heads_); 

    // float *Max;
    // cudaMalloc((void**)&Max, sizeof(float)*batch_dim*heads_*blocks_*blk_size_);

    // float *Sum;
    // cudaMalloc((void**)&Sum, sizeof(float)*batch_dim*heads_*blocks_*blk_size_);

    // float *Maxc = (float*)malloc(sizeof(float)*batch_dim*heads_*blocks_*blk_size_);
    // float *Sumc = (float*)malloc(sizeof(float)*batch_dim*heads_*blocks_*blk_size_);
    // float *yc = (float*)malloc(sizeof(float)*blk_size_*blk_size_*batch_dim*heads_*blocks_);

    // double time_avg1 = 0;
    // double time_min1 = 10000;

    // double time_avg2 = 0;
    // double time_min2 = 10000;

    // double time_avg = 0;
    // double time_min = 10000;


    // bhalf *c_ptr;
    // cudaMalloc((void**)&c_ptr, sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // bhalf *c_ptr_cpu = (bhalf*)malloc(sizeof(bhalf)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // float *y1_ptr;
    // cudaMalloc((void**)&y1_ptr, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_*blk_size_));

    // CUstream custream;
    // cudaStreamCreate(&custream); 
    

    // for(int i=0;i<1000;i++){

    //     cudaDeviceSynchronize();
    //     gettimeofday(&(GET_TIME_START1), NULL);
    //     // gettimeofday(&(GET_TIME_START), NULL);
        
    //     // fusion_attention<float, float2>(custream, l_ptr, a_ptr, b_ptr, l2_ptr, rblk_flag, rblk_flag1, rblk_lut, m_ptr, Max, Sum, y_ptr,
    //     //                 blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
    //     //                 lut_heads, nt_lut_dim, scale);

    //     fusion_attention1<float, float2>(custream, l_ptr, a_ptr, b_ptr, c_ptr, l2_ptr, rblk_flag, rblk_flag1, rblk_lut, m_ptr, Max, Sum, y_ptr,
    //                     blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, 
    //                     lut_heads, nt_lut_dim, scale);

    //     // bst_sgemm_nt(custream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, nt_lut_dim);

    //     // cudaDeviceSynchronize();
    //     // gettimeofday(&(GET_TIME_END), NULL);
    //     // double time1 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;

    //     // cudaDeviceSynchronize();
    //     // gettimeofday(&(GET_TIME_START), NULL);       
    //     // BlocksparseMaskedSoftmax<float,float2>(custream, l2_ptr, m1_ptr, c_ptr, y1_ptr, blk_size_, blocks_, batch_dim, heads_, ctx_blks_a_, lut_heads, nn_lut_dim, nn_max, mask_heads, scale);

    //     cudaDeviceSynchronize();
    //     // gettimeofday(&(GET_TIME_END), NULL);
    //     gettimeofday(&(GET_TIME_END1), NULL);


    //     // double time2 = (GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0;
    //     double time = (GET_TIME_END1.tv_sec - GET_TIME_START1.tv_sec) * 1000.0 + (GET_TIME_END1.tv_usec - GET_TIME_START1.tv_usec) / 1000.0;

    //     time_avg+=time;
    //     time_min = time_min > time ? time : time_min;

    //     // time_avg1+=time1;
    //     // time_min1 = time_min1 > time1 ? time1 : time_min1;

    //     // time_avg2+=time2;
    //     // time_min2 = time_min2 > time2 ? time2 : time_min2;

    //     cudaMemset(rblk_flag, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(rblk_flag1, 0, sizeof(int)*(ctx_blks_a_*batch_dim*heads_));
    //     cudaMemset(Max, 0, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_));
    //     cudaMemset(Sum, 0, sizeof(float)*(batch_dim*heads_*blocks_*blk_size_));
    // }

    // printf("time_avg:%f \n", time_avg1/1000.0);
    // printf("time_min:%f \n", time_min1);

    // printf("time_avg:%f \n", time_avg2/1000.0);
    // printf("time_min:%f \n", time_min2);

    // printf("time_avg:%f \n", time_avg/1000.0);
    // printf("time_min:%f \n", time_min);



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


    // for(int i=0;i<blocks_*blk_size_;i++){
    //     for(int j=0;j<blk_size_;j++){
    //         printf("%f ", yc[i*blk_size_+j]);
    //     }
    //     printf("%d \n", i);
    // }

    // for(int i=0;i<(ctx_blks_a_*heads_);i++){
    //     printf("%d \n", rblk_flagc[i]);
    // }


    // for(int i=0;i<(blk_size_*blk_size_);i++){
    //     // if(c_ptr_cpu[i] != 128)
    //     //     printf("%f ", c_ptr_cpu[i]);
    //     printf("%f ", c_ptr_cpu[i]);
    // }
    // printf("\n");

    // return 0;


   


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



    //fixed mask

    const uint fixed_lut_heads = 4;
    const uint fixed_mask_heads= 4;
    const uint fixed_nn_max = 5;
    const uint fixed_tn_max = 10;
    const uint local_attn_ctx = 128;
    uint stride = local_attn_ctx/blk_size_;

    //need change
    uint fixed_nt_lut_dim;
    uint fixed_nn_lut_dim;
    uint fixed_tn_lut_dim;
    uint fixed_blocks = fixed_nt_lut_dim;

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

    fixed_nn_lut_dim = fixed_nt_lut_dim+ctx_blks_a_;
    fixed_tn_lut_dim = fixed_nt_lut_dim+ctx_blks_a_;

    uint2 *fixed_nt_lut = (uint2*)malloc(sizeof(uint2)*fixed_nt_lut_dim*heads_);

    for(int i=0;i<(heads_*ctx_blks_a_);i++){

        for(int j=0;j<ctx_blks_a_;j++){

            if(layout[i*ctx_blks_a_+ j] == 1){

                fixed_nt_lut[count].x = i%(ctx_blks_a_);
                fixed_nt_lut[count].y = j;
                count++;

            }
        }
    }

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

    uint2 *d_fixed_nn_lut;
    cudaMalloc((void**)&d_fixed_nn_lut, sizeof(uint2)*(fixed_nn_lut_dim*heads_));
    cudaMemcpy(d_fixed_nn_lut, fixed_nn_lut, sizeof(uint2)*(heads_*fixed_nn_lut_dim), cudaMemcpyHostToDevice);


    bool *fixed_mask = (bool*)malloc(sizeof(bool)*fixed_blocks*blk_size_*blk_size_);
    
    for(int l = 0;l<fixed_blocks;l++){

        if((fixed_nt_lut[l].x) == (fixed_nt_lut[l].y)){

            for(int i=0;i<blk_size_;i++){

                for(int j=0;j<blk_size_;j++){

                    if(i>=j){
                        fixed_mask[l*(blk_size_*blk_size_)+i*blk_size_+j] = 1;
                    }else{
                        fixed_mask[l*(blk_size_*blk_size_)+i*blk_size_+j] = 0;
                    }
                }
            }
        }else{
            for(int i=0;i<(blk_size_*blk_size_);i++){
                fixed_mask[(l*blk_size_*blk_size_)+ i] = 1;
            }
        }
        
    }

    unsigned int *fixed_mask_np = (unsigned int*)malloc(sizeof(unsigned int)*fixed_blocks*blk_size_);

    for(int i=0;i<(fixed_blocks*blk_size_);i++){
        fixed_mask_np[i] = 0;
        for(int j=0;j<32;j++){
            bool keep = fixed_mask[i*blk_size_+j];
            unsigned int temp;
            temp = keep << (j);
            fixed_mask_np[i] = fixed_mask_np[i] | temp;  
        }
    }

    





    





    return 0;

}
