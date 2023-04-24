// #define TRANSFORMER_DEBUG
#define DRAW_MAT

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
#include "cuda_kernels.h"
#include "validate.h"
#include "common.h"
#include "embedding.h"
#include "layernorm.h"
#include "matmul.h"
#include "ewops.h"
#include "optimizer.h"

struct timeval GET_TIME_START, GET_TIME_END;

using namespace std;

void time_check_begin()
{
    gettimeofday(&(GET_TIME_START), NULL);
}

double time_check_end()
{   
    gettimeofday(&(GET_TIME_END), NULL);
    return ((GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0);
}



int main(int argc, char* argv[]){

    if(argc != 6)
    {
        printf("./transformer_fp32 batch_size num_layers seq_len head_num size_per_head\n");
        printf("e.g., ./transformer_fp32 1 12 128 12 64\n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int num_layers = atoi(argv[2]);
    const int seq_len = atoi(argv[3]);
    const int head_num = atoi(argv[4]);
    const int size_per_head = atoi(argv[5]);
    const int epoch = 0; 
    const int vocab_size = 256;
    const int state = head_num * size_per_head;
    const int m = batch_size * seq_len;
    const int k = state;
    const int n = state;
    float scaler = 1.0 / (sqrt(size_per_head));
    float grad_sum_sum = 0.0;
    float clip_norm = 1.0;
    float norm_scale = 0.0;
    const float probe = 1.0-0.05;
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float learning_rate = 0.0005;
    const float epsilon = 0.00000001;
    float *loss = (float*)malloc(sizeof(float));

    float global_norm = 0;
    float grad_scale = 1.0;
    float clip_sigma = 0;

    float lr  = 0.0;
    int global_step = 1;
    float beta1_power = 1.0;
    float beta2_power = 1.0;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int offset = 0;
    int offset1 = 0;
    double used_time = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    srand((unsigned int)time(NULL));

    float *q_w = (float*)malloc(sizeof(float) * n * n * num_layers);
    float *k_w = (float*)malloc(sizeof(float) * n * n * num_layers);
    float *v_w = (float*)malloc(sizeof(float) * n * n * num_layers);
    float *a_w = (float*)malloc(sizeof(float) * n * n * num_layers);
    float *m1_w = (float*)malloc(sizeof(float) * n * n * 4 * num_layers);
    float *m2_w = (float*)malloc(sizeof(float) * n * n * 4 * num_layers);

    for(int i=0;i<num_layers;i++){

        random_normalInit(q_w+(i*n*n), 0, 0.02, n*n);
        random_normalInit(k_w+(i*n*n), 0, 0.02, n*n);
        random_normalInit(v_w+(i*n*n), 0, 0.02, n*n);
        random_normalInit(a_w+(i*n*n), 0, 0.02/num_layers, n*n);
        random_normalInit(m1_w+(i*n*n*4), 0, 0.02, n*n*4);
        random_normalInit(m2_w+(i*n*n*4), 0, 0.02, n*n*4);
    }

    float *d_qw, *d_kw, *d_vw, *d_aw, *d_m1w, *d_m2w;
    cudaMalloc((void **)&d_qw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_kw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_vw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_aw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_m1w, sizeof(float)*n*n*4*num_layers);
    cudaMalloc((void **)&d_m2w, sizeof(float)*n*n*4*num_layers);
    
    float *q_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(q_b, 0.0, n*num_layers);
    float *k_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(k_b, 0.0, n*num_layers);
    float *v_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(v_b, 0.0, n*num_layers);
    float *a_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(a_b, 0.0, n*num_layers);
    float *m1_b = (float*)malloc(sizeof(float) * n * 4 * num_layers);
    constantInit(m1_b, 0.0, n*4*num_layers);
    float *m2_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(m2_b, 0.0, n*num_layers);
    float *norm_a_g = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(norm_a_g, 1.0, n*num_layers);
    float *norm_a_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(norm_a_b, 0.0, n*num_layers);
    float *norm_m_g = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(norm_m_g, 1.0, n*num_layers);
    float *norm_m_b = (float*)malloc(sizeof(float) * n * num_layers);
    constantInit(norm_m_b, 0.0, n*num_layers);
    float *x_embed = (float*)malloc(sizeof(float) * vocab_size * state);
    random_normalInit(x_embed, 0, 0.02, vocab_size * state);
    float *p_embed = (float*)malloc(sizeof(float) * seq_len * state);
    random_normalInit(p_embed, 0, 0.01, seq_len * state);

    float *d_qb, *d_kb, *d_vb, *d_ab, *d_m1b, *d_m2b, *d_norm_ag, *d_norm_ab, *d_norm_mg, *d_norm_mb, *d_xembed, *d_pembed;
    cudaMalloc((void **)&d_qb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_kb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_vb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_ab, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_m1b, sizeof(float)*n*4*num_layers);
    cudaMalloc((void **)&d_m2b, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_ag, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_ab, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_mg, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_mb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_xembed, sizeof(float)*vocab_size*state);
    cudaMalloc((void **)&d_pembed, sizeof(float)*seq_len*state);

    cudaMemcpy(d_qw, q_w, sizeof(float)*n*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kw, k_w, sizeof(float)*n*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vw, v_w, sizeof(float)*n*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aw, a_w, sizeof(float)*n*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m1w, m1_w, sizeof(float)*n*n*4*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2w, m2_w, sizeof(float)*n*n*4*num_layers, cudaMemcpyHostToDevice);

    cudaMemcpy(d_qb, q_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kb, k_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, v_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ab, a_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m1b, m1_b, sizeof(float)*n*4*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2b, m2_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_ag, norm_a_g, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_ab, norm_a_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_mg, norm_m_g, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_mb, norm_m_b, sizeof(float)*n*num_layers, cudaMemcpyHostToDevice);

    cudaMemcpy(d_xembed, x_embed, sizeof(float)*vocab_size*state, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pembed, p_embed, sizeof(float)*seq_len*state, cudaMemcpyHostToDevice);



    float *embed_lookup, *x_after_dropout, *p_after_dropout;
    cudaMalloc((void **)&embed_lookup, sizeof(float)*m*n);
    cudaMalloc((void **)&x_after_dropout, sizeof(float)*m*n);
    cudaMalloc((void **)&p_after_dropout, sizeof(float)*seq_len*n);

    float *embed_add;
    cudaMalloc((void **)&embed_add, sizeof(float)*m*n*(num_layers+1));

    float *norm_a, *norm_a_mean, *norm_a_rstd;
    cudaMalloc((void **)&norm_a, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&norm_a_mean, sizeof(float)*m*num_layers);
    cudaMalloc((void **)&norm_a_rstd, sizeof(float)*m*num_layers);

    float *q_out, *k_out, *v_out;
    cudaMalloc((void **)&q_out, sizeof(float)*m*n);
    cudaMalloc((void **)&k_out, sizeof(float)*m*n);
    cudaMalloc((void **)&v_out, sizeof(float)*m*n);

    float *q_bias_out, *k_bias_out, *v_bias_out;
    cudaMalloc((void **)&q_bias_out, sizeof(float)*m*n);
    cudaMalloc((void **)&k_bias_out, sizeof(float)*m*n);
    cudaMalloc((void **)&v_bias_out, sizeof(float)*m*n);

    float *q_reshape, *k_reshape, *v_reshape;
    cudaMalloc((void **)&q_reshape, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&k_reshape, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&v_reshape, sizeof(float)*m*n*num_layers);

    float *qk_out, *softmax_out, *sv_out, *sv_out_reshape;
    cudaMalloc((void **)&qk_out, sizeof(float)*batch_size*head_num*seq_len*seq_len);
    cudaMalloc((void **)&softmax_out, sizeof(float)*batch_size*head_num*seq_len*seq_len*num_layers);
    cudaMalloc((void **)&sv_out, sizeof(float)*m*n);
    cudaMalloc((void **)&sv_out_reshape, sizeof(float)*m*n*num_layers);

    float *a_out, *a_bias_out;
    cudaMalloc((void **)&a_out, sizeof(float)*m*n);
    cudaMalloc((void **)&a_bias_out, sizeof(float)*m*n);

    float *a_after_dropout, *add_1;
    cudaMalloc((void **)&a_after_dropout, sizeof(float)*m*n);
    cudaMalloc((void **)&add_1, sizeof(float)*m*n*num_layers);

    float *norm_m, *norm_m_mean, *norm_m_rstd;
    cudaMalloc((void **)&norm_m, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&norm_m_mean, sizeof(float)*m*num_layers);
    cudaMalloc((void **)&norm_m_rstd, sizeof(float)*m*num_layers);

    float *m1_out, *m1_gelu_out, *m1_bias_out;
    cudaMalloc((void **)&m1_out, sizeof(float)*m*n*4);
    cudaMalloc((void **)&m1_gelu_out, sizeof(float)*m*n*4*num_layers);
    cudaMalloc((void **)&m1_bias_out, sizeof(float)*m*n*4*num_layers);

    float *m2_out, *m2_bias_out;
    cudaMalloc((void **)&m2_out, sizeof(float)*m*n);
    cudaMalloc((void **)&m2_bias_out, sizeof(float)*m*n);

    float *m_after_dropout;
    cudaMalloc((void **)&m_after_dropout, sizeof(float)*m*n);

    float *logits, *softmax_logits;
    cudaMalloc((void **)&logits, sizeof(float)*m*vocab_size);
    cudaMalloc((void **)&softmax_logits, sizeof(float)*m*vocab_size);

    float *d_qdw, *d_kdw, *d_vdw, *d_adw, *d_m1dw, *d_m2dw;
    cudaMalloc((void **)&d_qdw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_kdw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_vdw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_adw, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_m1dw, sizeof(float)*n*n*4*num_layers);
    cudaMalloc((void **)&d_m2dw, sizeof(float)*n*n*4*num_layers);

    float *d_qdb, *d_kdb, *d_vdb, *d_adb, *d_m1db, *d_m2db, *d_norm_adg, *d_norm_adb, *d_norm_mdg, *d_norm_mdb;
    cudaMalloc((void **)&d_qdb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_kdb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_vdb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_adb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_m1db, sizeof(float)*n*num_layers*4);
    cudaMalloc((void **)&d_m2db, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_adg, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_adb, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_mdg, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_mdb, sizeof(float)*n*num_layers);

    float *logits_dx, *logits_dw;
    cudaMalloc((void **)&logits_dx, sizeof(float)*(num_layers+1)*m*n);
    cudaMalloc((void **)&logits_dw, sizeof(float)*n*vocab_size);

    float *entropy_grad = (float*)malloc(sizeof(float)*m*vocab_size);
    cudaMalloc((void **)&entropy_grad, sizeof(float)*m*vocab_size);

    float *dx1, *dx2, *dx3, *grad_add1, *dx_41, *dx_42;
    cudaMalloc((void **)&dx1, sizeof(float)*n*m);
    cudaMalloc((void **)&dx2, sizeof(float)*n*m);
    cudaMalloc((void **)&dx3, sizeof(float)*n*m);
    cudaMalloc((void **)&grad_add1, sizeof(float)*n*m);
    cudaMalloc((void **)&dx_41, sizeof(float)*n*m*4);
    cudaMalloc((void **)&dx_42, sizeof(float)*n*m*4);

    float *q_grad, *k_grad, *v_grad, *q_grad_trans;
    cudaMalloc((void **)&q_grad, sizeof(float)*n*m);
    cudaMalloc((void **)&q_grad_trans, sizeof(float)*n*m);
    cudaMalloc((void **)&k_grad, sizeof(float)*n*m);
    cudaMalloc((void **)&v_grad, sizeof(float)*n*m);

    float *q_grad_reshape, *k_grad_reshape, *v_grad_reshape;
    cudaMalloc((void **)&q_grad_reshape, sizeof(float)*n*m);
    cudaMalloc((void **)&k_grad_reshape, sizeof(float)*n*m);
    cudaMalloc((void **)&v_grad_reshape, sizeof(float)*n*m);

    float *sv_grad, *softmaxgrad;
    cudaMalloc((void **)&sv_grad, sizeof(float)*batch_size*head_num*seq_len*seq_len);
    cudaMalloc((void **)&softmaxgrad, sizeof(float)*batch_size*head_num*seq_len*seq_len);

    float *p_embed_grad, *p_embed_dw, *embed_grad, *logits_dw_trans, *embed_add_out;
    cudaMalloc((void**)&p_embed_grad, sizeof(float)*seq_len*state);
    cudaMalloc((void**)&p_embed_dw, sizeof(float)*seq_len*state);
    cudaMalloc((void**)&embed_grad, sizeof(float)*vocab_size*state);
    cudaMalloc((void**)&logits_dw_trans, sizeof(float)*vocab_size*state);
    cudaMalloc((void**)&embed_add_out, sizeof(float)*vocab_size*state);

    
    
    char xs_path[100];
    char temp[100];
    int *xs_int = (int*)malloc(sizeof(int) * m);
    unsigned char *xs_char = (unsigned char*)malloc(sizeof(unsigned char) * m);
    
    char ys_path[100];
    char y_temp[100];
    int *ys_int = (int*)malloc(sizeof(int) * m);
    unsigned char *ys_char = (unsigned char*)malloc(sizeof(unsigned char) * m);
    int *one_hot_ys = (int*)malloc(sizeof(int)*m*vocab_size);

    float *gradients_sum;
    cudaMalloc((void**)&gradients_sum, sizeof(float)*(16*num_layers+2));
    cudaMemset(gradients_sum, 0.0, sizeof(float)*(16*num_layers+2));

    float *gradients_sumc = (float*)malloc(sizeof(float)*num_layers*16+2);
    for(int i=0;i<num_layers*16+2;i++){
        gradients_sumc[i] = 0.0;
    }

    

    float *d_norm_a_g_mt, *d_norm_a_g_vt, *d_norm_a_b_mt, *d_norm_a_b_vt;
    cudaMalloc((void **)&d_norm_a_g_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_a_g_vt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_a_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_a_b_vt, sizeof(float)*n*num_layers);

    cudaMemset(d_norm_a_g_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_a_g_vt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_a_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_a_b_vt, 0.0, sizeof(float)*n*num_layers);

    float *d_norm_m_g_mt, *d_norm_m_g_vt, *d_norm_m_b_mt, *d_norm_m_b_vt;
    cudaMalloc((void **)&d_norm_m_g_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_m_g_vt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_m_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_norm_m_b_vt, sizeof(float)*n*num_layers);

    cudaMemset(d_norm_m_g_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_m_g_vt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_m_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_norm_m_b_vt, 0.0, sizeof(float)*n*num_layers);

    float *d_q_b_mt, *d_q_b_vt, *d_k_b_mt, *d_k_b_vt, *d_v_b_mt, *d_v_b_vt;
    cudaMalloc((void **)&d_q_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_q_b_vt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_k_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_k_b_vt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_v_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_v_b_vt, sizeof(float)*n*num_layers);

    cudaMemset(d_q_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_q_b_vt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_k_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_k_b_vt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_v_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_v_b_vt, 0.0, sizeof(float)*n*num_layers);

    float *d_q_w_mt, *d_q_w_vt, *d_k_w_mt, *d_k_w_vt, *d_v_w_mt, *d_v_w_vt;
    cudaMalloc((void **)&d_q_w_mt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_q_w_vt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_k_w_mt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_k_w_vt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_v_w_mt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_v_w_vt, sizeof(float)*n*n*num_layers);

    cudaMemset(d_q_w_mt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_q_w_vt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_k_w_mt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_k_w_vt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_v_w_mt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_v_w_vt, 0.0, sizeof(float)*n*n*num_layers);

    float *d_a_b_mt, *d_a_b_vt, *d_a_w_mt, *d_a_w_vt;
    cudaMalloc((void **)&d_a_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_a_b_vt, sizeof(float)*n*num_layers);

    cudaMemset(d_a_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_a_b_vt, 0.0, sizeof(float)*n*num_layers);

    cudaMalloc((void **)&d_a_w_mt, sizeof(float)*n*n*num_layers);
    cudaMalloc((void **)&d_a_w_vt, sizeof(float)*n*n*num_layers);

    cudaMemset(d_a_w_mt, 0.0, sizeof(float)*n*n*num_layers);
    cudaMemset(d_a_w_vt, 0.0, sizeof(float)*n*n*num_layers);

    float *d_m1_b_mt, *d_m1_b_vt;
    cudaMalloc((void **)&d_m1_b_mt, sizeof(float)*n*4*num_layers);
    cudaMalloc((void **)&d_m1_b_vt, sizeof(float)*n*4*num_layers);

    cudaMemset(d_m1_b_mt, 0.0, sizeof(float)*n*4*num_layers);
    cudaMemset(d_m1_b_vt, 0.0, sizeof(float)*n*4*num_layers);

    float *d_m1_w_mt, *d_m1_w_vt, *d_m2_w_mt, *d_m2_w_vt;
    cudaMalloc((void **)&d_m1_w_mt, sizeof(float)*n*n*4*num_layers);
    cudaMalloc((void **)&d_m1_w_vt, sizeof(float)*n*n*4*num_layers);
    cudaMalloc((void **)&d_m2_w_mt, sizeof(float)*n*n*4*num_layers);
    cudaMalloc((void **)&d_m2_w_vt, sizeof(float)*n*n*4*num_layers);

    cudaMemset(d_m1_w_mt, 0.0, sizeof(float)*n*n*4*num_layers);
    cudaMemset(d_m1_w_vt, 0.0, sizeof(float)*n*n*4*num_layers);
    cudaMemset(d_m2_w_mt, 0.0, sizeof(float)*n*n*4*num_layers);
    cudaMemset(d_m2_w_vt, 0.0, sizeof(float)*n*n*4*num_layers);

    float *d_m2_b_mt, *d_m2_b_vt;
    cudaMalloc((void **)&d_m2_b_mt, sizeof(float)*n*num_layers);
    cudaMalloc((void **)&d_m2_b_vt, sizeof(float)*n*num_layers);

    cudaMemset(d_m2_b_mt, 0.0, sizeof(float)*n*num_layers);
    cudaMemset(d_m2_b_vt, 0.0, sizeof(float)*n*num_layers);

    float *d_p_embed_mt, *d_p_embed_vt;
    cudaMalloc((void **)&d_p_embed_mt, sizeof(float)*seq_len*state);
    cudaMalloc((void **)&d_p_embed_vt, sizeof(float)*seq_len*state);

    cudaMemset(d_p_embed_mt, 0.0, sizeof(float)*seq_len*state);
    cudaMemset(d_p_embed_vt, 0.0, sizeof(float)*seq_len*state);

    float *d_x_embed_mt, *d_x_embed_vt;
    cudaMalloc((void **)&d_x_embed_mt, sizeof(float)*vocab_size*state);
    cudaMalloc((void **)&d_x_embed_vt, sizeof(float)*vocab_size*state);

    cudaMemset(d_x_embed_mt, 0.0, sizeof(float)*vocab_size*state);
    cudaMemset(d_x_embed_vt, 0.0, sizeof(float)*vocab_size*state);

    

    int *entropy_random = (int*)malloc(sizeof(int)*batch_size*seq_len*state*3);

    for(int i=0;i<batch_size*seq_len*state*3;i++){
        entropy_random[i] = rand();
    }

    unsigned int *d_xmask, *d_pmask;
    cudaMalloc((void **)&d_xmask, sizeof(unsigned int)*seq_len*state);
    cudaMalloc((void **)&d_pmask, sizeof(unsigned int)*5120);

    
    
    unsigned int *d_amask, *d_mmask;
    cudaMalloc((void **)&d_amask, sizeof(unsigned int)*seq_len*state*num_layers);
    cudaMalloc((void **)&d_mmask, sizeof(unsigned int)*seq_len*state*num_layers);

#ifdef DRAW_MAT

    float *qk_out_cpu = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
    float *softmax_out_cpu = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
    int *mask = (int*)malloc(sizeof(int)*seq_len*seq_len);
    int *d_mask;
    cudaMalloc((void**)&d_mask, sizeof(int)*seq_len*seq_len);

#endif

    for(int iter=0;iter<1;iter++)
    {

        char xs_path_1[] = "xs";
        strcpy(xs_path, xs_path_1);
        sprintf(temp, "%d", 0);
        strcat(xs_path, temp);
        readbinary_char(xs_path, xs_char, m);
        for(int i=0;i<m;i++){
            xs_int[i] = (int)xs_char[i];  
        }

        char ys_path_1[] = "ys";
        strcpy(ys_path, ys_path_1);
        sprintf(y_temp, "%d", 0);
        strcat(ys_path, y_temp);
        readbinary_char(ys_path, ys_char, m);

       

        for(int i=0;i<m;i++){
            ys_int[i] = (int)ys_char[i];
        }
        to_one_hot(one_hot_ys, ys_int, batch_size, seq_len, vocab_size);

        int *d_xlabel, *d_ylabel;
        cudaMalloc((void **)&d_xlabel, sizeof(int)*m);
        cudaMalloc((void **)&d_ylabel, sizeof(int)*m*vocab_size);
        cudaMemcpy(d_xlabel, xs_int, sizeof(int)*batch_size*seq_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ylabel, one_hot_ys, sizeof(int)*m*vocab_size, cudaMemcpyHostToDevice);

        float *d_loss;
        cudaMalloc((void**)&d_loss, sizeof(float)*m);

         
#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif

        embedding_lookup_gpu<<<m, n>>>(embed_lookup, d_xembed, d_xlabel, batch_size, seq_len, state, vocab_size);
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("1 :embedding time: %f ms\n", used_time);
#endif


        int *d_entropy;
        cudaMalloc((void **)&d_entropy, sizeof(int)*batch_size*seq_len*state*3);
        cudaMemcpy(d_entropy, entropy_random, sizeof(int)*batch_size*seq_len*state*3, cudaMemcpyHostToDevice);

#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif

        GendropoutMask<<<m, n>>>(d_entropy, d_xmask, probe, seq_len*state);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("2 :gen x mask time: %f ms\n", used_time);
#endif

#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif

        ApplydropoutMask<<<320, 512>>>(x_after_dropout, embed_lookup, d_xmask, probe, seq_len*state);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("3 :x apply dropout time: %f ms\n", used_time);
#endif

#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif

        GendropoutMask<<<seq_len, n>>>(d_entropy, d_pmask, probe, seq_len*state);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("4 :gen p mask: %f ms\n", used_time);
#endif

#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif

        ApplydropoutMask_SmallSize<<<10, 512>>>(p_after_dropout, d_pembed, d_pmask, probe, 5120);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("5 :p apply dropout time: %f ms\n", used_time);
#endif


#ifdef TRANSFORMER_DEBUG
         cudaDeviceSynchronize();
         time_check_begin();
#endif
        tensor_add_matrix_gpu<<<m, n>>>(embed_add, x_after_dropout, p_after_dropout, batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("6 :tensor+matrix time: %f ms\n", used_time);
#endif
        
        for(int i=0;i<num_layers;i++){
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            layernorm_gpu<<<m, n>>>(norm_a+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), embed_add+(i*m*n), d_norm_ag+(i*n), d_norm_ab+(i*n), batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm a time in %d layer: %f ms\n", 7+i*22, i+1, used_time);
#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        state, m, state,
                        &alpha, 
                        d_qw+(i*n*n), state,
                        norm_a+(i*m*n), state,
                        &beta, 
                        q_out, state);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        state, (batch_size*seq_len), state, 
                        &alpha, 
                        d_kw+(i*n*n), state, 
                        norm_a+(i*m*n), state, 
                        &beta, 
                        k_out, state);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        state, (batch_size*seq_len), state, 
                        &alpha, 
                        d_vw+(i*n*n), state, 
                        norm_a+(i*m*n), state, 
                        &beta, 
                        v_out, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q&k&v dense mul time in %d layer: %f ms\n", 8+i*22, i+1, used_time);
#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_vector_gpu<<<m, n>>>(q_bias_out, q_out, d_qb+(i*n), batch_size, seq_len, state);
            tensor_add_vector_gpu<<<m, n>>>(k_bias_out, k_out, d_kb+(i*n), batch_size, seq_len, state);
            tensor_add_vector_gpu<<<m, n>>>(v_bias_out, v_out, d_vb+(i*n), batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q&k&v bias time in %d layer: %f ms\n", 9+i*22, i+1, used_time);
#endif
            

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(q_bias_out, q_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(k_bias_out, k_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(v_bias_out, v_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q&k&v reshape time in %d layer: %f ms\n", 10+i*22, i+1, used_time);
#endif
            
            offset = 0;
            offset1 = 0;

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            for(int j=0;j<batch_size*head_num;j++){

                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                seq_len, seq_len, size_per_head, 
                &alpha, 
                k_reshape+(i*m*n)+offset, size_per_head, 
                q_reshape+(i*m*n)+offset, size_per_head, 
                &beta, 
                qk_out+offset1, seq_len);
                offset+= (seq_len*size_per_head);
                offset1 += (seq_len*seq_len);
            }

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q*kT time in %d layer: %f ms\n", 11+i*22, i+1, used_time);
#endif  


#ifdef DRAW_MAT

        // generate_qk_mask(mask, 10, seq_len);
        // generate_qk_mask_big_bird(mask, seq_len, 32, 64);
        generate_qk_mask_fixed(mask, seq_len, 4);
        cudaMemset(d_mask, 0.0, sizeof(int)*seq_len*seq_len);
        cudaMemcpy(d_mask, mask, sizeof(int)*seq_len*seq_len, cudaMemcpyHostToDevice);

#endif




#ifdef DRAW_MAT

        
        cudaMemcpy(qk_out_cpu, qk_out, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);

        FILE *fp = NULL;

        fp = fopen("fixed_mask_after_qk.txt", "w");

        if(fp == NULL){

                printf("open error!\n");
        }

        for(int l = 0;l<head_num*batch_size;l++){
            for(int j=0;j<seq_len;j++){
                for(int k=0;k<seq_len;k++){
                    qk_out_cpu[l*seq_len*seq_len+j*seq_len+k] *= mask[j*seq_len+k];
                    // if(j<k){
                    //     qk_out_cpu[l*seq_len*seq_len+j*seq_len+k] = 0.0;
                    // }
                    fprintf(fp, "%.25f ", qk_out_cpu[l*seq_len*seq_len+j*seq_len+k]);
                }
            }
        }

        fclose(fp);
        
        return 0;

#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            // softmax_gpu<<<(batch_size*head_num), seq_len>>>(softmax_out+(i*batch_size*head_num*seq_len*seq_len), qk_out, scaler, batch_size, head_num, seq_len);
            softmax_gpu_with_mask<<<(batch_size*head_num), seq_len>>>(softmax_out+(i*batch_size*head_num*seq_len*seq_len), qk_out, d_mask, scaler, batch_size, head_num, seq_len);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :softmax time in %d layer: %f ms\n", 12+i*22, i+1, used_time);
#endif

                
#ifdef DRAW_MAT  

        cudaMemcpy(softmax_out_cpu, softmax_out, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);

        // FILE *fp1 = NULL;

        fp = fopen("bb_mask_softmax.txt", "w");

        if(fp == NULL){

                printf("open error!\n");
        }

        for(int j=0;j<batch_size*head_num*seq_len*seq_len;j++){

            fprintf(fp, "%.15f ", softmax_out_cpu[j]);
            
        }

        fclose(fp);

#endif

        
            offset = 0;
            offset1 = 0;
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            for(int j=0;j<batch_size*head_num;j++){
                
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        size_per_head, seq_len, seq_len, 
                        &alpha, 
                        v_reshape+(i*m*n)+offset, size_per_head, 
                        softmax_out+offset1,seq_len, 
                        &beta, 
                        sv_out+offset, size_per_head);
 
                offset1+= (seq_len*seq_len);
                offset += (seq_len*size_per_head); 
            }
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :qk*v time in %d layer: %f ms\n", 13+i*22, i+1, used_time);
#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            transpose_0213to0123_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(sv_out, sv_out_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :v reshape time in %d layer: %f ms\n", 14+i*22, i+1, used_time);
#endif            
            


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        state, (batch_size*seq_len), state, 
                        &alpha, 
                        d_aw+(i*n*n), state, 
                        sv_out_reshape+(i*m*n),state, 
                        &beta, 
                        a_out, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a dense mul time in %d layer: %f ms\n", 15+i*22, i+1, used_time);
#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_vector_gpu<<<m, n>>>(a_bias_out, a_out, d_ab+(i*n), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a bias time in %d layer: %f ms\n", 16+i*22, i+1, used_time);
#endif

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            
        GendropoutMask<<<m, n>>>(d_entropy, d_amask+i*seq_len*n, probe, seq_len*state); 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a gen mask time in %d layer: %f ms\n", 17+i*22, i+1, used_time);
#endif

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            ApplydropoutMask<<<320, 512>>>(a_after_dropout, a_bias_out, d_amask+i*seq_len*n, probe, seq_len*state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a dropout time in %d layer: %f ms\n", 18+i*22, i+1, used_time);
#endif
            


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_tensor_gpu<<<m, n>>>(add_1+(i*m*n), embed_add+(i*m*n), a_after_dropout, batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :tensor+tensor time in %d layer: %f ms\n", 19+i*22, i+1, used_time);
#endif            
            


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            layernorm_gpu<<<m, n>>>(norm_m+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), add_1+(i*m*n), d_norm_mg+(i*n), d_norm_mb+(i*n), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm m time in %d layer: %f ms\n", 20+i*22, i+1, used_time);
#endif           
            

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        n*4, m, n, 
                        &alpha, 
                        d_m1w+(i*n*n*4), n*4, 
                        norm_m+(i*n*m), n, 
                        &beta, 
                        m1_out, n*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 dense mul time in %d layer: %f ms\n", 21+i*22, i+1, used_time);
#endif
            

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_vector_gpu_2048<<<m, n>>>(m1_bias_out+(i*m*n*4), m1_out, d_m1b+(i*n*4), batch_size, seq_len, n*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 bias time in %d layer: %f ms\n", 22+i*22, i+1, used_time); 
#endif

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            gelu_gpu<<<m, n>>>(m1_gelu_out+(i*m*n*4), m1_bias_out+(i*m*n*4), batch_size, seq_len, state*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 gelu time in %d layer: %f ms\n", 23+i*22, i+1, used_time);
#endif           
            
 
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        state, (batch_size*seq_len), state*4, 
                        &alpha, 
                        d_m2w+(i*n*n*4), state, 
                        m1_gelu_out+(i*m*n*4), state*4, 
                        &beta, 
                        m2_out, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 dense mul time in %d layer: %f ms\n", 24+i*22, i+1, used_time);
#endif

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_vector_gpu<<<m, n>>>(m2_bias_out, m2_out, d_m2b+(i*n), batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 bias time in %d layer: %f ms\n", 25+i*22, i+1, used_time);
#endif

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            
        GendropoutMask<<<m, n>>>(d_entropy, d_mmask+i*seq_len*n, probe, seq_len*state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 gen mask time in %d layer: %f ms\n", 26+i*22, i+1, used_time);
#endif
          
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            ApplydropoutMask<<<320, 512>>>(m_after_dropout, m2_bias_out, d_mmask+i*seq_len*n, probe, seq_len*state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 dropout time in %d layer: %f ms\n", 27+i*22, i+1, used_time);
#endif            

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_tensor_gpu<<<m, n>>>(embed_add+(i+1)*m*n, add_1+(i*m*n), m_after_dropout, batch_size, seq_len, state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :tensor+tensor time in %d layer: %f ms\n", 28+i*22, i+1, used_time);
#endif
            

        }

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    vocab_size, m, n,
                    &alpha,
                    d_xembed, n,
                    embed_add+(num_layers*m*n), n,
                    &beta,
                    logits, vocab_size);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("139 :logits dense mul time: %f ms\n", used_time); 
#endif        

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        softmax_cross_entropy_with_logits_gpu<<<m, 256>>>(softmax_logits, d_loss, logits, d_ylabel, batch_size, seq_len, vocab_size);
        
        float *loss_val;
        cudaMalloc((void **)&loss_val, sizeof(float));

        loss_add<<<1, n>>>(loss_val, d_loss, m);

        float *loss_c = (float*)malloc(sizeof(float));
        cudaMemcpy(loss_c, loss_val, sizeof(float), cudaMemcpyDeviceToHost);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("140 :calculate loss time: %f ms\n", used_time); 
#endif

        // printf("%f \n", loss_c[0]);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        cross_entropy_grad_gpu<<<m, vocab_size>>>(entropy_grad, softmax_logits, batch_size, seq_len, vocab_size);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("141 :cross entropy grad time: %f ms\n", used_time);
#endif


#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, vocab_size,
                    &alpha,
                    d_xembed, n,
                    entropy_grad, vocab_size,
                    &beta,
                    logits_dx+(num_layers*m*n), n);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    vocab_size, n, m,
                    &alpha,
                    entropy_grad, vocab_size,
                    embed_add+(num_layers*m*n), n,
                    &beta,
                    logits_dw, vocab_size);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("142 :logits grad(dx&dw) time: %f ms\n", used_time); 
#endif


        for(int i = num_layers-1; i>=0; i--){
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            ApplydropoutMask<<<320, 512>>>(dx1, logits_dx+((i+1)*m*n), d_mmask+(i*seq_len*n), probe, seq_len*state); 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m dropout grad time in %d layer: %f ms\n", 143+((num_layers-1)-i)*25, i+1, used_time);
#endif    


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            bias_grad_db_gpu<<<seq_len, n>>>(d_m2db+(i*n), dx1, batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 grad(bias) time in %d layer: %f ms\n", 144+((num_layers-1)-i)*25, i+1, used_time);
#endif
            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(17+i*16), d_m2db+(i*n), state);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        state*4, m, n, 
                        &alpha,
                        d_m2w+(i*n*n*4), n,
                        dx1, n,
                        &beta,
                        dx_41, state*4);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        state, state*4, m, 
                        &alpha,
                        dx1, n,
                        m1_gelu_out+(i*m*n*4), n*4,
                        &beta,
                        d_m2dw+(i*n*n*4), n); 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m2 grad(dx&dw) time in %d layer: %f ms\n", 145+((num_layers-1)-i)*25, i+1, used_time);
#endif
            gradients_add_gpu_512_2048<<<2048, 512>>>(gradients_sum+(16+i*16), d_m2dw+(i*n*n*4), n*n*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            gelu_grad_gpu<<<m, n>>>(dx_42, dx_41, m1_bias_out+(i*m*n*4), d_m1b+(i*n*4), batch_size, seq_len, state*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 grad(gelu) time in %d layer: %f ms\n", 146+((num_layers-1)-i)*25, i+1, used_time);
#endif


            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            bias_grad_db_2048_gpu<<<seq_len, n>>>(d_m1db+(i*n*4), dx_42, batch_size, seq_len, state*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 grad(bias) time in %d layer: %f ms\n", 147+((num_layers-1)-i)*25, i+1, used_time);
#endif

            gradients_add_gpu_2048<<<4, 512>>>(gradients_sum+(15+(i*16)), d_m1db+(i*n*4), n*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif            
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, n*4, 
                        &alpha,
                        d_m1w+(i*n*n*4), n*4,
                        dx_42, n*4,
                        &beta,
                        dx1, n);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n*4, n, m, 
                        &alpha,
                        dx_42, n*4,
                        norm_m+(i*m*n), n,
                        &beta,
                        d_m1dw+(i*n*n*4), n*4);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :m1 grad(dx&dw) time in %d layer: %f ms\n", 148+((num_layers-1)-i)*25, i+1, used_time);
#endif 
            
            gradients_add_gpu_512_2048<<<2048, 512>>>(gradients_sum+(14+i*16), d_m1dw+(i*n*n*4), n*n*4);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            layernorm_dg_db_gpu<<<seq_len, n>>>(d_norm_mdg+(i*n), d_norm_mdb+(i*n), dx1, add_1+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm grad(db&dg) time in %d layer: %f ms\n", 149+((num_layers-1)-i)*25, i+1, used_time);
#endif

            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(13+i*16), d_norm_mdg+(i*n), n);
            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(12+i*16), d_norm_mdb+(i*n), n);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            layernorm_grad_dx_gpu<<<m, n>>>(dx2, add_1+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), dx1, d_norm_ag+(i*n), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm grad(dx) time in %d layer: %f ms\n", 150+((num_layers-1)-i)*25, i+1, used_time);
#endif
            

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            tensor_add_tensor_gpu<<<m, n>>>(grad_add1, logits_dx+((i+1)*m*n), dx2, batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :tensor+tensor time in %d layer: %f ms\n", 151+((num_layers-1)-i)*25, i+1, used_time);
#endif
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            ApplydropoutMask<<<320, 512>>>(dx2, grad_add1, d_amask+(i*seq_len*n), probe, seq_len*state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a dropout grad time in %d layer: %f ms\n", 152+((num_layers-1)-i)*25, i+1, used_time);  
#endif
            
                  
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            bias_grad_db_gpu<<<320, n>>>(d_adb+(i*n), dx2, batch_size, seq_len, state);   
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a grad(bias) time in %d layer: %f ms\n", 153+((num_layers-1)-i)*25, i+1, used_time); 
#endif
            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(11+i*16), d_adb+(i*n), n);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, n, 
                        &alpha,
                        d_aw+(i*n*n), n,
                        dx2, n,
                        &beta,
                        dx1, n);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n, n, m, 
                        &alpha,
                        dx2, n,
                        sv_out_reshape+(i*m*n), n,
                        &beta,
                        d_adw+(i*n*n), n);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a grad(dx&dw) time in %d layer: %f ms\n", 154+((num_layers-1)-i)*25, i+1, used_time); 
#endif

            gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+(10+i*16), d_adw+(i*n*n), n*n);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(dx1, dx2, batch_size, seq_len, head_num, size_per_head);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :a reshape time in %d layer: %f ms\n", 155+((num_layers-1)-i)*25, i+1, used_time); 
#endif
            
            offset = 0;
            offset1 = 0;

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            for(int j=0; j<batch_size*head_num;j++){
                
                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            seq_len, seq_len, size_per_head,
                            &alpha,
                            v_reshape+(i*m*n)+offset, size_per_head, 
                            dx2+offset, size_per_head,
                            &beta,
                            sv_grad+offset1, seq_len);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            size_per_head, seq_len, seq_len,
                            &alpha,
                            dx2+offset, size_per_head, 
                            softmax_out+(i*batch_size*head_num*seq_len*seq_len)+offset1, seq_len,
                            &beta,
                            v_grad+offset, size_per_head);

                offset += seq_len*size_per_head;
                offset1 += seq_len*seq_len;

            }

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :sv grad(dx&dw) time in %d layer: %f ms\n", 156+((num_layers-1)-i)*25, i+1, used_time);
#endif
 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            softmax_grad_gpu<<<(batch_size*head_num), seq_len>>>(softmaxgrad, sv_grad, softmax_out+(i*batch_size*head_num*seq_len*seq_len), scaler, batch_size, head_num, seq_len);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :softmax grad time in %d layer: %f ms\n", 157+((num_layers-1)-i)*25, i+1, used_time);
#endif
            

            offset = 0;
            offset1 = 0;
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            for(int j=0;j<batch_size*head_num;j++){

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            size_per_head, seq_len, seq_len,
                            &alpha,
                            k_reshape+(i*m*n)+offset, size_per_head, 
                            softmaxgrad+offset1, seq_len,
                            &beta,
                            k_grad+offset, size_per_head);

                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            seq_len, size_per_head, seq_len,
                            &alpha,
                            softmaxgrad+offset1, seq_len, 
                            q_reshape+(i*m*n)+offset, size_per_head,
                            &beta,
                            q_grad+offset, seq_len);
                
                offset += seq_len*size_per_head;
                offset1 += seq_len*seq_len;
            }

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :qk grad(dx&dw) time in %d layer: %f ms\n", 158+((num_layers-1)-i)*25, i+1, used_time);
#endif


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            transpose_0123to0132_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(q_grad, q_grad_trans, batch_size, seq_len, head_num, size_per_head);
            transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(q_grad_trans, q_grad_reshape, batch_size, seq_len, head_num, size_per_head);
            transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(k_grad, k_grad_reshape, batch_size, seq_len, head_num, size_per_head);
            transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(v_grad, v_grad_reshape, batch_size, seq_len, head_num, size_per_head);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q&k&v grad reshape time in %d layer: %f ms\n", 159+((num_layers-1)-i)*25, i+1, used_time);
#endif
            


#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            bias_grad_db_gpu<<<320, n>>>(d_qdb+(i*n), q_grad_reshape, batch_size, seq_len, state);
            bias_grad_db_gpu<<<320, n>>>(d_kdb+(i*n), k_grad_reshape, batch_size, seq_len, state);
            bias_grad_db_gpu<<<320, n>>>(d_vdb+(i*n), v_grad_reshape, batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q&k&v grad(bias) time in %d layer: %f ms\n", 160+((num_layers-1)-i)*25, i+1, used_time);
#endif
            
            gradients_add_gpu_512<<<1,512>>>(gradients_sum+(9+i*16), d_qdb+(i*n), n);
            gradients_add_gpu_512<<<1,512>>>(gradients_sum+(8+i*16), d_kdb+(i*n), n);
            gradients_add_gpu_512<<<1,512>>>(gradients_sum+(7+i*16), d_vdb+(i*n), n);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, n, 
                        &alpha,
                        d_vw+(i*n*n), n,
                        v_grad_reshape, n,
                        &beta,
                        dx1, n);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n, n, m, 
                        &alpha,
                        v_grad_reshape, n,
                        norm_a+(i*m*n), n,
                        &beta,
                        d_vdw+(i*n*n), n);  

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :v grad(dx&dw) time in %d layer: %f ms\n", 161+((num_layers-1)-i)*25, i+1, used_time);
#endif
            
            gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+(6+i*16), d_vdw+(i*n*n), n*n);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, n, 
                        &alpha,
                        d_kw+(i*n*n), n,
                        k_grad_reshape, n,
                        &beta,
                        dx2, n);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n, n, m, 
                        &alpha,
                        k_grad_reshape, n,
                        norm_a+(i*m*n), n,
                        &beta,
                        d_kdw+(i*n*n), n); 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :k grad(dx&dw) time in %d layer: %f ms\n", 162+((num_layers-1)-i)*25, i+1, used_time);
#endif 
            gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+(5+i*16), d_kdw+(i*n*n), n*n);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, n, 
                        &alpha,
                        d_qw+(i*n*n), n,
                        q_grad_reshape, n,
                        &beta,
                        dx3, n);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        n, n, m, 
                        &alpha,
                        q_grad_reshape, n,
                        norm_a+(i*m*n), n,
                        &beta,
                        d_qdw+(i*n*n), n); 

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :q grad(dx&dw) time in %d layer: %f ms\n", 163+((num_layers-1)-i)*25, i+1, used_time);
#endif

            gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+(4+i*16), d_qdw+(i*n*n), n*n);

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
        //     tensor_add_tensor_gpu<<<m, n>>>(v_grad, dx1, dx2, batch_size, seq_len, state);
        //     tensor_add_tensor_gpu<<<m, n>>>(dx1, v_grad, dx3, batch_size, seq_len, state);
        tensor_add_tensor_add_tensor_gpu<<<m, n>>>(v_grad, dx1, dx2, dx3, batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :(tensor+tensor)x2 time in %d layer: %f ms\n", 164+((num_layers-1)-i)*25, i+1, used_time);
#endif
            

#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif
            layernorm_dg_db_gpu<<<seq_len, n>>>(d_norm_adg+(i*n), d_norm_adb+(i*n), v_grad, embed_add+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*n), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm grad(dg&db) time in %d layer: %f ms\n", 165+((num_layers-1)-i)*25, i+1, used_time);
#endif
            
            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(3+i*16), d_norm_adg+(i*n), n);
            gradients_add_gpu_512<<<1, 512>>>(gradients_sum+(2+i*16), d_norm_adb+(i*n), n);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif     
            layernorm_grad_dx_gpu<<<m, n>>>(dx2, embed_add+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), v_grad, d_norm_ag+(i*n), batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :layernorm grad(dx) time in %d layer: %f ms\n", 166+((num_layers-1)-i)*25, i+1, used_time);
#endif   
            

            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            time_check_begin();
#endif 
            tensor_add_tensor_gpu<<<m, n>>>(logits_dx+(i*m*n), grad_add1, dx2, batch_size, seq_len, state);
            
#ifdef TRANSFORMER_DEBUG
            cudaDeviceSynchronize();
            used_time = time_check_end();
            printf("%d :tensor+tensor time in %d layer: %f ms\n", 167+((num_layers-1)-i)*25, i+1, used_time);
#endif 
            
        }
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        ApplydropoutMask<<<320, 512>>>(dx1, logits_dx, d_xmask, probe, seq_len*state);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("293 :x dropout grad time: %f ms\n", used_time);
#endif


#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        add_grad_gpu<<<seq_len,n>>>(p_embed_grad, dx1, batch_size, seq_len, state);

        ApplydropoutMask_SmallSize<<<10,512>>>(p_embed_dw, p_embed_grad, d_pmask, probe, 5120);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("294 :p dropout grad time: %f ms\n", used_time);
#endif
        gradients_add_gpu_512_512<<<320, 512>>>(gradients_sum+1, p_embed_dw, seq_len*state);
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        embedding_lookup_grad_gpu<<<m, n>>>(embed_grad, dx1, d_xlabel, batch_size, seq_len, state, vocab_size);
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("295 :embedding grad time: %f ms\n", used_time);
#endif
        


#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        transpose_0123to0132_gpu<<<vocab_size, n>>>(logits_dw, logits_dw_trans, 1, vocab_size, 1, state);
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("296 :logits dw trans time: %f ms\n", used_time);
#endif
        

        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        tensor_add_tensor_gpu<<<vocab_size, n>>>(embed_add_out, logits_dw_trans, embed_grad, 1, vocab_size, state);
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("297 :tensor+tensor time: %f ms\n", used_time);
#endif
        
        gradients_add_gpu_512_512<<<256, 512>>>(gradients_sum, embed_add_out, vocab_size*state);

        for(int i=0;i<(num_layers*16+2);i++){
            cudaMemcpy(gradients_sumc+i, gradients_sum+i, sizeof(float), cudaMemcpyDeviceToHost);
            printf("%.25f %d\n", gradients_sumc[i], i);
            grad_sum_sum += gradients_sumc[i];
        }


        global_norm = sqrt(grad_sum_sum);
        norm_scale = clip_by_global_norm(global_norm, clip_norm);

        // printf("global_norm:%.10f\n", global_norm);
        // printf("norm_sacle:%.10f\n", norm_scale);

        lr = global_step * (1.0/1000) < 1 ? global_step * (1.0/1000) : 1;
        lr *= learning_rate;

        beta1_power = adam_got_beta_power(beta1, global_step);
        beta2_power = adam_got_beta_power(beta2, global_step);

        float lr_2 = adam_got_lr(lr, beta1_power, beta2_power);

#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        time_check_begin();
#endif
        adam_apply_gradients_gpu<<<1, n>>>(d_norm_ag, d_norm_adg, d_norm_a_g_mt, d_norm_a_g_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2, 
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_norm_ab, d_norm_adb, d_norm_a_b_mt, d_norm_a_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_norm_mg, d_norm_mdg, d_norm_m_g_mt, d_norm_m_g_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_qb, d_qdb, d_q_b_mt, d_q_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_kb, d_kdb, d_k_b_mt, d_k_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_vb, d_vdb, d_v_b_mt, d_v_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_ab, d_adb, d_a_b_mt, d_a_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<4, n>>>(d_m1b, d_m1db, d_m1_b_mt, d_m1_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*4*num_layers);

        adam_apply_gradients_gpu<<<1, n>>>(d_m2b, d_m2db, d_m2_b_mt, d_m2_b_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

        adam_apply_gradients_gpu<<<n, n>>>(d_qw, d_qdw, d_q_w_mt, d_q_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);

        adam_apply_gradients_gpu<<<n, n>>>(d_kw, d_kdw, d_k_w_mt, d_k_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);

        adam_apply_gradients_gpu<<<n, n>>>(d_vw, d_vdw, d_v_w_mt, d_v_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers); 

        adam_apply_gradients_gpu<<<n, n>>>(d_aw, d_adw, d_a_w_mt, d_a_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);   

        adam_apply_gradients_gpu<<<n*4, n>>>(d_m1w, d_m1dw, d_m1_w_mt, d_m1_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*4*num_layers);

        adam_apply_gradients_gpu<<<n*4, n>>>(d_m2w, d_m2dw, d_m2_w_mt, d_m2_w_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, state*state*4*num_layers); 

        adam_apply_gradients_gpu<<<seq_len, n>>>(d_pembed, p_embed_dw, d_p_embed_mt, d_p_embed_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, seq_len*state*num_layers); 

        adam_apply_gradients_gpu<<<vocab_size, n>>>(d_xembed, embed_add_out, d_x_embed_mt, d_x_embed_vt, 
                                        beta1, beta2, beta1_power, beta2_power, lr_2,
                                        epsilon, norm_scale, grad_scale, clip_sigma, vocab_size*state*num_layers);
        
#ifdef TRANSFORMER_DEBUG
        cudaDeviceSynchronize();
        used_time = time_check_end();
        printf("298 :Adam update time: %f ms\n", used_time);
#endif
        
    }

 


    return 0;
}