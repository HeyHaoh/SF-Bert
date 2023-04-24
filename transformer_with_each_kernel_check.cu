#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// #include <cblas.h>
// #include "helper_cuda.h"
#include "cuda_kernels.h"
#include "validate.h"
#include "common.h"
#include "embedding.h"
#include "layernorm.h"
#include "matmul.h"
#include "ewops.h"
#include "optimizer.h"

using namespace std;

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
    cudaMalloc((void **)&q_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&k_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&v_out, sizeof(float)*m*n*num_layers);

    float *q_bias_out, *k_bias_out, *v_bias_out;
    cudaMalloc((void **)&q_bias_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&k_bias_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&v_bias_out, sizeof(float)*m*n*num_layers);

    float *q_reshape, *k_reshape, *v_reshape;
    cudaMalloc((void **)&q_reshape, sizeof(float)*m*n);
    cudaMalloc((void **)&k_reshape, sizeof(float)*m*n);
    cudaMalloc((void **)&v_reshape, sizeof(float)*m*n);

    float *qk_out, *softmax_out, *sv_out, *sv_out_reshape;
    cudaMalloc((void **)&qk_out, sizeof(float)*batch_size*head_num*seq_len*seq_len);
    cudaMalloc((void **)&softmax_out, sizeof(float)*batch_size*head_num*seq_len*seq_len);
    cudaMalloc((void **)&sv_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&sv_out_reshape, sizeof(float)*m*n*num_layers);

    float *a_out, *a_bias_out;
    cudaMalloc((void **)&a_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&a_bias_out, sizeof(float)*m*n*num_layers);

    float *a_after_dropout, *add_1;
    cudaMalloc((void **)&a_after_dropout, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&add_1, sizeof(float)*m*n*num_layers);

    float *norm_m, *norm_m_mean, *norm_m_rstd;
    cudaMalloc((void **)&norm_m, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&norm_m_mean, sizeof(float)*m*num_layers);
    cudaMalloc((void **)&norm_m_rstd, sizeof(float)*m*num_layers);

    float *m1_out, *m1_gelu_out, *m1_bias_out;
    cudaMalloc((void **)&m1_out, sizeof(float)*m*n*4*num_layers);
    cudaMalloc((void **)&m1_gelu_out, sizeof(float)*m*n*4*num_layers);
    cudaMalloc((void **)&m1_bias_out, sizeof(float)*m*n*4*num_layers);

    float *m2_out, *m2_bias_out;
    cudaMalloc((void **)&m2_out, sizeof(float)*m*n*num_layers);
    cudaMalloc((void **)&m2_bias_out, sizeof(float)*m*n*num_layers);

    float *m_after_dropout;
    cudaMalloc((void **)&m_after_dropout, sizeof(float)*m*n*num_layers);

    float *add_2;
    cudaMalloc((void **)&add_2, sizeof(float)*m*n*num_layers);

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
    
    char xs_path[100];
    char temp[100];
    int *xs_int = (int*)malloc(sizeof(int) * m);
    unsigned char *xs_char = (unsigned char*)malloc(sizeof(unsigned char) * m);
    
    char ys_path[100];
    char y_temp[100];
    int *ys_int = (int*)malloc(sizeof(int) * m);
    unsigned char *ys_char = (unsigned char*)malloc(sizeof(unsigned char) * m);
    int *one_hot_ys = (int*)malloc(sizeof(int)*m*vocab_size);

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

    //cpu

    float *norm_a_g_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *norm_a_g_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(norm_a_g_mt, 0.0, state*num_layers);
    constantInit(norm_a_g_vt, 0.0, state*num_layers);

    float *norm_a_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *norm_a_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(norm_a_b_mt, 0.0, state*num_layers);
    constantInit(norm_a_b_vt, 0.0, state*num_layers);

    float *norm_m_g_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *norm_m_g_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(norm_m_g_mt, 0.0, state*num_layers);
    constantInit(norm_m_g_vt, 0.0, state*num_layers);

    float *norm_m_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *norm_m_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(norm_m_b_mt, 0.0, state*num_layers);
    constantInit(norm_m_b_vt, 0.0, state*num_layers);

    float *q_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *q_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(q_b_mt, 0.0, state*num_layers);
    constantInit(q_b_vt, 0.0, state*num_layers);

    float *k_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *k_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(k_b_mt, 0.0, state*num_layers);
    constantInit(k_b_vt, 0.0, state*num_layers);

    float *v_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *v_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(v_b_mt, 0.0, state*num_layers);
    constantInit(v_b_vt, 0.0, state*num_layers);

    float *q_w_mt = (float*)malloc(sizeof(float) * state * state * num_layers);
    float *q_w_vt = (float*)malloc(sizeof(float) * state * state * num_layers);
    constantInit(q_w_mt, 0.0, state*state*num_layers);
    constantInit(q_w_vt, 0.0, state*state*num_layers);

    float *k_w_mt = (float*)malloc(sizeof(float) * state * state * num_layers);
    float *k_w_vt = (float*)malloc(sizeof(float) * state * state * num_layers);
    constantInit(k_w_mt, 0.0, state*state*num_layers);
    constantInit(k_w_vt, 0.0, state*state*num_layers);

    float *v_w_mt = (float*)malloc(sizeof(float) * state * state * num_layers);
    float *v_w_vt = (float*)malloc(sizeof(float) * state * state * num_layers);
    constantInit(v_w_mt, 0.0, state*state*num_layers);
    constantInit(v_w_vt, 0.0, state*state*num_layers);

    float *a_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *a_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(a_b_mt, 0.0, state*num_layers);
    constantInit(a_b_vt, 0.0, state*num_layers);

    float *a_w_mt = (float*)malloc(sizeof(float) * state * state * num_layers);
    float *a_w_vt = (float*)malloc(sizeof(float) * state * state * num_layers);
    constantInit(a_w_mt, 0.0, state*state*num_layers);
    constantInit(a_w_vt, 0.0, state*state*num_layers);

    float *m1_b_mt = (float*)malloc(sizeof(float) * state * 4 * num_layers);
    float *m1_b_vt = (float*)malloc(sizeof(float) * state * 4 * num_layers);
    constantInit(m1_b_mt, 0.0, state*4*num_layers);
    constantInit(m1_b_vt, 0.0, state*4*num_layers);

    float *m1_w_mt = (float*)malloc(sizeof(float) * state * state * 4 * num_layers);
    float *m1_w_vt = (float*)malloc(sizeof(float) * state * state * 4 * num_layers);
    constantInit(m1_w_mt, 0.0, state*state*4*num_layers);
    constantInit(m1_w_vt, 0.0, state*state*4*num_layers);

    float *m2_w_mt = (float*)malloc(sizeof(float) * state * state * 4 * num_layers);
    float *m2_w_vt = (float*)malloc(sizeof(float) * state * state * 4 * num_layers);
    constantInit(m2_w_mt, 0.0, state*state*4*num_layers);
    constantInit(m2_w_vt, 0.0, state*state*4*num_layers);

    float *m2_b_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *m2_b_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(m2_b_mt, 0.0, state*num_layers);
    constantInit(m2_b_vt, 0.0, state*num_layers);

    float* p_embed_mt = (float*)malloc(sizeof(float) * seq_len * state);
    float* p_embed_vt = (float*)malloc(sizeof(float) * seq_len * state);
    constantInit(p_embed_mt, 0.0, seq_len*state);
    constantInit(p_embed_vt, 0.0, seq_len*state);

    float* x_embed_mt = (float*)malloc(sizeof(float) * vocab_size * state);
    float* x_embed_vt = (float*)malloc(sizeof(float) * vocab_size * state);
    constantInit(x_embed_mt, 0.0, vocab_size*state);
    constantInit(x_embed_vt, 0.0, vocab_size*state);



    //Embedding START
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

    int num_blocks, num_threads;
    num_blocks = m;
    num_threads = n;

    embedding_lookup_gpu<<<m, n>>>(embed_lookup, d_xembed, d_xlabel, batch_size, seq_len, state, vocab_size);

    float *embed_lookupc = (float*)malloc(sizeof(float) * batch_size * seq_len * state);
    cudaMemcpy(embed_lookupc, embed_lookup, sizeof(float)* batch_size * seq_len * state, cudaMemcpyDeviceToHost);

    float *embed_lookup_vali = (float*)malloc(sizeof(float) * batch_size * seq_len * state);
    embedding_lookup(embed_lookup_vali, x_embed, xs_int, batch_size, seq_len, head_num, size_per_head);

    // for(int i=0; i<m*n; i++){
    //     printf("%.25f , %.25f\n", embed_lookupc[i], embed_lookup_vali[i]);
    //     if(fabs(embed_lookupc[i] - embed_lookup_vali[i]) > 0.00000001){
    //         printf("error! gpu:%.25f  cpu:%.25f  loc:%d \n", embed_lookupc[i], embed_lookup_vali[i], i);
    //     }
    // }
    // return 0;

    int *entropy_random = (int*)malloc(sizeof(int)*batch_size*seq_len*state*3);

    for(int i=0;i<batch_size*seq_len*state*3;i++){
        entropy_random[i] = rand();
    }

    int *d_entropy;
    cudaMalloc((void **)&d_entropy, sizeof(int)*batch_size*seq_len*state*3);
    cudaMemcpy(d_entropy, entropy_random, sizeof(int)*batch_size*seq_len*state*3, cudaMemcpyHostToDevice);

    unsigned int *d_xmask, *d_pmask;
    cudaMalloc((void **)&d_xmask, sizeof(unsigned int)*seq_len*state);
    cudaMalloc((void **)&d_pmask, sizeof(unsigned int)*5120);
    
    unsigned int *d_amask, *d_mmask;
    cudaMalloc((void **)&d_amask, sizeof(unsigned int)*seq_len*state);
    cudaMalloc((void **)&d_mmask, sizeof(unsigned int)*seq_len*state);

    GendropoutMask<<<m, n>>>(d_entropy, d_xmask, probe, seq_len*state);

    unsigned int *x_mask = (unsigned int*)malloc(sizeof(unsigned int) * seq_len * state);
    cudaMemcpy(x_mask, d_xmask, sizeof(unsigned int) * seq_len * state, cudaMemcpyDeviceToHost);

    // int sumof0 =0;
    // int sumofnon0 = 0;

    // for(int i=0;i<seq_len*n;i++){
    //     if(x_mask[i] == 0){
    //         sumof0++;
    //     }else{
    //         sumofnon0++;
    //     }
    // }
    // printf("sum of 0:%d sum 0f non 0:%d \n", sumof0, sumofnon0);

    // return 0 ;

    ApplydropoutMask<<<320, 512>>>(x_after_dropout, embed_lookup, d_xmask, probe, seq_len*state);

    float *x_after_dropoutc = (float*)malloc(sizeof(float)*m*n);
    cudaMemcpy(x_after_dropoutc, x_after_dropout, sizeof(float)*m*n, cudaMemcpyDeviceToHost); 

    float *x_after_dropout_vali = (float*)malloc(sizeof(float)*m*n);
    dropout1(x_after_dropout_vali, embed_lookupc, x_mask, probe, seq_len*state, batch_size, seq_len, state);

    // for(int i=0; i<m*n; i++){
    //     // printf("gpu:%.15f  cpu:%.15f  loc:%d\n", x_after_dropoutc[i], x_after_dropout_vali[i], i);
    //     if(fabs(x_after_dropoutc[i] - x_after_dropout_vali[i]) > 0.0000001){
    //         printf("error! gpu:%.25f  cpu:%.25f  loc:%d \n", x_after_dropoutc[i], x_after_dropout_vali[i], i);
    //     }
    // }   

    // return 0;

    GendropoutMask<<<seq_len, n>>>(d_entropy, d_pmask, probe, seq_len*state);

    unsigned int* p_mask = (unsigned int*)malloc(sizeof(unsigned int) * 5120);
    cudaMemcpy(p_mask, d_pmask, sizeof(unsigned int)*5120, cudaMemcpyDeviceToHost);

    // sumof0 = 0;
    // sumofnon0 = 0;
    // for(int i=0;i<5120;i++){
    //     if(p_mask[i] == 0){
    //         sumof0++;
    //     }else{
    //         sumofnon0++;
    //     }
    // }
    // printf("sum of 0:%d sum 0f non 0:%d \n", sumof0, sumofnon0);

    ApplydropoutMask_SmallSize<<<10, 512>>>(p_after_dropout, d_pembed, d_pmask, probe, 5120);

    float *p_after_dropoutc = (float*)malloc(sizeof(float)*seq_len*state);
    cudaMemcpy(p_after_dropoutc, p_after_dropout, sizeof(float)*seq_len*state, cudaMemcpyDeviceToHost);

    float *p_after_dropout_vali = (float*)malloc(sizeof(float)*seq_len*state);
    // dropout(p_after_dropout_vali, p_embed, p_mask, probe, 5120);
    dropout2(p_after_dropout_vali, p_embed, p_mask, probe, 5120, n);

    // for(int i=0; i<seq_len*state; i++){
    //     // printf("gpu:%.25f  cpu:%.25f  loc:%d\n", p_after_dropoutc[i], p_after_dropout_vali[i], i);
    //     if(fabs(p_after_dropoutc[i] - p_after_dropout_vali[i]) > 0.0000001){
    //         printf("error! gpu:%f  cpu:%f  loc:%d \n", p_after_dropoutc[i], p_after_dropout_vali[i], i);
    //     }
    // }
    
    // return 0;
    
    tensor_add_matrix_gpu<<<m, n>>>(embed_add, x_after_dropout, p_after_dropout, batch_size, seq_len, state);

    float *embed_addc1 = (float*)malloc(sizeof(float)* m*n);
    cudaMemcpy(embed_addc1, embed_add, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    float *embed_add_vali = (float*)malloc(sizeof(float)* m*n);
    tensor_add_matrix(embed_add_vali, x_after_dropoutc, p_after_dropoutc, batch_size, seq_len, state);

    // for(int i=0; i<m*n; i++){
    //     // printf("gpu:%.25f  cpu:%.25f loc:%d\n", embed_addc1[i], embed_add_vali[i], i);
    //     if(fabs(embed_addc1[i] - embed_add_vali[i]) > 0.0000001){
    //         printf("error! gpu:%.25f  cpu:%.25f  loc:%d \n", embed_addc1[i], embed_add_vali[i], i);
    //         // break;
    //     }
    // }

    // return 0;

    GendropoutMask<<<m, n>>>(d_entropy, d_amask, probe, seq_len*state); 
    
    unsigned int *a_mask = (unsigned int*)malloc(sizeof(unsigned int)*seq_len*state);
    cudaMemcpy(a_mask, d_amask, sizeof(unsigned int)*seq_len*n, cudaMemcpyDeviceToHost);

    // cudaMemcpy(entropy_random, d_entropy, sizeof(int)*m*n*3, cudaMemcpyDeviceToHost);

    // for(int i=0;i<m*n*3;i++){
    //     printf("%d\n", entropy_random[i]);
    // }
    // return 0;

    // sumof0 =0;
    // sumofnon0 = 0;

    // for(int i=0;i<seq_len*n;i++){
    //     if(a_mask[i] == 0){
    //         sumof0++;
    //     }else{
    //         sumofnon0++;
    //     }
    // }
    // printf("sum of 0:%d sum 0f non 0:%d \n", sumof0, sumofnon0);


    GendropoutMask<<<m, n>>>(d_entropy, d_mmask, probe, seq_len*state);

    unsigned int *m_mask = (unsigned int*)malloc(sizeof(unsigned int)*seq_len*state);
    cudaMemcpy(m_mask, d_mmask, sizeof(unsigned int)*seq_len*n, cudaMemcpyDeviceToHost);

    // sumof0 =0;
    // sumofnon0 = 0;

    // for(int i=0;i<seq_len*n;i++){
    //     if(m_mask[i] == 0){
    //         sumof0++;
    //     }else{
    //         sumofnon0++;
    //     }
    // }
    // printf("sum of 0:%d sum 0f non 0:%d \n", sumof0, sumofnon0);


    // return 0;

    cudaFree(d_entropy);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *m1_gelu_outc = (float*)malloc(sizeof(float)*m*n*4);
    float *m1_bias_outc = (float*)malloc(sizeof(float)*m*n*4);
    float *add_1c = (float*)malloc(sizeof(float)*m*n);
    float *sv_reshapec = (float*)malloc(sizeof(float)*m*n);
    float *v_reshapec = (float*)malloc(sizeof(float)*m*n);
    float *softmax_outc = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
    float *q_reshapec = (float*)malloc(sizeof(float)*m*n);
    float *k_reshapec = (float*)malloc(sizeof(float)*m*n);
    float *norm_ac = (float*)malloc(sizeof(float)*m*n);
    float *norm_a_meanc = (float*)malloc(sizeof(float)*m);
    float *norm_a_rstdc = (float*)malloc(sizeof(float)*m);
    float *norm_a_vali = (float*)malloc(sizeof(float)*m*n);
    float *norm_a_mean_vali = (float*)malloc(sizeof(float)*m);
    float *norm_a_rstd_vali = (float*)malloc(sizeof(float)*m);
    float *q_reshape_vali = (float*)malloc(sizeof(float)*m*n);
    float *k_reshape_vali = (float*)malloc(sizeof(float)*m*n);
    float *v_reshape_vali = (float*)malloc(sizeof(float)*m*n);
    float *softmax_out_vali = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
    float *add_1_vali = (float*)malloc(sizeof(float)*m*n);
    float *norm_mc = (float*)malloc(sizeof(float)*m*n);
    float *norm_m_meanc = (float*)malloc(sizeof(float)*m);
    float *norm_m_rstdc = (float*)malloc(sizeof(float)*m);  
    float *norm_m_vali = (float*)malloc(sizeof(float)*m*n);
    float *norm_m_mean_vali = (float*)malloc(sizeof(float)*m);
    float *norm_m_rstd_vali = (float*)malloc(sizeof(float)*m);
    float *m1_bias_out_vali = (float*)malloc(sizeof(float)*m*n*4);
    float *m1_gelu_out_vali = (float*)malloc(sizeof(float)*m*n*4);
    float *embed_addc2 = (float*)malloc(sizeof(float)*m*n);
    float *embed_add2_vali = (float*)malloc(sizeof(float)*m*n);
    float *sv_reshape_vali = (float*)malloc(sizeof(float)*m*n);
    
    for(int i=0;i<num_layers;i++){

        layernorm_gpu<<<m, n>>>(norm_a, norm_a_mean, norm_a_rstd, embed_add, d_norm_ag, d_norm_ab, batch_size, seq_len, state);
        // layernorm_gpu1<<<20, n>>>(norm_a, norm_a_mean, norm_a_rstd, embed_add, d_norm_ag, d_norm_ab, batch_size, seq_len, state);

        // cudaMemcpy(norm_ac, norm_a, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_a_meanc, norm_a_mean, sizeof(float)*m, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_a_rstdc, norm_a_rstd, sizeof(float)*m, cudaMemcpyDeviceToHost);
        
        // layernorm(norm_a_vali, norm_a_mean_vali, norm_a_rstd_vali, embed_add_vali, norm_a_g, norm_a_b, batch_size, seq_len, state);

        // for(int j=0; j<m*n; j++){
        //     // printf("gpu:%.25f  cpu:%.25f  loc:%d \n", norm_ac[j], norm_a_vali[j], j);
        //     if(fabs(norm_ac[j] - norm_a_vali[j]) > 0.000001){
        //         printf("error! gpu:%.25f  cpu:%.25f  loc:%d \n", norm_ac[j], norm_a_vali[j], j);
        //         // break;
        //     }
        // }

        // for(int j=0; j<m; j++){
        //     printf("gpu:%.25f  cpu:%.25f  loc:%d \n", norm_a_meanc[j], norm_a_mean_vali[j], j);
        //     if(fabs(norm_a_meanc[j] - norm_a_mean_vali[j]) > 0.000000000001){
        //         printf("error! gpu:%.25f  cpu:%.25f  loc:%d \n", norm_a_meanc[j], norm_a_mean_vali[j], j);
        //     }
        // }

        // for(int j=0; j<m; j++){
        //     printf("gpu:%.15f  cpu:%.15f  loc:%d \n", norm_a_rstdc[j], norm_a_rstd_vali[j], j);
        //     if(fabs(norm_a_rstdc[j] - norm_a_rstd_vali[j]) > 0.0001){
        //         printf("ERROR! gpu:%.15f  cpu:%.15f  loc:%d \n", norm_a_rstdc[j], norm_a_rstd_vali[j], j);
        //     }
        // }

        // return 0;

        //cublas里面矩阵全部都是列主序，但是我的代码都是行主序。这点要注意。
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    state, m, state,
                    &alpha, 
                    d_qw, state,
                    norm_a, state,
                    &beta, 
                    q_out, state);

        // float *q_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(q_outc, q_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *q_out_vali = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a_vali, n, q_w, n, 0, q_out_vali, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.25f, cpu:%.25f, loc:%d\n", q_outc[j], q_out_vali[j], j);
        //     if(fabs(q_outc[j] - q_out_vali[j]) > 0.000001){
        //         printf("ERROR! gpu:%.25f, cpu:%.25f, loc:%d\n", q_outc[j], q_out_vali[j], j);
        //     }
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    state, (batch_size*seq_len), state, 
                    &alpha, 
                    d_kw, state, 
                    norm_a, state, 
                    &beta, 
                    k_out, state);

        // float *k_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(k_outc, k_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *k_out_vali = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a_vali, n, k_w, n, 0, k_out_vali, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", k_outc[j], k_out_vali[j], j);
        //     if(fabs(k_outc[j] - k_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", k_outc[j], k_out_vali[j], j);
        //     }
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    state, (batch_size*seq_len), state, 
                    &alpha, 
                    d_vw, state, 
                    norm_a, state, 
                    &beta, 
                    v_out, state);
        
        // float *v_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(v_outc, v_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *v_out_vali = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a_vali, n, v_w, n, 0, v_out_vali, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", v_outc[j], v_out_vali[j], j);
        //     if(fabs(v_outc[j] - v_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", v_outc[j], v_out_vali[j], j);
        //     }
        // }

        // return 0;

        tensor_add_vector_gpu<<<m, n>>>(q_bias_out, q_out, d_qb, batch_size, seq_len, state);

        // float *q_bias_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(q_bias_outc, q_bias_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        
        // float *q_bias_out_vali = (float*)malloc(sizeof(float)*m*n);
        // bias(q_bias_out_vali, q_out_vali, q_b, n, m);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", q_bias_outc[j], q_bias_out_vali[j], j);
        //     if(fabs(q_bias_outc[j] - q_bias_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", q_bias_outc[j], q_bias_out_vali[j], j);
        //     }
        // }
        // return 0;

        tensor_add_vector_gpu<<<m, n>>>(k_bias_out, k_out, d_qb, batch_size, seq_len, state);

        // float *k_bias_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(k_bias_outc, k_bias_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        
        // float *k_bias_out_vali = (float*)malloc(sizeof(float)*m*n);
        // bias(k_bias_out_vali, k_out_vali, k_b, n, m);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", k_bias_outc[j], k_bias_out_vali[j], j);
        //     if(fabs(k_bias_outc[j] - k_bias_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", k_bias_outc[j], k_bias_out_vali[j], j);
        //     }
        // }
        // return 0;
        

        tensor_add_vector_gpu<<<m, n>>>(v_bias_out, v_out, d_qb, batch_size, seq_len, state);

        // float *v_bias_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(v_bias_outc, v_bias_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        
        // float *v_bias_out_vali = (float*)malloc(sizeof(float)*m*n);
        // bias(v_bias_out_vali, v_out_vali, v_b, n, m);

        // for(int j=0;j<m*n;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", v_bias_outc[j], v_bias_out_vali[j], j);
        //     if(fabs(v_bias_outc[j] - v_bias_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", v_bias_outc[j], v_bias_out_vali[j], j);
        //     }
        // }
        // return 0;
        
        transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(q_bias_out, q_reshape, batch_size, seq_len, head_num, size_per_head);
        transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(k_bias_out, k_reshape, batch_size, seq_len, head_num, size_per_head);
        transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(v_bias_out, v_reshape, batch_size, seq_len, head_num, size_per_head);
    
        // cudaMemcpy(q_reshapec, q_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // cudaMemcpy(k_reshapec, k_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // cudaMemcpy(v_reshapec, v_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // transpose_0123to0213(q_bias_out_vali, q_reshape_vali, batch_size, seq_len, head_num, size_per_head);
        // transpose_0123to0213(k_bias_out_vali, k_reshape_vali, batch_size, seq_len, head_num, size_per_head);
        // transpose_0123to0213(v_bias_out_vali, v_reshape_vali, batch_size, seq_len, head_num, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     printf("gpu:%.15f, cpu:%.15f, loc:%d\n", q_reshapec[j], q_reshape_vali[j], j);
        //     if(fabs(q_reshapec[j] - q_reshape_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", q_reshapec[j], q_reshape_vali[j], j);
        //     }
        // }

        // for(int j=0;j<m*n;j++){
        //     printf("gpu:%.15f, cpu:%.15f, loc:%d\n", k_reshapec[j], k_reshape_vali[j], j);
        //     if(fabs(k_reshapec[j] - k_reshape_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", k_reshapec[j], k_reshape_vali[j], j);
        //     }
        // }

        // for(int j=0;j<m*n;j++){
        //     printf("gpu:%.15f, cpu:%.15f, loc:%d\n", v_reshapec[j], v_reshape_vali[j], j);
        //     if(fabs(v_reshapec[j] - v_reshape_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", v_reshapec[j], v_reshape_vali[j], j);
        //     }
        // }

        // return 0;

        // cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        //                             seq_len, seq_len, size_per_head,
        //                             &alpha,
        //                             k_reshape, size_per_head, seq_len*size_per_head,
        //                             q_reshape, size_per_head, seq_len*size_per_head,
        //                             &beta,  
        //                             qk_out, seq_len, seq_len*seq_len,
        //                             batch_size*head_num
        //                             );

        offset = 0;
        offset1 = 0;

        for(int j=0;j<batch_size*head_num;j++){

            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            seq_len, seq_len, size_per_head, 
            &alpha, 
            k_reshape+offset, size_per_head, 
            q_reshape+offset, size_per_head, 
            &beta, 
            qk_out+offset1, seq_len);
            offset+= (seq_len*size_per_head);
            offset1 += (seq_len*seq_len);
        }

        // float *qk_outc = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // cudaMemcpy(qk_outc, qk_out, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);

        // float *qk_out_vali = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // multiheadmatmulNT<float, float, float>(qk_out_vali, q_reshape_vali, k_reshape_vali, batch_size, head_num, seq_len, size_per_head, seq_len);
        
        // for(int j=0;j<batch_size*head_num*seq_len*seq_len;j++){
        //     // printf("gpu:%.15f, cpu:%.15f, loc:%d\n", qk_outc[j], qk_out_vali[j], j);
        //     if(fabs(qk_outc[j] - qk_out_vali[j]) > 0.00001){
        //         printf("ERROR! gpu:%.15f, cpu:%.15f, loc:%d\n", qk_outc[j], qk_out_vali[j], j);
        //     }
        // }

        // return 0;


        softmax_gpu<<<(batch_size*head_num), seq_len>>>(softmax_out, qk_out, scaler, batch_size, head_num, seq_len);

        // cudaMemcpy(softmax_outc, softmax_out, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);
        // softmax(softmax_out_vali, qk_out_vali, scaler, seq_len, batch_size, head_num);

        // for(int j=0;j<batch_size*head_num*seq_len*seq_len;j++){
        //     // printf("%.15f %.15f loc:%d \n", softmax_outc[j], softmax_out_vali[j], j);
        //     if(fabs(softmax_outc[j] - softmax_out_vali[j]) > 0.000001){
        //         printf("%.15f %.15f loc:%d \n", softmax_outc[j], softmax_out_vali[j], j);
        //     }

        // }
        
        // return 0;

        // cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
        //                             seq_len, seq_len, size_per_head,
        //                             &alpha,
        //                             v_reshape, size_per_head, seq_len*size_per_head,
        //                             softmax_out, seq_len, seq_len*seq_len,
        //                             &beta,
        //                             sv_out, size_per_head, seq_len*size_per_head,
        //                             batch_size*head_num
        //                             );

        offset = 0;
        offset1 = 0;

        for(int j=0;j<batch_size*head_num;j++){
            
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    size_per_head, seq_len, seq_len, 
                    &alpha, 
                    v_reshape+offset, size_per_head, 
                    softmax_out+offset1,seq_len, 
                    &beta, 
                    sv_out+offset, size_per_head);

            offset1+= (seq_len*seq_len);
            offset += (seq_len*size_per_head); 
        }
        
        // float *sv_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(sv_outc, sv_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *sv_out_vali = (float*)malloc(sizeof(float)*m*n);
        // multiheadmatmulNN<float, float, float>(sv_out_vali, softmax_out_vali, v_reshape_vali, batch_size, head_num, seq_len, seq_len, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.15f %.15f loc:%d \n", sv_outc[j], sv_out_vali[j], j);
        //     if(fabs(sv_outc[j] - sv_out_vali[j]) > 0.00001){
        //         printf("%.15f %.15f loc:%d \n", sv_outc[j], sv_out_vali[j], j);
        //     }
        // }
        // return 0;

        transpose_0213to0123_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(sv_out, sv_out_reshape, batch_size, seq_len, head_num, size_per_head);
        // cudaMemcpy(sv_reshapec, sv_out_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        
        // transpose_0123to0213(sv_out_vali, sv_reshape_vali, batch_size, head_num, seq_len, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     printf("%.15f %.15f loc:%d \n", sv_reshapec[j], sv_reshape_vali[j], j);
        //     if(fabs(sv_reshapec[j]-sv_reshape_vali[j]) > 0.00001)
        //         printf("ERROR! %f %f loc:%d \n", sv_reshapec[j], sv_reshape_vali[j], j);
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    state, (batch_size*seq_len), state, 
                    &alpha, 
                    d_aw, state, 
                    sv_out_reshape,state, 
                    &beta, 
                    a_out, state);

        // float *a_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(a_outc, a_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *a_out_vali = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, sv_reshape_vali, n, a_w, n, 0, a_out_vali, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.15f %.15f loc:%d \n", a_outc[j], a_out_vali[j], j);
        //     if(fabs(a_outc[j]-a_out_vali[j]) > 0.000001){
        //         printf("%.15f %.15f loc:%d \n", a_outc[j], a_out_vali[j], j);
        //     }
        // }

        // return 0;

        tensor_add_vector_gpu<<<m, n>>>(a_bias_out, a_out, d_ab, batch_size, seq_len, state);

        // float *a_bias_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(a_bias_outc, a_bias_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *a_bias_out_vali = (float*)malloc(sizeof(float)*m*n);
        // bias(a_bias_out_vali, a_out_vali, a_b, n, m);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.15f %.15f loc:%d \n", a_bias_outc[j], a_bias_out_vali[j], j);
        //     if(fabs(a_bias_outc[j] - a_bias_out_vali[j]) > 0.000001){
        //         printf("%.15f %.15f loc:%d \n", a_bias_outc[j], a_bias_out_vali[j], j);
        //     }
        // }

        // return 0;

        
    

        ApplydropoutMask<<<320, 512>>>(a_after_dropout, a_bias_out, d_amask, probe, seq_len*state);

        // float *a_after_dropoutc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(a_after_dropoutc, a_after_dropout, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *a_after_dropout_vali = (float*)malloc(sizeof(float)*m*n);
        // dropout1(a_after_dropout_vali, a_bias_out_vali, a_mask, probe, seq_len*state, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     // printf("%.15f %.15f loc:%d \n", a_after_dropoutc[j], a_after_dropout_vali[j], j);
        //     if(fabs(a_after_dropoutc[j] - a_after_dropout_vali[j]) > 0.00001){
        //         printf("%.15f %.15f loc:%d \n", a_after_dropoutc[j], a_after_dropout_vali[j], j);
        //     }
        // }

        // return 0; 


        tensor_add_tensor_gpu<<<m, n>>>(add_1, embed_add, a_after_dropout, batch_size, seq_len, state);
        // cudaMemcpy(add_1c, add_1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // tensor_add_tensor(add_1_vali, embed_add_vali, a_after_dropout_vali, batch_size, seq_len, state);
        
        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", add_1c[j], add_1_vali[j], j);
        //     if(fabs(add_1c[j] - add_1_vali[j]) > 0.000001){
        //         printf("%.25f %.25f loc:%d \n", add_1c[j], add_1_vali[j], j);
        //     }
        // }

        // return 0; 

        //HERE

        layernorm_gpu<<<m, n>>>(norm_m, norm_m_mean, norm_m_rstd, add_1, d_norm_mg, d_norm_mb, batch_size, seq_len, state);

        // cudaMemcpy(norm_mc, norm_m, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_m_meanc, norm_m_mean, sizeof(float)*m, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_m_rstdc, norm_m_rstd, sizeof(float)*m, cudaMemcpyDeviceToHost);

        // layernorm(norm_m_vali, norm_m_mean_vali, norm_m_rstd_vali, add_1_vali, norm_m_g, norm_m_b, batch_size, seq_len, state);

        // for(int j=0; j<m*n; j++){
        //     // printf("gpu:%.15f  cpu:%.15f  loc:%d \n", norm_mc[j], norm_m_vali[j], j);
        //     if(fabs(norm_mc[j] - norm_m_vali[j]) > 0.000001){
        //         printf("error! gpu:%f  cpu:%f  loc:%d \n", norm_mc[j], norm_m_vali[j], j);
        //         // break;
        //     }
        // }

        // for(int j=0; j<m; j++){
        //     printf("gpu:%.25f  cpu:%.25f  loc:%d \n", norm_m_meanc[j], norm_m_mean_vali[j], j);
        //     // if(fabs(norm_m_meanc[j] - norm_m_mean_vali[j]) > 0.000001){
        //     //     printf("error! gpu:%f  cpu:%f  loc:%d \n", norm_m_meanc[j], norm_m_mean_vali[j], j);
        //     // }
        // }

        // for(int j=0; j<m; j++){
        //     printf("gpu:%.15f  cpu:%.15f  loc:%d \n", norm_m_rstdc[j], norm_m_rstd_vali[j], j);
        //     if(fabs(norm_m_rstdc[j] - norm_m_rstd_vali[j]) > 0.0001){
        //         printf("ERROR! gpu:%.15f  cpu:%.15f  loc:%d \n", norm_m_rstdc[j], norm_m_rstd_vali[j], j);
        //     }
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n*4, m, n, 
                    &alpha, 
                    d_m1w, n*4, 
                    norm_m, n, 
                    &beta, 
                    m1_out, n*4);
        
        // float *m1_outc = (float*)malloc(sizeof(float)*m*n*4);
        // cudaMemcpy(m1_outc, m1_out, sizeof(float)*m*n*4, cudaMemcpyDeviceToHost);

        // float *m1_out_vali = (float*)malloc(sizeof(float)*m*n*4);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n*4, n, 1, norm_m_vali, n, m1_w, n*4, 0, m1_out_vali, n*4);

        // for(int j=0;j<m*n*4;j++){
        //     // printf("%.15f %.15f loc:%d \n", m1_outc[j], m1_out_vali[j], j);
        //     if(fabs(m1_outc[j] - m1_out_vali[j]) > 0.00001){
        //         printf("%.15f %.15f loc:%d \n", m1_outc[j], m1_out_vali[j], j);
        //         // break;
        //     }
        // }

        // return 0;

        tensor_add_vector_gpu_2048<<<m, n>>>(m1_bias_out, m1_out, d_m1b, batch_size, seq_len, n*4);
        // cudaMemcpy(m1_bias_outc, m1_bias_out, sizeof(float)*m*n*4, cudaMemcpyDeviceToHost);

        // bias(m1_bias_out_vali, m1_out_vali, m1_b, n*4, m);

        // for(int j=0;j<m*n*4;j++){
        //     // printf("%.15f %.15f loc:%d \n", m1_bias_outc[j], m1_bias_out_vali[j], j);
        //     if(fabs(m1_bias_outc[j] - m1_bias_out_vali[j]) > 0.00001){
        //         printf("%.15f %.15f loc:%d \n", m1_bias_outc[j], m1_bias_out_vali[j], j);
        //         break;
        //     }
        // }

        // return 0;

        
        gelu_gpu<<<m, n>>>(m1_gelu_out, m1_bias_out, batch_size, seq_len, state*4);
        
        // cudaMemcpy(m1_gelu_outc, m1_gelu_out, sizeof(float)*m*n*4, cudaMemcpyDeviceToHost);

        // gelu(m1_gelu_out_vali, m1_bias_out_vali, batch_size, seq_len, n*4);

        // for(int j=0;j<m*n*4;j++){

        //     // printf("%.25f %.25f loc:%d \n", m1_gelu_outc[j], m1_gelu_out_vali[j], j);
        //     if(fabs(m1_gelu_outc[j] - m1_gelu_out_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d \n", m1_gelu_outc[j], m1_gelu_out_vali[j], j);
        //     }

        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    state, (batch_size*seq_len), state*4, 
                    &alpha, 
                    d_m2w, state, 
                    m1_gelu_out, state*4, 
                    &beta, 
                    m2_out, state);
        
        // float *m2_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(m2_outc, m2_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *m2_out_vali = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n*4, 1, m1_gelu_out_vali, n*4, m2_w, n, 0, m2_out_vali, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", m2_outc[j], m2_out_vali[j], j);
        //     if(fabs(m2_outc[j] - m2_out_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d \n", m2_outc[j], m2_out_vali[j], j);
        //     }
        // }

        // return 0;

        tensor_add_vector_gpu<<<m, n>>>(m2_bias_out, m2_out, d_m2b, batch_size, seq_len, state);

        // float *m2_bias_outc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(m2_bias_outc, m2_bias_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *m2_bias_out_vali = (float*)malloc(sizeof(float)*m*n);
        // bias(m2_bias_out_vali, m2_out_vali, m2_b, n, m);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", m2_bias_outc[j], m2_bias_out_vali[j], j);
        //     if(fabs(m2_bias_outc[j] - m2_bias_out_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d \n", m2_bias_outc[j], m2_bias_out_vali[j], j);
        //     }
        // }

        // return 0;

        ApplydropoutMask<<<320, 512>>>(m_after_dropout, m2_bias_out, d_mmask, probe, seq_len*state);

        // float *m_after_dropoutc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(m_after_dropoutc, m_after_dropout, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *m_after_dropout_vali = (float*)malloc(sizeof(float)*m*n);
        // dropout1(m_after_dropout_vali, m2_bias_out_vali, m_mask, probe, seq_len*state, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", m_after_dropoutc[j], m_after_dropout_vali[j], j);
        //     if(fabs(m_after_dropoutc[j] - m_after_dropout_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d \n", m_after_dropoutc[j], m_after_dropout_vali[j], j);
        //     }
        // }

        // return 0;

        tensor_add_tensor_gpu<<<m, n>>>(embed_add+(i+1)*m*n, add_1, m_after_dropout, batch_size, seq_len, state);
        // cudaMemcpy(embed_addc2, embed_add+(i+1)*m*n, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // tensor_add_tensor(embed_add2_vali, add_1_vali, m_after_dropout_vali, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", embed_addc2[j], embed_add2_vali[j], j);
        //     if(fabs(embed_addc2[j] - embed_add2_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d \n", embed_addc2[j], embed_add2_vali[j], j);
        //     }
        // }

        // return 0;

    }

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                vocab_size, m, n,
                &alpha,
                d_xembed, n,
                embed_add+(num_layers*m*n), n,
                &beta,
                logits, vocab_size);

    // float *logitsc = (float*)malloc(sizeof(float)*m*vocab_size);
    // cudaMemcpy(logitsc, logits, sizeof(float)*m*vocab_size, cudaMemcpyDeviceToHost);

    // float *logits_vali = (float*)malloc(sizeof(float)*m*vocab_size);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, vocab_size, n, 1, embed_add2_vali, n, x_embed, n, 0, logits_vali, vocab_size);

    // for(int j=0;j<m*vocab_size;j++){
    //     // printf("%.25f %.25f loc:%d \n", logitsc[j], logits_vali[j], j);
    //     if(fabs(logitsc[j] - logits_vali[j]) > 0.00001){
    //         printf("%.25f %.25f loc:%d \n", logitsc[j], logits_vali[j], j);
    //     }
    // }

    // return 0;

    softmax_cross_entropy_with_logits_gpu<<<m, 256>>>(softmax_logits, d_loss, logits, d_ylabel, batch_size, seq_len, vocab_size);
    
    // float *softmax_logitsc = (float*)malloc(sizeof(float)*m*vocab_size);
    // cudaMemcpy(softmax_logitsc, softmax_logits, sizeof(float)*m*vocab_size, cudaMemcpyDeviceToHost);

    // float *softmax_logits_vali = (float*)malloc(sizeof(float)*m*vocab_size);
    
    // float loss_vali;
    // loss_vali = softmax_cross_entropy_with_logits(softmax_logits_vali, logits_vali, one_hot_ys, batch_size, seq_len, vocab_size);

    // for(int j=0;j<m*vocab_size;j++){
    //     // printf("%.25f %.25f loc:%d \n", softmax_logitsc[j], softmax_logits_vali[j], j);
    //     if(fabs(softmax_logitsc[j] - softmax_logits_vali[j]) > 0.00001){
    //         printf("%.25f %.25f loc:%d \n", softmax_logitsc[j], softmax_logits_vali[j], j);
    //     }
    // }

    // return 0;

    float *loss_val;
    cudaMalloc((void **)&loss_val, sizeof(float));

    loss_add<<<1, n>>>(loss_val, d_loss, m);

    float *loss_c = (float*)malloc(sizeof(float));
    cudaMemcpy(loss_c, loss_val, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("%f %f\n", loss_c[0], loss_vali);
    printf("%f \n", loss_c[0]);

    // return 0;

    //forward done

    cross_entropy_grad_gpu<<<m, vocab_size>>>(entropy_grad, softmax_logits, batch_size, seq_len, vocab_size);

    // float* entropy_gradc = (float*)malloc(sizeof(float)*m*vocab_size);
    // cudaMemcpy(entropy_gradc, entropy_grad, sizeof(float)*m*vocab_size, cudaMemcpyDeviceToHost);

    // float* entropy_grad_vali = (float*)malloc(sizeof(float)*m*vocab_size);
    // cross_entropy_grad(entropy_grad_vali, softmax_logits_vali, batch_size, seq_len, vocab_size);

    // for(int j=0;j<m*vocab_size;j++){
    //     // printf("%.25f %.25f loc:%d \n", entropy_gradc[j], entropy_grad_vali[j], j);
    //     if(fabs(entropy_gradc[j] - entropy_grad_vali[j]) > 0.00000001){
    //         printf("%.25f %.25f loc:%d \n", entropy_gradc[j], entropy_grad_vali[j], j);
    //     }
    // }

    // return 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, vocab_size,
                &alpha,
                d_xembed, n,
                entropy_grad, vocab_size,
                &beta,
                logits_dx+(m*n), n);

    // float *logits_dxc1 = (float*)malloc(sizeof(float)*m*n);
    // cudaMemcpy(logits_dxc1, logits_dx+m*n, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // float *logits_dx_vali = (float*)malloc(sizeof(float)*m*n);   
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, vocab_size, 1, entropy_grad_vali, vocab_size, x_embed, n, 0, logits_dx_vali, n);
    
    // for(int j=0;j<m*n;j++){
    //     // printf("%.25f %.25f loc:%d \n", logits_dxc1[j], logits_dx_vali[j], j);
    //     if(fabs(logits_dxc1[j] - logits_dx_vali[j]) > 0.0000000001){
    //         printf("%.25f %.25f loc:%d \n", logits_dxc1[j], logits_dx_vali[j], j);
    //     }
    // }

    // return 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                vocab_size, n, m,
                &alpha,
                entropy_grad, vocab_size,
                embed_add+m*n, n,
                &beta,
                logits_dw, vocab_size);

    
    // float *logits_dwc = (float*)malloc(sizeof(float)*n*vocab_size);
    // cudaMemcpy(logits_dwc, logits_dw, sizeof(float)*n*vocab_size, cudaMemcpyDeviceToHost);

    // float *logits_dw_vali = (float*)malloc(sizeof(float)*n*vocab_size);
    // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, vocab_size, m, 1, embed_add2_vali, n, entropy_grad_vali, vocab_size, 0, logits_dw_vali, vocab_size);

    // for(int j=0;j<vocab_size*n;j++){
    //     // printf("%.25f %.25f loc:%d \n", logits_dwc[j], logits_dw_vali[j], j);
    //     if(fabs(logits_dwc[j] - logits_dw_vali[j]) > 0.000001){
    //         printf("%.25f %.25f loc:%d \n", logits_dwc[j], logits_dw_vali[j], j);
    //         break;
    //     }
    // }

    // return 0;    


    float *dx1c = (float*)malloc(sizeof(float)*m*n);
    float *dx2c = (float*)malloc(sizeof(float)*m*n); 
    float *dx3c = (float*)malloc(sizeof(float)*m*n); 

    float *dx_41c = (float*)malloc(sizeof(float)*m*n*4);
    float *dx_42c = (float*)malloc(sizeof(float)*m*n*4);

    float *gradients_sum;
    cudaMalloc((void**)&gradients_sum, sizeof(float)*(16*num_layers+2));
    cudaMemset(gradients_sum, 0.0, sizeof(float)*(16*num_layers+2));

    float *gradients_sum_vali = (float*)malloc(sizeof(float)*num_layers*16+2);
    float *gradients_sumc = (float*)malloc(sizeof(float)*num_layers*16+2);

    for(int i=0;i<num_layers*16+2;i++){
        gradients_sum_vali[i] = 0.0;
        gradients_sumc[i] = 0.0;
    }



    float *logits_dxc2 = (float*)malloc(sizeof(float)*m*n);
    float *logits_dx2_vali = (float*)malloc(sizeof(float)*m*n);
    float *d_norm_adg_vali = (float*)malloc(sizeof(float)*n);
    float *d_norm_adb_vali = (float*)malloc(sizeof(float)*n);
    float *d_m1dbc = (float*)malloc(sizeof(float)*n*4);
    float *d_m1db_vali = (float*)malloc(sizeof(float)*n*4);
    float *d_qdw_vali = (float*)malloc(sizeof(float)*n*n);
    float *d_m1dw_vali = (float*)malloc(sizeof(float)*n*n*4);

    for(int i = num_layers-1; i>=0; i--){

        ApplydropoutMask<<<320, 512>>>(dx1, logits_dx+(m*n), d_mmask, probe, seq_len*state);
        // cudaMemcpy(dx1c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *m_after_dropout_grad_vali = (float*)malloc(sizeof(float)*m*n);
        // dropout1(m_after_dropout_grad_vali, logits_dx_vali, m_mask, probe, seq_len*state, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx1c[j], m_after_dropout_grad_vali[j], j);
        //     if(fabs(dx1c[j] - m_after_dropout_grad_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d \n", dx1c[j], m_after_dropout_grad_vali[j], j);
        //         break;
        //     }
        // }

        // return 0;          

        bias_grad_db_gpu<<<seq_len, n>>>(d_m2db, dx1, batch_size, seq_len, state);

        // float *d_m2dbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_m2dbc, d_m2db, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_m2db_vali = (float*)malloc(sizeof(float)*n);
        // bias_grad_db(d_m2db_vali, dx1c, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_m2dbc[j], d_m2db_vali[j], j);
        //     if(fabs(d_m2dbc[j] - d_m2db_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d\n", d_m2dbc[j] , d_m2db_vali[j], j);
        //         break;
        //     } 
        // }
        // return 0;

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+17+i*16, d_m2db, state);
        cudaMemcpy(gradients_sumc+17+i*16, gradients_sum+17+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[17+i*16] = gradients_add(d_m2db_vali, 1, state);
        // printf("%.15f %.15f\n", gradients_sumc[17+i*16], gradients_sum_vali[17+i*16]);

        // return 0;        

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    state*4, m, n, 
                    &alpha,
                    d_m2w, n,
                    dx1, n,
                    &beta,
                    dx_41, state*4);
        
        // cudaMemcpy(dx_41c, dx_41, sizeof(float)*m*n*4, cudaMemcpyDeviceToHost);

        // float *dx_41_vali = (float*)malloc(sizeof(float)*m*n*4);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n*4, n, 1, m_after_dropout_grad_vali, n, m2_w, n, 0, dx_41_vali, n*4);
        
        // for(int j=0;j<n*m*4;j++){
        //     printf("%.25f %.25f loc:%d \n", dx_41c[j], dx_41_vali[j], j);
        //     if(fabs(dx_41c[j] - dx_41_vali[j]) > 0.000000001){
        //         printf("%.25f %.25f loc:%d\n", dx_41c[j] , dx_41_vali[j], j);
        //         break;
        //     }

        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    state, state*4, m, 
                    &alpha,
                    dx1, n,
                    m1_gelu_out, n*4,
                    &beta,
                    d_m2dw, n); 

        // float *d_m2dwc = (float*)malloc(sizeof(float)*n*n*4);
        // cudaMemcpy(d_m2dwc, d_m2dw, sizeof(float)*n*n*4, cudaMemcpyDeviceToHost);

        // float *d_m2dw_vali = (float*)malloc(sizeof(float)*n*n*4);
        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n*4, n, m, 1, m1_gelu_out_vali, n*4, m_after_dropout_grad_vali, n, 0, d_m2dw_vali, n);

        // for(int j=0;j<n*n*4;j++){
        //     printf("%.25f %.25f loc:%d \n", d_m2dwc[j], d_m2dw_vali[j], j);
        //     if(fabs(d_m2dwc[j] - d_m2dw_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", d_m2dwc[j] , d_m2dw_vali[j], j);
        //         break;
        //     }

        // }

        // return 0;

        gradients_add_gpu_512_2048<<<2048, 512>>>(gradients_sum+16+i*16, d_m2dw, n*n*4);
        cudaMemcpy(gradients_sumc+16+i*16, gradients_sum+16+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[i*16+16] = gradients_add(d_m2dw_vali, state, state*4);
        // printf("%.15f %.15f\n", gradients_sumc[16+i*16], gradients_sum_vali[16+i*16]);

        // return 0;

        gelu_grad_gpu<<<m, n>>>(dx_42, dx_41, m1_bias_out, d_m1b, batch_size, seq_len, state*4);
        // cudaMemcpy(dx_42c, dx_42, sizeof(float)*m*n*4, cudaMemcpyDeviceToHost);

        // float *gelu_grad_vali = (float*)malloc(sizeof(float)*m*n*4);
        // gelu_grad(gelu_grad_vali, dx_41_vali, m1_bias_out_vali, m1_b, batch_size, seq_len, state*4);

        // for(int j=0;j<m*n*4;j++){
        //     // printf("%.25f %.25f loc:%d \n", dx_42c[j], gelu_grad_vali[j], j);
        //     if(fabs(dx_42c[j] - gelu_grad_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", dx_42c[j] , gelu_grad_vali[j], j);
        //         break;
        //     }
        // }
        // return 0;

        bias_grad_db_2048_gpu<<<seq_len, n>>>(d_m1db, dx_42, batch_size, seq_len, state*4);


        // cudaMemcpy(d_m1dbc, d_m1db, sizeof(float)*n*4, cudaMemcpyDeviceToHost);


        // bias_grad_db(d_m1db_vali, dx_42c, batch_size, seq_len, state*4);

        // for(int j=0;j<n*4;j++){
        //     printf("%.25f %.25f loc:%d \n", d_m1dbc[j], d_m1db_vali[j], j);
        //     if(fabs(d_m1dbc[j] - d_m1db_vali[j]) > 0.00001){
        //         printf("%.25f %.25f loc:%d\n", d_m1dbc[j] , d_m1db_vali[j], j);
        //         break;
        //     }    
        // }

        // return 0;

        gradients_add_gpu_2048<<<4, 512>>>(gradients_sum+15+(i*16), d_m1db, n*4);
        cudaMemcpy(gradients_sumc+15+i*16, gradients_sum+15+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[15+(i*16)] = gradients_add(d_m1db_vali, 1, 2048);
        // printf("%.15f %.15f\n", gradients_sumc[15+i*16], gradients_sum_vali[15+i*16]);

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, n*4, 
                    &alpha,
                    d_m1w, n*4,
                    dx_42, n*4,
                    &beta,
                    dx1, n);

        // cudaMemcpy(dx1c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *m1_dx = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n*4, 1, gelu_grad_vali, n*4, m1_w, n*4, 0, m1_dx, n); 
       
        // for(int j=0;j<n*m;j++){
        //     printf("%.25f %.25f loc:%d \n", dx1c[j], m1_dx[j], j);
        //     if(fabs(dx1c[j] - m1_dx[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", dx1c[j] , m1_dx[j], j);
        //         break;
        //     }    
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n*4, n, m, 
                    &alpha,
                    dx_42, n*4,
                    norm_m, n,
                    &beta,
                    d_m1dw, n*4);

        // float *d_m1dwc = (float*)malloc(sizeof(float)*n*n*4);
        // cudaMemcpy(d_m1dwc, d_m1dw, sizeof(float)*n*n*4, cudaMemcpyDeviceToHost);


        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n*4, m, 1, norm_m_vali, n, gelu_grad_vali, n*4, 0, d_m1dw_vali, n*4);
            
        // for(int j=0;j<n*n*4;j++){
        //     printf("%.25f %.25f loc:%d \n", d_m1dwc[j], d_m1dw_vali[j], j);
        //     if(fabs(d_m1dwc[j] - d_m1dw_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", d_m1dwc[j] , d_m1dw_vali[j], j);
        //         break;
        //     }    
        // }

        // return 0;

        gradients_add_gpu_512_2048<<<2048, 512>>>(gradients_sum+14+i*16, d_m1dw, n*n*4);
        cudaMemcpy(gradients_sumc+14+i*16, gradients_sum+14+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[14+i*16] = gradients_add(d_m1dw_vali, n, n*4);
        // printf("%.15f %.15f\n", gradients_sumc[14+i*16], gradients_sum_vali[14+i*16]);

        // return 0;

        layernorm_dg_db_gpu<<<seq_len, n>>>(d_norm_mdg, d_norm_mdb, dx1, add_1, norm_m_mean, norm_m_rstd, batch_size, seq_len, state);

        // float *d_norm_mdgc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_norm_mdgc, d_norm_mdg, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_norm_mdbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_norm_mdbc, d_norm_mdb, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_norm_mdg_vali = (float*)malloc(sizeof(float)*n);
        // float *d_norm_mdb_vali = (float*)malloc(sizeof(float)*n);

        // layernorm_dg_db(d_norm_mdg_vali, d_norm_mdb_vali, m1_dx, add_1_vali, norm_m_mean_vali, norm_m_rstd_vali, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_norm_mdgc[j], d_norm_mdg_vali[j], j);
        //     if(fabs(d_norm_mdgc[j] - d_norm_mdg_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", d_norm_mdgc[j] , d_norm_mdg_vali[j], j);
        //         break;
        //     }    
        // }

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_norm_mdbc[j], d_norm_mdb_vali[j], j);
        //     if(fabs(d_norm_mdbc[j] - d_norm_mdb_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", d_norm_mdbc[j] , d_norm_mdb_vali[j], j);
        //         break;
        //     }    
        // }

        // return 0;

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+13+i*16, d_norm_mdg, n);
        cudaMemcpy(gradients_sumc+13+i*16, gradients_sum+13+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[13+i*16] = gradients_add(d_norm_mdg_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[13+i*16], gradients_sum_vali[13+i*16]);

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+12+i*16, d_norm_mdb, n);
        cudaMemcpy(gradients_sumc+12+i*16, gradients_sum+12+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[12+i*16] = gradients_add(d_norm_mdb_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[12+i*16], gradients_sum_vali[12+i*16]);

        // return 0;

        layernorm_grad_dx_gpu<<<m, n>>>(dx2, add_1, norm_m_mean, norm_m_rstd, dx1, d_norm_ag, batch_size, seq_len, state);
        // cudaMemcpy(dx2c, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *norm_m_dx = (float*)malloc(sizeof(float)*m*n);
        // layernorm_grad_dx(norm_m_dx, add_1_vali, norm_m_mean_vali, norm_m_rstd_vali, m1_dx, norm_m_g, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx2c[j], norm_m_dx[j], j);
        //     if(fabs(dx2c[j] - norm_m_dx[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", dx2c[j] , norm_m_dx[j], j);
        //         break;
        //     }
        // }
        // return 0 ;

        tensor_add_tensor_gpu<<<m, n>>>(grad_add1, logits_dx+(m*n), dx2, batch_size, seq_len, state);

        // float *grad_add1c = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(grad_add1c, grad_add1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *tempA = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(tempA, logits_dx+m*n, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *tempB = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(tempB, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *grad_add1_vali = (float*)malloc(sizeof(float)*m*n);
        // tensor_add_tensor(grad_add1_vali, logits_dx_vali, norm_m_dx, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f A: %f, %f B: %f,%f loc:%d \n", grad_add1c[j], grad_add1_vali[j], tempA[j], logits_dx_vali[j], tempB[j], norm_m_dx[j], j);
        //     if(fabs(grad_add1c[j] - grad_add1_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f A: %.15f, %.15f B: %.15f, %.15f loc:%d \n", grad_add1c[j], grad_add1_vali[j], tempA[j], logits_dx_vali[j], tempB[j], norm_m_dx[j], j);
        //         // break;
        //     }
        // }

        // return 0 ;


        ApplydropoutMask<<<320, 512>>>(dx2, grad_add1, d_amask, probe, seq_len*state);
        
        // cudaMemcpy(dx2c, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *a_after_dropout_grad = (float*)malloc(sizeof(float)*m*n);
        // dropout1(a_after_dropout_grad, grad_add1_vali, a_mask, probe, seq_len*state, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx2c[j], a_after_dropout_grad[j], j);
        //     if(fabs(dx2c[j] - a_after_dropout_grad[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", dx2c[j] , a_after_dropout_grad[j], j);
        //         break;
        //     }
        // }
        // return 0 ;
        
        bias_grad_db_gpu<<<320, n>>>(d_adb, dx2, batch_size, seq_len, state);   
    
        // float *d_adbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_adbc, d_adb, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_adb_vali = (float*)malloc(sizeof(float)*n);
        // bias_grad_db(d_adb_vali, a_after_dropout_grad, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     // printf("%.25f %.25f A: %.25f, %.25f loc:%d \n", d_adbc[j], d_adb_vali[j], dx2c[j], a_after_dropout_grad[j], j);
        //     if(fabs(d_adbc[j] - d_adb_vali[j]) > 0.000001){
        //         printf("%.25f %.25f A: %.25f, %.25f loc:%d \n", d_adbc[j], d_adb_vali[j], dx2c[j], a_after_dropout_grad[j], j);
        //         // break;
        //     }
        // }

        // return 0 ;    

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+11+i*16, d_adb, n);
        cudaMemcpy(gradients_sumc+11+i*16, gradients_sum+11+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[11+i*16] = gradients_add(d_adb_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[11+i*16], gradients_sum_vali[11+i*16]);

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, n, 
                    &alpha,
                    d_aw, n,
                    dx2, n,
                    &beta,
                    dx1, n);

        // cudaMemcpy(dx1c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *a_dx = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, a_after_dropout_grad, n, a_w, n, 0, a_dx, n);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx1c[j], a_dx[j], j);
        //     if(fabs(dx1c[j] - a_dx[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", dx1c[j] , a_dx[j], j);
        //         break;
        //     }
        // }

        // return 0 ;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, m, 
                    &alpha,
                    dx2, n,
                    sv_out_reshape, n,
                    &beta,
                    d_adw, n);

        // float *d_adwc = (float*)malloc(sizeof(float)*n*n);
        // cudaMemcpy(d_adwc, d_adw, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

        // float *d_adw_vali = (float*)malloc(sizeof(float)*n*n);
        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, sv_reshape_vali, n, a_after_dropout_grad, n, 0, d_adw_vali, n);

        // for(int j=0;j<n*n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_adwc[j], d_adw_vali[j], j);
        //     if(fabs(d_adwc[j] - d_adw_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_adwc[j] , d_adw_vali[j], j);
        //         break;
        //     }       
        // }

        // return 0 ;

        gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+10+i*16, d_adw, n*n);
        cudaMemcpy(gradients_sumc+10+i*16, gradients_sum+10+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[10+i*16] = gradients_add(d_adw_vali, n, n);
        // printf("%.15f %.15f\n", gradients_sumc[10+i*16], gradients_sum_vali[10+i*16]);

        // return 0;

        transpose_0123to0213_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(dx1, dx2, batch_size, seq_len, head_num, size_per_head);
        
        // cudaMemcpy(dx2c, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *reshape_a_dx = (float*)malloc(sizeof(float)*m*n);
        // transpose_0123to0213(a_dx, reshape_a_dx, batch_size, seq_len, head_num, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx2c[j], reshape_a_dx[j], j);
        //     if(fabs(dx2c[j] - reshape_a_dx[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", dx2c[j] , reshape_a_dx[j], j);
        //         break;
        //     }       
        // }

        // return 0 ;
        

        offset = 0;
        offset1 = 0;
        for(int j=0; j<batch_size*head_num;j++){
            
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        seq_len, seq_len, size_per_head,
                        &alpha,
                        v_reshape+offset, size_per_head, 
                        dx2+offset, size_per_head,
                        &beta,
                        sv_grad+offset1, seq_len);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        size_per_head, seq_len, seq_len,
                        &alpha,
                        dx2+offset, size_per_head, 
                        softmax_out+offset1, seq_len,
                        &beta,
                        v_grad+offset, size_per_head);

            offset += seq_len*size_per_head;
            offset1 += seq_len*seq_len;

        }

        // float *sv_gradc = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // cudaMemcpy(sv_gradc, sv_grad, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);

        // float *sv_grad_vali = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // multiheadmatmulNT<float, float, float>(sv_grad_vali, reshape_a_dx, v_reshape_vali, batch_size, head_num, seq_len, size_per_head, seq_len);

        // for(int j=0;j<(batch_size*head_num*seq_len*seq_len);j++){
        //     printf("%.25f %.25f loc:%d \n", sv_gradc[j], sv_grad_vali[j], j);
        //     if(fabs(sv_gradc[j] - sv_grad_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", sv_gradc[j] , sv_grad_vali[j], j);
        //         break;
        //     }       
        // }

        // float *v_gradc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(v_gradc, v_grad, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *v_grad_vali = (float*)malloc(sizeof(float)*m*n);
        // multiheadmatmulTN<float, float, float>(v_grad_vali, softmax_out_vali, reshape_a_dx, batch_size, head_num, seq_len, seq_len, size_per_head);
        
        // for(int j=0;j<(m*n);j++){
        //     printf("%.25f %.25f loc:%d \n", v_gradc[j], v_grad_vali[j], j);
        //     if(fabs(v_gradc[j] - v_grad_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", v_gradc[j] , v_grad_vali[j], j);
        //         break;
        //     }       
        // }

        // return 0 ;

        softmax_grad_gpu<<<(batch_size*head_num), seq_len>>>(softmaxgrad, sv_grad, softmax_out, scaler, batch_size, head_num, seq_len);

        // float *softmaxgradc = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // cudaMemcpy(softmaxgradc, softmaxgrad, sizeof(float)*batch_size*head_num*seq_len*seq_len, cudaMemcpyDeviceToHost);

        // float *softmaxgrad_vali = (float*)malloc(sizeof(float)*batch_size*head_num*seq_len*seq_len);
        // softmax_grad<float, float, float>(softmaxgrad_vali, sv_grad_vali, softmax_out_vali, scaler, batch_size, head_num, seq_len);

        // for(int j=0;j<batch_size*head_num*seq_len*seq_len;j++){
        //     printf("%.25f %.25f loc:%d \n", softmaxgradc[j], softmaxgrad_vali[j], j);
        //     if(fabs(softmaxgradc[j] - softmaxgrad_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", softmaxgradc[j] , softmaxgrad_vali[j], j);
        //         break;
        //     }
        // }
        // return 0;

        offset = 0;
        offset1 = 0;
        for(int j=0;j<batch_size*head_num;j++){

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        size_per_head, seq_len, seq_len,
                        &alpha,
                        k_reshape+offset, size_per_head, 
                        softmaxgrad+offset1, seq_len,
                        &beta,
                        k_grad+offset, size_per_head);

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        seq_len, size_per_head, seq_len,
                        &alpha,
                        softmaxgrad+offset1, seq_len, 
                        q_reshape+offset, size_per_head,
                        &beta,
                        q_grad+offset, seq_len);
            
            offset += seq_len*size_per_head;
            offset1 += seq_len*seq_len;

        }

        // float *k_gradc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(k_gradc, k_grad, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *k_grad_vali = (float*)malloc(sizeof(float)*m*n);
        // multiheadmatmulNN(k_grad_vali, softmaxgrad_vali, k_reshape_vali, batch_size, head_num, seq_len, seq_len, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", k_gradc[j], k_grad_vali[j], j);
        //     if(fabs(k_gradc[j] - k_grad_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", k_gradc[j], k_grad_vali[j], j);
        //         break;
        //     }
        // }
        // return 0;

        // float *q_gradc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(q_gradc, q_grad, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *q_grad_vali = (float*)malloc(sizeof(float)*m*n);
        // multiheadmatmulTN(q_grad_vali, q_reshape_vali, softmaxgrad_vali, batch_size, head_num, seq_len, size_per_head, seq_len);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", q_gradc[j], q_grad_vali[j], j);
        //     if(fabs(q_gradc[j] - q_grad_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", q_gradc[j], q_grad_vali[j], j);
        //         break;
        //     }
        // }
        // return 0;


        transpose_0123to0132_gpu<<<(batch_size*head_num*seq_len), size_per_head>>>(q_grad, q_grad_trans, batch_size, seq_len, head_num, size_per_head);
        
        // float *q_grad_transc = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(q_grad_transc, q_grad_trans, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *q_grad_trans_vali = (float*)malloc(sizeof(float)*m*n);
        // transpose_0123to0132(q_grad_vali, q_grad_trans_vali, batch_size, head_num, size_per_head, seq_len);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", q_grad_transc[j], q_grad_trans_vali[j], j);
        //     if(fabs(q_grad_transc[j] - q_grad_trans_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", q_grad_transc[j] , q_grad_trans_vali[j], j);
        //         break;
        //     } 
        // }
        // return 0;


        transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(q_grad_trans, q_grad_reshape, batch_size, seq_len, head_num, size_per_head);
        transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(k_grad, k_grad_reshape, batch_size, seq_len, head_num, size_per_head);
        transpose_0213to0123_gpu<<<(batch_size*seq_len*head_num), size_per_head>>>(v_grad, v_grad_reshape, batch_size, seq_len, head_num, size_per_head);

        // float *q_grad_reshapec = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(q_grad_reshapec, q_grad_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *k_grad_reshapec = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(k_grad_reshapec, k_grad_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *v_grad_reshapec = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(v_grad_reshapec, v_grad_reshape, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *q_grad_reshape_vali = (float*)malloc(sizeof(float)*m*n);
        // transpose_0123to0213(q_grad_trans_vali, q_grad_reshape_vali, batch_size, head_num, seq_len, size_per_head);

        // float *k_grad_reshape_vali = (float*)malloc(sizeof(float)*m*n);
        // transpose_0123to0213(k_grad_vali, k_grad_reshape_vali, batch_size, head_num, seq_len, size_per_head);

        // float *v_grad_reshape_vali = (float*)malloc(sizeof(float)*m*n);
        // transpose_0123to0213(v_grad_vali, v_grad_reshape_vali, batch_size, head_num, seq_len, size_per_head);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", q_grad_reshapec[j], q_grad_reshape_vali[j], j);
        //     if(fabs(q_grad_reshapec[j] - q_grad_reshape_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", q_grad_reshapec[j] , q_grad_reshape_vali[j], j);
        //         break;
        //     }
        // }

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", k_grad_reshapec[j], k_grad_reshape_vali[j], j);
        //     if(fabs(k_grad_reshapec[j] - k_grad_reshape_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", k_grad_reshapec[j] , k_grad_reshape_vali[j], j);
        //         break;
        //     }
        // }

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", v_grad_reshapec[j], v_grad_reshape_vali[j], j);
        //     if(fabs(v_grad_reshapec[j] - v_grad_reshape_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", v_grad_reshapec[j] , v_grad_reshape_vali[j], j);
        //         break;
        //     }
        // }

        // return 0;

        bias_grad_db_gpu<<<320, n>>>(d_qdb, q_grad_reshape, batch_size, seq_len, state);
        bias_grad_db_gpu<<<320, n>>>(d_kdb, k_grad_reshape, batch_size, seq_len, state);
        bias_grad_db_gpu<<<320, n>>>(d_vdb, v_grad_reshape, batch_size, seq_len, state);

        // float *d_qdbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_qdbc, d_qdb, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_qdb_vali = (float*)malloc(sizeof(float)*n);
        // bias_grad_db(d_qdb_vali, q_grad_reshape_vali, batch_size, seq_len, state);

        // float *d_kdbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_kdbc, d_kdb, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_kdb_vali = (float*)malloc(sizeof(float)*n);
        // bias_grad_db(d_kdb_vali, k_grad_reshape_vali, batch_size, seq_len, state);

        // float *d_vdbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_vdbc, d_vdb, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_vdb_vali = (float*)malloc(sizeof(float)*n);
        // bias_grad_db(d_vdb_vali, v_grad_reshape_vali, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_qdbc[j], d_qdb_vali[j], j);
        //     if(fabs(d_qdbc[j] - d_qdb_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_qdbc[j] , d_qdb_vali[j], j);
        //         break;
        //     }
        // }       

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_kdbc[j], d_kdb_vali[j], j);
        //     if(fabs(d_kdbc[j] - d_kdb_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_kdbc[j] , d_kdb_vali[j], j);
        //         break;
        //     }
        // }

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_vdbc[j], d_vdb_vali[j], j);
        //     if(fabs(d_vdbc[j] - d_vdb_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_vdbc[j] , d_vdb_vali[j], j);
        //         break;
        //     }
        // }

        // return 0;

        gradients_add_gpu_512<<<1,512>>>(gradients_sum+9+i*16, d_qdb, n);
        cudaMemcpy(gradients_sumc+9+i*16, gradients_sum+9+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[9+i*16] = gradients_add(d_qdb_vali, 1, n);
        // printf("%.25f %.25f\n", gradients_sumc[9+i*16], gradients_sum_vali[9+i*16]);

        gradients_add_gpu_512<<<1,512>>>(gradients_sum+8+i*16, d_kdb, n);
        cudaMemcpy(gradients_sumc+8+i*16, gradients_sum+8+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[8+i*16] = gradients_add(d_kdb_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[8+i*16], gradients_sum_vali[8+i*16]);

        gradients_add_gpu_512<<<1,512>>>(gradients_sum+7+i*16, d_vdb, n);
        cudaMemcpy(gradients_sumc+7+i*16, gradients_sum+7+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[7+i*16] = gradients_add(d_vdb_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[7+i*16], gradients_sum_vali[7+i*16]);

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, n, 
                    &alpha,
                    d_vw, n,
                    v_grad_reshape, n,
                    &beta,
                    dx1, n);

        // cudaMemcpy(dx1c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *v_dx = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, v_grad_reshape_vali, n, v_w, n, 0, v_dx, n);

        // for(int j=0;j<m*n;j++){
        //     // printf("%.25f %.25f loc:%d \n", dx1c[j], v_dx[j], j);
        //     if(fabs(dx1c[j] - v_dx[j]) > 0.0000000001){
        //         printf("%.25f %.25f loc:%d\n", dx1c[j] , v_dx[j], j);
        //         break;
        //     }
        // }

        // return 0;


        

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, m, 
                    &alpha,
                    v_grad_reshape, n,
                    norm_a, n,
                    &beta,
                    d_vdw, n);  

        // float *d_vdwc = (float*)malloc(sizeof(float)*n*n);
        // cudaMemcpy(d_vdwc, d_vdw, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

        // float *d_vdw_vali = (float*)malloc(sizeof(float)*n*n);
        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a_vali, n, v_grad_reshape_vali, n, 0, d_vdw_vali, n);

        // for(int j=0;j<n*n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_vdwc[j], d_vdw_vali[j], j);
        //     if(fabs(d_vdwc[j] - d_vdw_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_vdwc[j] , d_vdw_vali[j], j);
        //         break;
        //     } 
        // }

        // return 0;

        gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+6+i*16, d_vdw, n*n);
        cudaMemcpy(gradients_sumc+6+i*16, gradients_sum+6+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[6+i*16] = gradients_add(d_vdw_vali, n, n);
        // printf("%.15f %.15f\n", gradients_sumc[6+i*16], gradients_sum_vali[6+i*16]);

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, n, 
                    &alpha,
                    d_kw, n,
                    k_grad_reshape, n,
                    &beta,
                    dx2, n);

        // cudaMemcpy(dx2c, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *k_dx = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, k_grad_reshape_vali, n, k_w, n, 0, k_dx, n);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx2c[j], k_dx[j], j);
        //     if(fabs(dx2c[j] - k_dx[j]) > 0.0000000001){
        //         printf("%.25f %.25f loc:%d\n", dx2c[j] , k_dx[j], j);
        //         break;
        //     }
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, m, 
                    &alpha,
                    k_grad_reshape, n,
                    norm_a, n,
                    &beta,
                    d_kdw, n);  

        // float *d_kdwc = (float*)malloc(sizeof(float)*n*n);
        // cudaMemcpy(d_kdwc, d_kdw, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

        // float *d_kdw_vali = (float*)malloc(sizeof(float)*n*n);
        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a_vali, n, k_grad_reshape_vali, n, 0, d_kdw_vali, n);

        // for(int j=0;j<n*n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_kdwc[j], d_kdw_vali[j], j);
        //     if(fabs(d_kdwc[j] - d_kdw_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_kdwc[j] , d_kdw_vali[j], j);
        //         break;
        //     } 
        // }

        // return 0;

        gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+5+i*16, d_kdw, n*n);
        cudaMemcpy(gradients_sumc+5+i*16, gradients_sum+5+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[5+i*16] = gradients_add(d_kdw_vali, n, n);
        // printf("%.15f %.15f\n", gradients_sumc[5+i*16], gradients_sum_vali[5+i*16]);

        // return 0; 

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, n, 
                    &alpha,
                    d_qw, n,
                    q_grad_reshape, n,
                    &beta,
                    dx3, n);

        // cudaMemcpy(dx3c, dx3, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *q_dx = (float*)malloc(sizeof(float)*m*n);
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, q_grad_reshape_vali, n, q_w, n, 0, q_dx, n);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx3c[j], q_dx[j], j);
        //     if(fabs(dx3c[j] - q_dx[j]) > 0.0000000001){
        //         printf("%.25f %.25f loc:%d\n", dx3c[j] , q_dx[j], j);
        //         break;
        //     }
        // }

        // return 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, m, 
                    &alpha,
                    q_grad_reshape, n,
                    norm_a, n,
                    &beta,
                    d_qdw, n);   

        // float *d_qdwc = (float*)malloc(sizeof(float)*n*n);
        // cudaMemcpy(d_qdwc, d_qdw, sizeof(float)*n*n, cudaMemcpyDeviceToHost);


        // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a_vali, n, q_grad_reshape_vali, n, 0, d_qdw_vali, n);

        // for(int j=0;j<n*n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_qdwc[j], d_qdw_vali[j], j);
        //     if(fabs(d_qdwc[j] - d_qdw_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", d_qdwc[j] , d_qdw_vali[j], j);
        //         break;
        //     } 
        // }

        // return 0;

        gradients_add_gpu_512_512<<<512,512>>>(gradients_sum+4+i*16, d_qdw, n*n);
        cudaMemcpy(gradients_sumc+4+i*16, gradients_sum+4+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[4+i*16] = gradients_add(d_qdw_vali, n, n);
        // printf("%.15f %.15f\n", gradients_sumc[4+i*16], gradients_sum_vali[4+i*16]);

        // return 0;  

        tensor_add_tensor_gpu<<<m, n>>>(v_grad, dx1, dx2, batch_size, seq_len, state);
        tensor_add_tensor_gpu<<<m, n>>>(dx1, v_grad, dx3, batch_size, seq_len, state);

        // float *grad_add2c = (float*)malloc(sizeof(float)*m*n);
        // cudaMemcpy(grad_add2c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *grad_add2_vali = (float*)malloc(sizeof(float)*m*n);
        // float *grad_add2_mid = (float*)malloc(sizeof(float)*m*n);
        // tensor_add_tensor(grad_add2_mid, q_dx, k_dx, batch_size, seq_len, state);
        // tensor_add_tensor(grad_add2_vali, grad_add2_mid, v_dx, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", grad_add2c[j], grad_add2_vali[j], j);
        //     if(fabs(grad_add2c[j] - grad_add2_vali[j]) > 0.0000001){
        //         printf("%.25f %.25f loc:%d\n", grad_add2c[j] , grad_add2_vali[j], j);
        //         break;
        //     } 
        // }

        // return 0;


        layernorm_dg_db_gpu<<<seq_len, n>>>(d_norm_adg, d_norm_adb, dx1, embed_add, norm_a_mean, norm_a_rstd, batch_size, seq_len, state);
        
        // float *d_norm_adgc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_norm_adgc, d_norm_adg, sizeof(float)*n, cudaMemcpyDeviceToHost);

        // float *d_norm_adbc = (float*)malloc(sizeof(float)*n);
        // cudaMemcpy(d_norm_adbc, d_norm_adb, sizeof(float)*n, cudaMemcpyDeviceToHost);

       
        // layernorm_dg_db(d_norm_adg_vali, d_norm_adb_vali, grad_add2_vali, embed_add_vali, norm_a_mean_vali, norm_a_rstd_vali, batch_size, seq_len, state);

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_norm_adgc[j], d_norm_adg_vali[j], j);
        //     if(fabs(d_norm_adgc[j] - d_norm_adg_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", d_norm_adgc[j] , d_norm_adg_vali[j], j);
        //         break;
        //     }    
        // }

        // for(int j=0;j<n;j++){
        //     printf("%.25f %.25f loc:%d \n", d_norm_adbc[j], d_norm_adb_vali[j], j);
        //     if(fabs(d_norm_adbc[j] - d_norm_adb_vali[j]) > 0.000001){
        //         printf("%.25f %.25f loc:%d\n", d_norm_adbc[j] , d_norm_adb_vali[j], j);
        //         break;
        //     }    
        // }

        // return 0;

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+3+i*16, d_norm_adg, n);
        cudaMemcpy(gradients_sumc+3+i*16, gradients_sum+3+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[3+i*16] = gradients_add(d_norm_adg_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[3+i*16], gradients_sum_vali[3+i*16]);

        gradients_add_gpu_512<<<1, 512>>>(gradients_sum+2+i*16, d_norm_adb, n);
        cudaMemcpy(gradients_sumc+2+i*16, gradients_sum+2+i*16, sizeof(float), cudaMemcpyDeviceToHost);

        // gradients_sum_vali[2+i*16] = gradients_add(d_norm_adb_vali, 1, n);
        // printf("%.15f %.15f\n", gradients_sumc[2+i*16], gradients_sum_vali[2+i*16]);

        // return 0;

        
        layernorm_grad_dx_gpu<<<m, n>>>(dx2, embed_add, norm_a_mean, norm_a_rstd, dx1, d_norm_ag, batch_size, seq_len, state);
        // cudaMemcpy(dx2c, dx2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

        // float *norm_a_dx = (float*)malloc(sizeof(float)*m*n);
        // layernorm_grad_dx(norm_a_dx, embed_add_vali, norm_a_mean_vali, norm_a_rstd_vali, grad_add2_vali, norm_m_g, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", dx2c[j], norm_a_dx[j], j);
        //     if(fabs(dx2c[j] - norm_a_dx[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", dx2c[j] , norm_a_dx[j], j);
        //         break;
        //     }
        // }
        // return 0 ;

        tensor_add_tensor_gpu<<<m, n>>>(logits_dx, grad_add1, dx2, batch_size, seq_len, state);


        // cudaMemcpy(logits_dxc2, logits_dx, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        // tensor_add_tensor(logits_dx2_vali, grad_add1_vali, norm_a_dx, batch_size, seq_len, state);

        // for(int j=0;j<m*n;j++){
        //     printf("%.25f %.25f loc:%d \n", logits_dxc2[j], logits_dx2_vali[j], j);
        //     if(fabs(logits_dxc2[j] - logits_dx2_vali[j]) > 0.00000001){
        //         printf("%.25f %.25f loc:%d\n", logits_dxc2[j] , logits_dx2_vali[j], j);
        //         break;
        //     }
        // }
        // return 0 ;
    }

    ApplydropoutMask<<<320, 512>>>(dx1, logits_dx, d_xmask, probe, seq_len*state);
    // cudaMemcpy(dx1c, dx1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    
    // float *x_dropout_grad = (float*)malloc(sizeof(float)*m*n);
    // dropout1(x_dropout_grad, logits_dx2_vali, x_mask, probe, seq_len*state, batch_size, seq_len, state);

    // for(int j=0;j<m*n;j++){
    //     printf("%.25f %.25f loc:%d \n", dx1c[j], x_dropout_grad[j], j);
    //     if(fabs(dx1c[j] - x_dropout_grad[j]) > 0.00000001){
    //         printf("%.25f %.25f loc:%d\n", dx1c[j] , x_dropout_grad[j], j);
    //         break;
    //     }
    // }
    // return 0;

    float *p_embed_grad, *p_embed_dw, *embed_grad, *logits_dw_trans, *embed_add_out;
    cudaMalloc((void**)&p_embed_grad, sizeof(float)*seq_len*state);
    cudaMalloc((void**)&p_embed_dw, sizeof(float)*seq_len*state);
    cudaMalloc((void**)&embed_grad, sizeof(float)*vocab_size*state);
    cudaMalloc((void**)&logits_dw_trans, sizeof(float)*vocab_size*state);
    cudaMalloc((void**)&embed_add_out, sizeof(float)*vocab_size*state);

    add_grad_gpu<<<seq_len,n>>>(p_embed_grad, dx1, batch_size, seq_len, state);
    // float *p_embed_gradc = (float*)malloc(sizeof(float)*seq_len*state);
    // cudaMemcpy(p_embed_gradc, p_embed_grad, sizeof(float)*seq_len*state, cudaMemcpyDeviceToHost);

    // float *p_embed_grad_vali = (float*)malloc(sizeof(float)*seq_len*state);
    // add_grad(p_embed_grad_vali, x_dropout_grad, batch_size, seq_len, state);

    // for(int j=0;j<seq_len*n;j++){
    //     printf("%.25f %.25f loc:%d \n", p_embed_gradc[j], p_embed_grad_vali[j], j);
    //     if(fabs(p_embed_gradc[j] - p_embed_grad_vali[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", p_embed_gradc[j] , p_embed_grad_vali[j], j);
    //         break;
    //     }    
    // }

    // return 0;

    ApplydropoutMask_SmallSize<<<10,512>>>(p_embed_dw, p_embed_grad, d_pmask, probe, 5120);

    // float *p_embed_dwc = (float*)malloc(sizeof(float)*seq_len*state);
    // cudaMemcpy(p_embed_dwc, p_embed_dw, sizeof(float)*seq_len*state,cudaMemcpyDeviceToHost);

    // float *p_embed_dw_vali = (float*)malloc(sizeof(float)*seq_len*state);
    // dropout2(p_embed_dw_vali, p_embed_grad_vali, p_mask, probe, 5120, n);

    // for(int j=0;j<seq_len*n;j++){
    //     printf("%.25f %.25f loc:%d \n", p_embed_dwc[j], p_embed_dw_vali[j], j);
    //     if(fabs(p_embed_dwc[j] - p_embed_dw_vali[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", p_embed_dwc[j] , p_embed_dw_vali[j], j);
    //         break;
    //     }    
    // }

    // return 0;

    gradients_add_gpu_512_512<<<320, 512>>>(gradients_sum+1, p_embed_dw, seq_len*state);
    cudaMemcpy(gradients_sumc+1, gradients_sum+1, sizeof(float), cudaMemcpyDeviceToHost);

    // gradients_sum_vali[1] = gradients_add(p_embed_dw_vali, seq_len, state);
    // printf("%.15f %.15f\n", gradients_sumc[1], gradients_sum_vali[1]);

    // return 0;

    embedding_lookup_grad_gpu<<<m, n>>>(embed_grad, dx1, d_xlabel, batch_size, seq_len, state, vocab_size);
    
    // float *embed_gradc = (float*)malloc(sizeof(float)*vocab_size*state);
    // cudaMemcpy(embed_gradc, embed_grad, sizeof(float)*vocab_size*state, cudaMemcpyDeviceToHost);

    // float *embed_grad_vali = (float*)malloc(sizeof(float)*vocab_size*state);
    // embedding_lookup_grad(embed_grad_vali, x_dropout_grad, xs_int, batch_size, seq_len, vocab_size, state);

    // for(int j=0;j<n;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", embed_gradc[j], embed_grad_vali[j], j);
    //     if(fabs(embed_gradc[j] - embed_grad_vali[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", embed_gradc[j] , embed_grad_vali[j], j);
    //         break;
    //     } 
    // }

    // return 0;


    transpose_0123to0132_gpu<<<vocab_size, n>>>(logits_dw, logits_dw_trans, 1, vocab_size, 1, state);

    // float* logits_dw_transc = (float*)malloc(sizeof(float)*vocab_size*state);
    // cudaMemcpy(logits_dw_transc, logits_dw_trans, sizeof(float)*vocab_size*state, cudaMemcpyDeviceToHost);

    // float* logits_dw_trans_vali = (float*)malloc(sizeof(float)*vocab_size*state);
    // transpose_01to10(logits_dw_vali, logits_dw_trans_vali, n, vocab_size);

    // for(int j=0;j<vocab_size*state;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", logits_dw_transc[j], logits_dw_trans_vali[j], j);
    //     if(fabs(logits_dw_transc[j] - logits_dw_trans_vali[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", logits_dw_transc[j] , logits_dw_trans_vali[j], j);
    //         break;
    //     }
    // }
    // return 0;
    
    tensor_add_tensor_gpu<<<vocab_size, n>>>(embed_add_out, logits_dw_trans, embed_grad, 1, vocab_size, state);
    
    // float *embed_add_outc = (float*)malloc(sizeof(float)*vocab_size*state);
    // cudaMemcpy(embed_add_outc, embed_add_out, sizeof(float)*vocab_size*state, cudaMemcpyDeviceToHost);

    // float *embed_add_out_vali = (float*)malloc(sizeof(float)*vocab_size*state);
    // tensor_add_tensor(embed_add_out_vali, logits_dw_trans_vali, embed_grad_vali, 1, vocab_size, state);

    // for(int j=0;j<vocab_size*state;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", embed_add_outc[j], embed_add_out_vali[j], j);
    //     if(fabs(embed_add_outc[j] - embed_add_out_vali[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", embed_add_outc[j] , embed_add_out_vali[j], j);
    //         break;
    //     }
    // }

    // return 0;

    gradients_add_gpu_512_512<<<256, 512>>>(gradients_sum, embed_add_out, vocab_size*state);
    cudaMemcpy(gradients_sumc, gradients_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // gradients_sum_vali[0] = gradients_add(embed_add_out_vali, vocab_size, state);
    // printf("%.15f %.15f\n", gradients_sumc[0], gradients_sum_vali[0]);


    for(int i=0;i<(num_layers*16+2);i++){
        // printf("%.25f %d\n", gradients_sumc[i], i);
        grad_sum_sum += gradients_sumc[i];
    }


    global_norm = sqrt(grad_sum_sum);
    norm_scale = clip_by_global_norm(global_norm, clip_norm);

    printf("global_norm:%.10f\n", global_norm);
    printf("norm_sacle:%.10f\n", norm_scale);

    // return 0;

    lr = global_step * (1.0/1000) < 1 ? global_step * (1.0/1000) : 1;
    lr *= learning_rate;

    beta1_power = adam_got_beta_power(beta1, global_step);
    beta2_power = adam_got_beta_power(beta2, global_step);

    float lr_2 = adam_got_lr(lr, beta1_power, beta2_power);

    adam_apply_gradients_gpu<<<1, n>>>(d_norm_ag, d_norm_adg, d_norm_a_g_mt, d_norm_a_g_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2, 
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);

    // float *d_norm_agc = (float*)malloc(sizeof(float)*n);
    // cudaMemcpy(d_norm_agc, d_norm_ag, sizeof(float)*n, cudaMemcpyDeviceToHost);
                                    
    // adam_apply_gradients(norm_a_g, d_norm_adg_vali, norm_a_g_mt, norm_a_g_vt, 
    //                     beta1, beta2, beta1_power, beta2_power, lr_2, 
    //                     epsilon, norm_scale, grad_scale, clip_sigma, n*num_layers);

    // for(int j=0;j<state;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", d_norm_agc[j], norm_a_g[j], j);
    //     if(fabs(d_norm_agc[j] - norm_a_g[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", d_norm_agc[j] , norm_a_g[j], j);
    //         break;
    //     }
    // }

    // return 0;

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
    //check this
    adam_apply_gradients_gpu<<<4, n>>>(d_m1b, d_m1db, d_m1_b_mt, d_m1_b_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*4*num_layers);

    // float *d_m1bc = (float*)malloc(sizeof(float)*n*4);
    // cudaMemcpy(d_m1bc, d_m1b, sizeof(float)*n*4, cudaMemcpyDeviceToHost);
                                    
    // adam_apply_gradients(m1_b, d_m1db_vali, m1_b_mt, m1_b_vt, 
    //                     beta1, beta2, beta1_power, beta2_power, lr_2, 
    //                     epsilon, norm_scale, grad_scale, clip_sigma, n*4*num_layers);

    // for(int j=0;j<state*4;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", d_m1bc[j], m1_b[j], j);
    //     if(fabs(d_m1bc[j] - m1_b[j]) > 0.000000001){
    //         printf("%.25f %.25f loc:%d\n", d_m1bc[j] , m1_b[j], j);
    //         break;
    //     }
    // }

    // return 0;

    adam_apply_gradients_gpu<<<1, n>>>(d_m2b, d_m2db, d_m2_b_mt, d_m2_b_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*num_layers);
    //check this
    adam_apply_gradients_gpu<<<n, n>>>(d_qw, d_qdw, d_q_w_mt, d_q_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);

    // float *d_qwc = (float*)malloc(sizeof(float)*n*n);
    // cudaMemcpy(d_qwc, d_qw, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
                                    
    // adam_apply_gradients(q_w, d_qdw_vali, q_w_mt, q_w_vt, 
    //                     beta1, beta2, beta1_power, beta2_power, lr_2, 
    //                     epsilon, norm_scale, grad_scale, clip_sigma, n*n*num_layers);

    // for(int j=0;j<state*state;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", d_qwc[j], q_w[j], j);
    //     if(fabs(d_qwc[j] - q_w[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", d_qwc[j] , q_w[j], j);
    //         break;
    //     }
    // }

    // return 0;

    adam_apply_gradients_gpu<<<n, n>>>(d_kw, d_kdw, d_k_w_mt, d_k_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);

    adam_apply_gradients_gpu<<<n, n>>>(d_vw, d_vdw, d_v_w_mt, d_v_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers); 

    adam_apply_gradients_gpu<<<n, n>>>(d_aw, d_adw, d_a_w_mt, d_a_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*num_layers);   
    //check this
    adam_apply_gradients_gpu<<<n*4, n>>>(d_m1w, d_m1dw, d_m1_w_mt, d_m1_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*4*num_layers);

    // float *d_m1wc = (float*)malloc(sizeof(float)*n*n*4);
    // cudaMemcpy(d_m1wc, d_m1w, sizeof(float)*n*n*4, cudaMemcpyDeviceToHost);
                                    
    // adam_apply_gradients(m1_w, d_m1dw_vali, m1_w_mt, m1_w_vt, 
    //                     beta1, beta2, beta1_power, beta2_power, lr_2, 
    //                     epsilon, norm_scale, grad_scale, clip_sigma, n*n*4*num_layers);

    // for(int j=0;j<state*state*4;j++)
    // {
    //     printf("%.25f %.25f loc:%d \n", d_m1wc[j], m1_w[j], j);
    //     if(fabs(d_m1wc[j] - m1_w[j]) > 0.0000001){
    //         printf("%.25f %.25f loc:%d\n", d_m1wc[j] , m1_w[j], j);
    //         break;
    //     }
    // }

    // return 0;

    

    adam_apply_gradients_gpu<<<n*4, n>>>(d_m2w, d_m2dw, d_m2_w_mt, d_m2_w_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, state*state*4*num_layers); 
    //check this
    adam_apply_gradients_gpu<<<seq_len, n>>>(d_pembed, p_embed_dw, d_p_embed_mt, d_p_embed_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, seq_len*state*num_layers); 
    //check this
    adam_apply_gradients_gpu<<<vocab_size, n>>>(d_xembed, embed_add_out, d_x_embed_mt, d_x_embed_vt, 
                                    beta1, beta2, beta1_power, beta2_power, lr_2,
                                    epsilon, norm_scale, grad_scale, clip_sigma, vocab_size*state*num_layers); 
}