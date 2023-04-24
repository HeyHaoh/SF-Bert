/*********
 * 
 * g++ transformer_inference_part.cpp -o transformer_inference_part -w -lm -lopenblas
 *  ./transformer_inference_part 1 12 128 12 64
 *  
 * *********/

#include <stdio.h>
#include <cmath>
#include "embedding.h"
#include "validate.h"
#include "common.h"
#include "layernorm.h"
#include "matmul.h"
#include "ewops.h"
#include "optimizer.h"
#include <cblas.h>
#include <time.h>
#include <string.h>

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
    float loss = 0;
    float global_norm = 0;
    float grad_scale = 1.0;
    float clip_sigma = 0;
    
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

    float *embed_lookup = (float*)malloc(sizeof(float) * m * n);

    unsigned int *x_mask = (unsigned int*)malloc(sizeof(unsigned int)* m * state);
    unsigned int *p_mask = (unsigned int*)malloc(sizeof(unsigned int)* seq_len * state);

    float* x_after_dropout = (float*)malloc(sizeof(float) * m * n);
    float* p_after_dropout = (float*)malloc(sizeof(float) * seq_len * n);

    float* embed_add = (float*)malloc(sizeof(float) * m * n * (num_layers + 1));

    float* norm_a = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* norm_a_mean = (float*)malloc(sizeof(float) * m * num_layers);
    float* norm_a_rstd = (float*)malloc(sizeof(float) * m * num_layers);

    float *q_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float *k_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float *v_out = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* q_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* k_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* v_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* q_reshape = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* k_reshape = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* v_reshape = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* qk_out = (float*)malloc(sizeof(float) * batch_size * head_num * seq_len * seq_len * num_layers);
    float* softmax_out = (float*)malloc(sizeof(float) * batch_size * head_num * seq_len * seq_len * num_layers);
    float* sv_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* sv_trans = (float*)malloc(sizeof(float) * m * n * num_layers);

    float *a_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float *a_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);

    unsigned int* a_mask = (unsigned int*)malloc(sizeof(unsigned int) * m * state * num_layers);
    for(int i=0;i<num_layers;i++){
        gendropoutmask(a_mask+(i*m*n), probe, seq_len*state);
    }

    float* a_after_dropout = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* add_1 = (float*)malloc(sizeof(float) * m * n * num_layers);
    
    float* norm_m = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* norm_m_mean = (float*)malloc(sizeof(float) * m * num_layers);
    float* norm_m_rstd = (float*)malloc(sizeof(float) * m * num_layers);
    
    float* m1_out = (float*)malloc(sizeof(float) * m * n * 4 * num_layers);
    float* m1_gelu_out = (float*)malloc(sizeof(float) * m * n * 4 * num_layers);
    float* m1_bias_out = (float*)malloc(sizeof(float) * m * n * 4 * num_layers);
    
    float* m2_out = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* m2_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);

    unsigned int* m_mask = (unsigned int*)malloc(sizeof(unsigned int) * m * state * num_layers);
    for(int i=0;i<num_layers;i++){
        gendropoutmask(m_mask+(i*m*n), probe, seq_len*state);
    }
    float* m_after_dropout = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* add_2 = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* logits = (float*)malloc(sizeof(float) * m * vocab_size);
    int *ys_int = (int*)malloc(sizeof(int) * m);
    unsigned char *ys_char = (unsigned char*)malloc(sizeof(unsigned char) * m);
    int *one_hot_ys = (int*)malloc(sizeof(int) * m * vocab_size);

    float *softmax_logits = (float*)malloc(sizeof(float) * m * vocab_size);

    char xs_path[100];
    char temp[100];
    int *xs_int = (int*)malloc(sizeof(int) * m);
    unsigned char *xs_char = (unsigned char*)malloc(sizeof(unsigned char) * m);


    char ys_path[100];
    char y_temp[100];


    for(int iter = 0; iter<1; iter++)
    {   
        char xs_path_1[] = "xs";
        strcpy(xs_path, xs_path_1);
        sprintf(temp, "%d", iter);
        strcat(xs_path, temp);
        readbinary_char(xs_path, xs_char, m);

        for(int i=0;i<m;i++){
            xs_int[i] = (int)xs_char[i];  
        }
        embedding_lookup(embed_lookup, x_embed, xs_int, batch_size, seq_len, head_num, size_per_head);

        gendropoutmask(x_mask, probe, seq_len*state);
        gendropoutmask(p_mask, probe, 5120);

        dropout(x_after_dropout, embed_lookup, x_mask, probe, seq_len*state);
        dropout(p_after_dropout, p_embed, p_mask, probe, 5120);

        tensor_add_matrix(embed_add, x_after_dropout, p_after_dropout, batch_size, seq_len, state);
        for(int i=0; i<num_layers;i++){

            layernorm(norm_a+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), embed_add+(i*m*n), norm_a_g+(i*n), norm_a_b+(i*n), batch_size, seq_len, state);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, q_w+(i*n*n), n, 0, q_out+(i*m*n), n);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, k_w+(i*n*n), n, 0, k_out+(i*m*n), n);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, v_w+(i*n*n), n, 0, v_out+(i*m*n), n);
            bias(q_bias_out+(i*m*n), q_out+(i*m*n), q_b+(i*n), state, m);
            bias(k_bias_out+(i*m*n), k_out+(i*m*n), k_b+(i*n), state, m);
            bias(v_bias_out+(i*m*n), v_out+(i*m*n), v_b+(i*n), state, m);
            transpose_0123to0213(q_bias_out+(i*m*n), q_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213(k_bias_out+(i*m*n), k_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213(v_bias_out+(i*m*n), v_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            multiheadmatmulNT<float, float, float>(qk_out+(i*batch_size*head_num*seq_len*seq_len), q_reshape+(i*m*n), k_reshape+(i*m*n), batch_size, head_num, seq_len, size_per_head, seq_len);
            softmax<float, float>(softmax_out+(i*batch_size*head_num*seq_len*seq_len), qk_out+(i*batch_size*head_num*seq_len*seq_len), scaler, seq_len, batch_size, head_num);    
            multiheadmatmulNN<float, float, float>(sv_out+(i*m*n), softmax_out+(i*batch_size*head_num*seq_len*seq_len), v_reshape+(i*m*n), batch_size, head_num, seq_len, seq_len, size_per_head);
            transpose_0123to0213(sv_out+(i*m*n), sv_trans+(i*m*n), batch_size, head_num, seq_len, size_per_head);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, sv_trans+(i*m*n), n, a_w+(i*n*n), n, 0, a_out+(i*m*n), n);
            bias(a_bias_out+(i*m*n), a_out+(i*m*n), a_b+(i*n), state, m);
            dropout(a_after_dropout+(i*m*n), a_bias_out+(i*m*n), a_mask, probe, seq_len*state);
            tensor_add_tensor(add_1+(i*m*n), embed_add+(i*m*n), a_after_dropout+(i*m*n), batch_size, seq_len, state);
            layernorm(norm_m+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), add_1+(i*m*n), norm_m_g+(i*n), norm_m_b+(i*n), batch_size, seq_len, state);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n*4, n, 1, norm_m+(i*m*n), n, m1_w+(i*n*n*4), n*4, 0, m1_out+(i*m*n*4), n*4);
            bias(m1_bias_out+(i*m*n*4), m1_out+(i*m*n*4), m1_b+(i*n*4), state*4, m);
            gelu(m1_gelu_out+(i*m*n*4), m1_bias_out+(i*m*n*4), batch_size, seq_len, state*4);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n*4, 1, m1_gelu_out+(i*m*n*4), n*4, m2_w+(i*n*n*4), n, 0, m2_out+(i*m*n), n);
            bias(m2_bias_out+(i*m*n), m2_out+(i*m*n), m2_b+(i*n), state, m);
            dropout(m_after_dropout+(i*m*n), m2_bias_out+(i*m*n), m_mask, probe, seq_len*state);
            tensor_add_tensor(embed_add+((i+1)*m*n), add_1+(i*m*n), m_after_dropout+(i*m*n), batch_size, seq_len, state);

        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, vocab_size, n, 1, embed_add+(num_layers*m*n), n, x_embed, n, 0, logits, vocab_size);
        
        char ys_path_1[] = "ys";
        strcpy(ys_path, ys_path_1);
        sprintf(y_temp, "%d", iter);
        strcat(ys_path, y_temp);
        readbinary_char(ys_path, ys_char, m);

        for(int i=0;i<m;i++){
            ys_int[i] = (int)ys_char[i];
        }
        to_one_hot(one_hot_ys, ys_int, batch_size, seq_len, vocab_size);
        
        loss = softmax_cross_entropy_with_logits(softmax_logits, logits, one_hot_ys, batch_size, seq_len, vocab_size);
        printf("loss: %.10f \n" , loss);
    }
}