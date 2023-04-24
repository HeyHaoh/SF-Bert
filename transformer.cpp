/*********
 * 
 * g++ transformer.cpp -o transformer_fp32 -w -lm -lopenblas
 *  ./transformer_fp32 1 12 128 12 64
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
#include "iter_data.h"
#include <cblas.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

using namespace std;    

struct timeval GET_TIME_START, GET_TIME_END;

void time_check_begin()
{
    gettimeofday(&(GET_TIME_START), NULL);
}

void time_check_end()
{   
    gettimeofday(&(GET_TIME_END), NULL);
    // printf("%f \n",((GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0));
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
    float loss = 0;
    float global_norm = 0;
    float grad_scale = 1.0;
    float clip_sigma = 0;
    int n_total = n_train + n_valid + n_test;
    
    srand((unsigned int)time(NULL));
    
    unsigned char *X = (unsigned char*)malloc(sizeof(unsigned char) * n_total);
    unsigned char *trX = (unsigned char*)malloc(sizeof(unsigned char) * n_train);
    unsigned char *vaX = (unsigned char*)malloc(sizeof(unsigned char) * n_valid);
    unsigned char *teX = (unsigned char*)malloc(sizeof(unsigned char) * n_test);

    unsigned char *x = (unsigned char*)calloc(batch_size*(seq_len+1), sizeof(unsigned char));
    unsigned char *xs = (unsigned char*)malloc(sizeof(unsigned char)*batch_size*seq_len);
    unsigned char *ys = (unsigned char*)malloc(sizeof(unsigned char)*batch_size*seq_len);

    enwik8(trX, vaX, teX, X);

    free(X);
    free(vaX);
    free(teX);
    
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
    float* a_bias_out = (float*)malloc(sizeof(float) * m * n * num_layers);

    // unsigned int* a_mask = (unsigned int*)malloc(sizeof(unsigned int) * seq_len * state);
    // char a_mask_path[] = "/home/shuhui/Desktop/transformer_self_version/one_layer/a_mask";
    // readmask(a_mask_path, a_mask, seq_len*state);

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

    // unsigned int* m_mask = (unsigned int*)malloc(sizeof(unsigned int) * seq_len * state);
    // char m_mask_path[] = "/home/shuhui/Desktop/transformer_self_version/one_layer/m_mask";
    // readmask(m_mask_path, m_mask, seq_len*state);

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
    float *entropy_grad = (float*)malloc(sizeof(float) * m * vocab_size);
    float *logits_dx = (float*)malloc(sizeof(float) * m * n * (num_layers+1));
    float *logits_dw = (float*)malloc(sizeof(float) * vocab_size * n);
    float* logits_dw_trans = (float*)malloc(sizeof(float) * vocab_size * n);

    float lr  = 0.0;
    int global_step = 1;
    float beta1_power = 1.0;
    float beta2_power = 1.0;


    float *norm_a_g_mt = (float*)malloc(sizeof(float) * state * num_layers);
    float *norm_a_g_vt = (float*)malloc(sizeof(float) * state * num_layers);
    constantInit(norm_a_g_mt, 0.0, state*num_layers);
    constantInit(norm_a_g_vt, 0.0, state*num_layers);

    float *norm_a_b_mt = (float*)malloc(sizeof(float) * state *num_layers);
    float *norm_a_b_vt = (float*)malloc(sizeof(float) * state *num_layers);
    constantInit(norm_a_b_mt, 0.0, state*num_layers);
    constantInit(norm_a_b_vt, 0.0, state*num_layers);

    float *norm_m_g_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *norm_m_g_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(norm_m_g_mt, 0.0, state*num_layers);
    constantInit(norm_m_g_vt, 0.0, state*num_layers);

    float *norm_m_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *norm_m_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(norm_m_b_mt, 0.0, state*num_layers);
    constantInit(norm_m_b_vt, 0.0, state*num_layers);

    float *q_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *q_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(q_b_mt, 0.0, state*num_layers);
    constantInit(q_b_vt, 0.0, state*num_layers);

    float *k_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *k_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(k_b_mt, 0.0, state*num_layers);
    constantInit(k_b_vt, 0.0, state*num_layers);

    float *v_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *v_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(v_b_mt, 0.0, state*num_layers);
    constantInit(v_b_vt, 0.0, state*num_layers);

    float *q_w_mt = (float*)malloc(sizeof(float) * state * state*num_layers);
    float *q_w_vt = (float*)malloc(sizeof(float) * state * state*num_layers);
    constantInit(q_w_mt, 0.0, state*state*num_layers);
    constantInit(q_w_vt, 0.0, state*state*num_layers);

    float *k_w_mt = (float*)malloc(sizeof(float) * state * state*num_layers);
    float *k_w_vt = (float*)malloc(sizeof(float) * state * state*num_layers);
    constantInit(k_w_mt, 0.0, state*state*num_layers);
    constantInit(k_w_vt, 0.0, state*state*num_layers);

    float *v_w_mt = (float*)malloc(sizeof(float) * state * state*num_layers);
    float *v_w_vt = (float*)malloc(sizeof(float) * state * state*num_layers);
    constantInit(v_w_mt, 0.0, state*state*num_layers);
    constantInit(v_w_vt, 0.0, state*state*num_layers);

    float *a_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *a_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(a_b_mt, 0.0, state*num_layers);
    constantInit(a_b_vt, 0.0, state*num_layers);

    float *a_w_mt = (float*)malloc(sizeof(float) * state * state*num_layers);
    float *a_w_vt = (float*)malloc(sizeof(float) * state * state*num_layers);
    constantInit(a_w_mt, 0.0, state*state*num_layers);
    constantInit(a_w_vt, 0.0, state*state*num_layers);

    float *m1_b_mt = (float*)malloc(sizeof(float) * state * 4*num_layers);
    float *m1_b_vt = (float*)malloc(sizeof(float) * state * 4*num_layers);
    constantInit(m1_b_mt, 0.0, state*4*num_layers);
    constantInit(m1_b_vt, 0.0, state*4*num_layers);

    float *m1_w_mt = (float*)malloc(sizeof(float) * state * state * 4*num_layers);
    float *m1_w_vt = (float*)malloc(sizeof(float) * state * state * 4*num_layers);
    constantInit(m1_w_mt, 0.0, state*state*4*num_layers);
    constantInit(m1_w_vt, 0.0, state*state*4*num_layers);

    float *m2_w_mt = (float*)malloc(sizeof(float) * state * state * 4*num_layers);
    float *m2_w_vt = (float*)malloc(sizeof(float) * state * state * 4*num_layers);
    constantInit(m2_w_mt, 0.0, state*state*4*num_layers);
    constantInit(m2_w_vt, 0.0, state*state*4*num_layers);

    float *m2_b_mt = (float*)malloc(sizeof(float) * state*num_layers);
    float *m2_b_vt = (float*)malloc(sizeof(float) * state*num_layers);
    constantInit(m2_b_mt, 0.0, state*num_layers);
    constantInit(m2_b_vt, 0.0, state*num_layers);

    float* p_embed_mt = (float*)malloc(sizeof(float) * seq_len *state);
    float* p_embed_vt = (float*)malloc(sizeof(float) * seq_len *state);
    constantInit(p_embed_mt, 0.0, seq_len*state);
    constantInit(p_embed_vt, 0.0, seq_len*state);

    float* x_embed_mt = (float*)malloc(sizeof(float) * vocab_size *state);
    float* x_embed_vt = (float*)malloc(sizeof(float) * vocab_size *state);
    constantInit(x_embed_mt, 0.0, vocab_size*state);
    constantInit(x_embed_vt, 0.0, vocab_size*state);


    char xs_path[100];
    char temp[100];
    int *xs_int = (int*)malloc(sizeof(int) * m);
    unsigned char *xs_char = (unsigned char*)malloc(sizeof(unsigned char) * m);


    char ys_path[100];
    char y_temp[100];


    float* grad_sum = (float*)malloc(sizeof(float) * 16 * num_layers + 2);

    float* m_dropout_dx = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* m2_db = (float*)malloc(sizeof(float) * n * num_layers);
    float* m2_dx = (float*)malloc(sizeof(float) * m * n * 4);
    float* m2_dw = (float*)malloc(sizeof(float) * n * n * 4 * num_layers); 

    float* gelu_grad_dx = (float*)malloc(sizeof(float) * m * n * 4);
    float* m1_db = (float*)malloc(sizeof(float) * n * 4 * num_layers);
    float* m1_dx = (float*)malloc(sizeof(float) * m * n);
    float* m1_dw = (float*)malloc(sizeof(float) * n * n * 4 * num_layers);

    float* norm_m_dg = (float*)malloc(sizeof(float) * n * num_layers);
    float* norm_m_db = (float*)malloc(sizeof(float) * n * num_layers);
    float* norm_m_dx = (float*)malloc(sizeof(float) * m * n);

    float* grad_add1 = (float*)malloc(sizeof(float) * m * n);

    float* a_dropout_dx = (float*)malloc(sizeof(float) * m * n);
    float* a_db = (float*)malloc(sizeof(float) * n * num_layers);
    float* a_dx = (float*)malloc(sizeof(float) * m * n);
    float* a_dw = (float*)malloc(sizeof(float) * n * n * num_layers);
    float* a_dx_reshape = (float*)malloc(sizeof(float) * m * n);

    float *NN_grad_dw = (float*)malloc(sizeof(float) * m * n);
    float *NN_grad_dx = (float*)malloc(sizeof(float) * batch_size * head_num * seq_len * seq_len);
    float* softmax_dx = (float*)malloc(sizeof(float) * batch_size * head_num * seq_len * seq_len);
    float* NT_grad_dx = (float*)malloc(sizeof(float) * m * n);
    float* NT_grad_dw = (float*)malloc(sizeof(float) * m * n);
    float* NN_grad_dw_reshape = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* NT_grad_dx_reshape = (float*)malloc(sizeof(float) * m * n * num_layers);
    float* NT_grad_dw_reshape1 = (float*)malloc(sizeof(float) * m * n);
    float* NT_grad_dw_reshape2 = (float*)malloc(sizeof(float) * m * n * num_layers);

    float* v_db = (float*)malloc(sizeof(float) * n * num_layers);
    float* k_db = (float*)malloc(sizeof(float) * n * num_layers);
    float* q_db = (float*)malloc(sizeof(float) * n * num_layers);

    float* q_dx = (float*)malloc(sizeof(float) * m * n);
    float* q_dw = (float*)malloc(sizeof(float) * n * n * num_layers);
    float* k_dx = (float*)malloc(sizeof(float) * m * n);
    float* k_dw = (float*)malloc(sizeof(float) * n * n * num_layers);
    float* v_dx = (float*)malloc(sizeof(float) * m * n);
    float* v_dw = (float*)malloc(sizeof(float) * n * n * num_layers);

    float * grad_add2_mid = (float*)malloc(sizeof(float) * m * n);
    float * grad_add2_out = (float*)malloc(sizeof(float) * m * n);

    float* norm_a_dx = (float*)malloc(sizeof(float) * m * n);
    float* norm_a_dg = (float*)malloc(sizeof(float) * n * num_layers);
    float* norm_a_db = (float*)malloc(sizeof(float) * n * num_layers);

    float *x_dropout_dx = (float*)malloc(sizeof(float) * m * n);
    float* p_embed_grad = (float*)malloc(sizeof(float) * seq_len * state);
    float* p_dropout_dw = (float*)malloc(sizeof(float) * seq_len * state);
    float* embed_grad = (float*)malloc(sizeof(float) * n * vocab_size);
    float* embed_grad_add_out = (float*)malloc(sizeof(float) * vocab_size * state);

    int offset = rand()%seq_len;   

    int idxs_counter = 0;
    int len_idxs = ((n_train -(seq_len + 1)-offset)/seq_len);
    int *idxs = (int*)malloc(sizeof(int) * len_idxs);
    for(int i = offset; i < (n_train -(seq_len + 1)); i += seq_len){
        idxs[idxs_counter] = i;
        idxs_counter++;
    }
    if(idxs_counter - 1 != len_idxs){
        cout<<"Error: idxs allocate fail: \n"<<"      idxs_counter"<<idxs_counter<<"\n    ((X0_size -(n_timesteps + 1)-offsets)/n_timesteps):"<<len_idxs;
        exit(0);
    }


    int len=len_idxs;  /* upset the order of idxs */
    for(int i=len;i>1;i--) {    
        int cur=len-i+(rand()%i); 
        int tmp=idxs[len-i]; 
        idxs[len-i]=idxs[cur]; 
        idxs[cur]=tmp; 
    }
    
    int sequences_per_batch = batch_size;
    int length = (len_idxs/sequences_per_batch) * sequences_per_batch;
    if(length != len_idxs){
        my1_print_rank0(len_idxs - length);
    }

    /* idxs = idxs[:length] */  //but we cannot free just a part of idxs in Cpp Language
    int *idxs2 = (int*)malloc(sizeof(int)*length);
    memcpy(idxs2,idxs,sizeof(int)*length);
    free(idxs);
    //cout<<"        length aka(idxs.size) : "<<length<<endl;


    /* idxs = idxs.reshape([-1, mpi_size, n_batch]) */
    /* organize matirx as K*mpi_size*n_batch, K is length/(mpi_size*n_batch)  */
    int K = length/batch_size;
    // cout<<" Check point:K as '-1' aka len(idxs): "<<K<<endl;

    int id3_counter = 0;
    int *idxs3 = (int*)malloc(sizeof(int)*length);
    
    for(int d1=0;d1 < K;d1++){
        for(int d2=0;d2<1;d2++){
            for(int d3=0; d3<batch_size ;d3++){
                idxs3[id3_counter++] = idxs2[d1*(batch_size * 1) + d2*batch_size + d3];
            }
        }
    }
    
    

    free(idxs2);
    my2_print_rank0(K);

    // struct timeval starttime, endtime, starttime1, endtime1;
    // double time = 0.0;


    for(int iter = 0; iter<K; iter++)
    {   
        
        printf("iteration: %d\n", iter);
        for(int i=0;i<(num_layers*16+2);i++){
            grad_sum[i] = 0.0;
        }

        // char xs_path_1[] = "/home/shuhui/Desktop/transformer_self_version/xsys/xs";
        // strcpy(xs_path, xs_path_1);
        // sprintf(temp, "%d", iter);
        // strcat(xs_path, temp);
        // readbinary_char(xs_path, xs_char, m);

        // for(int i=0;i<m;i++){
        //     xs_int[i] = (int)xs_char[i];  
        // }
        int *starting_indices = (int*)malloc(sizeof(int)*batch_size);
        for(int i = 0;i<batch_size;i++){
            starting_indices[i] = idxs3[iter*(batch_size*1) + 0*batch_size + i];
        }

        for(int i = 0;i<batch_size;i++){
            for(int j = 0;j<seq_len+1;j++){
                x[i*(seq_len+1) + j] = trX[starting_indices[i] + j];
            }
        }
        /*  yield x[:, :-1], x[:, 1:] */

        for(int i = 0;i<batch_size;i++){
            for(int j = 0;j<seq_len+1;j++){
                if(j!= seq_len)//not last
                    memcpy(xs+sizeof(unsigned char)*(i*seq_len + j),x+sizeof(unsigned char)*(i*(seq_len+1) + j),sizeof(unsigned char));
                if(j!= 0)//not first
                    memcpy(ys+sizeof(unsigned char)*(i*seq_len+j-1),x+sizeof(unsigned char)*(i*(seq_len+1) + j),sizeof(unsigned char));
            } 
        }
 
        for(int i=0;i<m;i++){
            xs_int[i] = (int)xs[i];  
        }


        embedding_lookup(embed_lookup, x_embed, xs_int, batch_size, seq_len, head_num, size_per_head);
        gendropoutmask(x_mask, probe, seq_len*state);
        gendropoutmask(p_mask, probe, 5120);
        // unsigned int *x_mask= (unsigned int*)malloc(sizeof(unsigned int)* seq_len * state);
        // char x_mask_path[] = "/home/shuhui/Desktop/transformer_self_version/one_layer/x_mask";
        // readmask(x_mask_path, x_mask, (seq_len * state));

        // unsigned int *p_mask = (unsigned int*)malloc(sizeof(unsigned int)* 5120);
        // char p_mask_path[] = "/home/shuhui/Desktop/transformer_self_version/one_layer/p_mask";
        // readmask(p_mask_path, p_mask, 5120);
        dropout(x_after_dropout, embed_lookup, x_mask, probe, seq_len*state);
        dropout(p_after_dropout, p_embed, p_mask, probe, 5120);
        tensor_add_matrix(embed_add, x_after_dropout, p_after_dropout, batch_size, seq_len, state);
        
        
        
        for(int i=0; i<num_layers;i++){
            
            time_check_begin();
            layernorm(norm_a+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), embed_add+(i*m*n), norm_a_g+(i*n), norm_a_b+(i*n), batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, q_w+(i*n*n), n, 0, q_out+(i*m*n), n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, k_w+(i*n*n), n, 0, k_out+(i*m*n), n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, norm_a+(i*m*n), n, v_w+(i*n*n), n, 0, v_out+(i*m*n), n);
            time_check_end();
            time_check_begin();
            bias(q_bias_out+(i*m*n), q_out+(i*m*n), q_b+(i*n), state, m);
            time_check_end();
            time_check_begin();
            bias(k_bias_out+(i*m*n), k_out+(i*m*n), k_b+(i*n), state, m);
            time_check_end();
            time_check_begin();
            bias(v_bias_out+(i*m*n), v_out+(i*m*n), v_b+(i*n), state, m);
            time_check_end();
            transpose_0123to0213(q_bias_out+(i*m*n), q_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213(k_bias_out+(i*m*n), k_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            transpose_0123to0213(v_bias_out+(i*m*n), v_reshape+(i*m*n), batch_size, seq_len, head_num, size_per_head);
            time_check_begin();
            multiheadmatmulNT<float, float, float>(qk_out+(i*batch_size*head_num*seq_len*seq_len), q_reshape+(i*m*n), k_reshape+(i*m*n), batch_size, head_num, seq_len, size_per_head, seq_len);
            time_check_end();
            time_check_begin();
            softmax<float, float>(softmax_out+(i*batch_size*head_num*seq_len*seq_len), qk_out+(i*batch_size*head_num*seq_len*seq_len), scaler, seq_len, batch_size, head_num);    
            time_check_end();
            time_check_begin();
            multiheadmatmulNN<float, float, float>(sv_out+(i*m*n), softmax_out+(i*batch_size*head_num*seq_len*seq_len), v_reshape+(i*m*n), batch_size, head_num, seq_len, seq_len, size_per_head);
            time_check_end();
            transpose_0123to0213(sv_out+(i*m*n), sv_trans+(i*m*n), batch_size, head_num, seq_len, size_per_head);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1, sv_trans+(i*m*n), n, a_w+(i*n*n), n, 0, a_out+(i*m*n), n);
            time_check_end();
            time_check_begin();
            bias(a_bias_out+(i*m*n), a_out+(i*m*n), a_b+(i*n), state, m);
            time_check_end();
            time_check_begin();
            dropout(a_after_dropout+(i*m*n), a_bias_out+(i*m*n), a_mask, probe, seq_len*state);
            time_check_end();
            time_check_begin();
            tensor_add_tensor(add_1+(i*m*n), embed_add+(i*m*n), a_after_dropout+(i*m*n), batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            layernorm(norm_m+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), add_1+(i*m*n), norm_m_g+(i*n), norm_m_b+(i*n), batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n*4, n, 1, norm_m+(i*m*n), n, m1_w+(i*n*n*4), n*4, 0, m1_out+(i*m*n*4), n*4);
            time_check_end();
            time_check_begin();
            bias(m1_bias_out+(i*m*n*4), m1_out+(i*m*n*4), m1_b+(i*n*4), state*4, m);
            time_check_end();
            time_check_begin();
            gelu(m1_gelu_out+(i*m*n*4), m1_bias_out+(i*m*n*4), batch_size, seq_len, state*4);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n*4, 1, m1_gelu_out+(i*m*n*4), n*4, m2_w+(i*n*n*4), n, 0, m2_out+(i*m*n), n);
            time_check_end();
            time_check_begin();
            bias(m2_bias_out+(i*m*n), m2_out+(i*m*n), m2_b+(i*n), state, m);
            time_check_end();
            time_check_begin();
            dropout(m_after_dropout+(i*m*n), m2_bias_out+(i*m*n), m_mask, probe, seq_len*state);
            time_check_end();
            time_check_begin();
            tensor_add_tensor(embed_add+((i+1)*m*n), add_1+(i*m*n), m_after_dropout+(i*m*n), batch_size, seq_len, state);
            time_check_end();

        }

        time_check_begin();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, vocab_size, n, 1, embed_add+(num_layers*m*n), n, x_embed, n, 0, logits, vocab_size);
        time_check_end();
        // char ys_path_1[] = "/home/shuhui/Desktop/transformer_self_version/xsys/ys";
        // strcpy(ys_path, ys_path_1);
        // sprintf(y_temp, "%d", iter);
        // strcat(ys_path, y_temp);
        // readbinary_char(ys_path, ys_char, m);
        // for(int i=0;i<m;i++){
        //     ys_int[i] = (int)ys_char[i];
        // }

        for(int i=0;i<m;i++){
            ys_int[i] = (int)ys[i];
        }
        
        to_one_hot(one_hot_ys, ys_int, batch_size, seq_len, vocab_size);
        time_check_begin();
        loss = softmax_cross_entropy_with_logits(softmax_logits, logits, one_hot_ys, batch_size, seq_len, vocab_size);
        time_check_end();
        printf("loss: %.10f \n" , loss);

        time_check_begin();
        cross_entropy_grad(entropy_grad, softmax_logits, batch_size, seq_len, vocab_size);
        time_check_end();

        time_check_begin();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, vocab_size, 1, entropy_grad, vocab_size, x_embed, n, 0, logits_dx+(num_layers*m*n), n);
        time_check_end();
        time_check_begin();
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, vocab_size, m, 1, embed_add+(num_layers*m*n), n, entropy_grad, vocab_size, 0, logits_dw, vocab_size);
        time_check_end();
        transpose_01to10(logits_dw, logits_dw_trans, n, vocab_size);

        for(int i=num_layers-1; i>=0; i--){

            time_check_begin();
            dropout(m_dropout_dx, logits_dx+((i+1)*m*n), m_mask, probe, seq_len*state);
            time_check_end();
            time_check_begin();
            bias_grad_db(m2_db+(i*n), m_dropout_dx, batch_size, seq_len, state);
            time_check_end();
            grad_sum[17+i*16] = gradients_add(m2_db+(i*n), 1, n);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n*4, n, 1, m_dropout_dx, n, m2_w+(i*n*n*4), n, 0, m2_dx, n*4);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n*4, n, m, 1, m1_gelu_out+(i*m*n*4), n*4, m_dropout_dx, n, 0, m2_dw+(i*n*n*4), n);
            time_check_end();
            grad_sum[16+i*16] = gradients_add(m2_dw+(i*n*n*4), n, n*4);
            time_check_begin();
            gelu_grad(gelu_grad_dx, m2_dx, m1_bias_out+(i*m*n*4), m1_b+(i*n*4), batch_size, seq_len, state*4);
            time_check_end();
            time_check_begin();
            bias_grad_db(m1_db+(i*n*4), gelu_grad_dx, batch_size, seq_len , state*4);
            time_check_end();
            grad_sum[15+i*16] = gradients_add(m1_db+(i*n*4), 1, n*4);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n*4, 1, gelu_grad_dx, n*4, m1_w+(i*n*n*4), n*4, 0, m1_dx, n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n*4, m, 1, norm_m+(i*m*n), n, gelu_grad_dx, n*4, 0, m1_dw+(i*n*n*4), n*4);
            time_check_end();
            grad_sum[14+i*16] = gradients_add(m1_dw+(i*n*n*4), n, n*4);
            time_check_begin();
            layernorm_dg_db(norm_m_dg+(i*n), norm_m_db+(i*n), m1_dx, add_1+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), batch_size, seq_len, state);
            time_check_end();
            grad_sum[13+i*16] = gradients_add(norm_m_dg+(i*n), 1, n);
            grad_sum[12+i*16] = gradients_add(norm_m_db+(i*n), 1, n);
            time_check_begin();
            layernorm_grad_dx(norm_m_dx, add_1+(i*m*n), norm_m_mean+(i*m), norm_m_rstd+(i*m), m1_dx, norm_m_g+(i*n), batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            tensor_add_tensor(grad_add1, logits_dx+((i+1)*m*n), norm_m_dx, batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            dropout(a_dropout_dx, grad_add1, a_mask, probe, seq_len*state);
            time_check_end();
            time_check_begin();
            bias_grad_db(a_db+(i*n), a_dropout_dx, batch_size, seq_len, state);
            time_check_end();
            grad_sum[11+i*16] = gradients_add(a_db+(i*n), 1, n);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, a_dropout_dx, n, a_w+(i*n*n), n, 0, a_dx, n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, sv_trans+(i*m*n), n, a_dropout_dx, n, 0, a_dw+(i*n*n), n);
            time_check_end();
            grad_sum[10+i*16] = gradients_add(a_dw+(i*n*n), n, n);
            transpose_0123to0213(a_dx, a_dx_reshape, batch_size, seq_len, head_num, size_per_head);
            time_check_begin();
            multiheadmatmulNT<float, float, float>(NN_grad_dx, a_dx_reshape, v_reshape+(i*m*n), batch_size, head_num, seq_len, size_per_head, seq_len);
            time_check_end();
            time_check_begin();
            multiheadmatmulTN<float, float, float>(NN_grad_dw, softmax_out+(i*head_num*seq_len*seq_len), a_dx_reshape, batch_size, head_num, seq_len, seq_len, size_per_head);
            time_check_end();
            time_check_begin();
            softmax_grad<float, float, float>(softmax_dx, NN_grad_dx, softmax_out+(i*head_num*seq_len*seq_len), scaler, batch_size, head_num, seq_len);
            time_check_end();
            time_check_begin();
            multiheadmatmulNN(NT_grad_dx, softmax_dx, k_reshape+(i*m*n), batch_size, head_num, seq_len, seq_len, size_per_head);
            time_check_end();
            time_check_begin();
            multiheadmatmulTN(NT_grad_dw, q_reshape+(i*m*n), softmax_dx, batch_size, head_num, seq_len, size_per_head, seq_len);
            time_check_end();
            transpose_0123to0213(NN_grad_dw, NN_grad_dw_reshape+(i*m*n), batch_size, head_num, seq_len, size_per_head);
            transpose_0123to0213(NT_grad_dx, NT_grad_dx_reshape+(i*m*n), batch_size, head_num, seq_len, size_per_head);
            transpose_0123to0132(NT_grad_dw, NT_grad_dw_reshape1, batch_size, head_num, size_per_head, seq_len);
            transpose_0123to0213(NT_grad_dw_reshape1, NT_grad_dw_reshape2+(i*m*n), batch_size, head_num, seq_len, size_per_head);
            time_check_begin();
            bias_grad_db(v_db+(i*n), NN_grad_dw_reshape+(i*m*n), batch_size, seq_len, state);
            time_check_end();
            grad_sum[9+i*16] = gradients_add(v_db+(i*n), 1, n);
            time_check_begin();
            bias_grad_db(k_db+(i*n), NT_grad_dw_reshape2+(i*m*n), batch_size, seq_len, state);
            time_check_end();
            grad_sum[8+i*16] = gradients_add(k_db+(i*n), 1, n);
            time_check_begin();
            bias_grad_db(q_db+(i*n), NT_grad_dx_reshape+(i*m*n), batch_size, seq_len, state);
            time_check_end();
            grad_sum[7+i*16] = gradients_add(q_db+(i*n), 1, n);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, NT_grad_dx_reshape+(i*m*n), n, q_w+(i*n*n), n, 0, q_dx, n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a+(i*m*n), n, NT_grad_dx_reshape+(i*m*n), n, 0, q_dw+(i*n*n), n);
            time_check_end();
            grad_sum[6+i*16] = gradients_add(q_dw+(i*n*n), n, n);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, NT_grad_dw_reshape2+(i*m*n), n, k_w+(i*n*n), n, 0, k_dx, n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a+(i*m*n), n, NT_grad_dw_reshape2+(i*m*n), n, 0, k_dw+(i*n*n), n);
            time_check_end();
            grad_sum[5+i*16] = gradients_add(k_dw+(i*n*n), n, n);
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, n, 1, NN_grad_dw_reshape+(i*m*n), n, v_w+(i*n*n), n, 0, v_dx, n);
            time_check_end();
            time_check_begin();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, 1, norm_a+(i*m*n), n, NN_grad_dw_reshape+(i*m*n), n, 0, v_dw+(i*n*n), n);
            time_check_end();
            grad_sum[4+i*16] = gradients_add(v_dw+(i*n*n), n, n);
            time_check_begin();
            tensor_add_tensor(grad_add2_mid, q_dx, k_dx, batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            tensor_add_tensor(grad_add2_out, grad_add2_mid, v_dx, batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            layernorm_dg_db(norm_a_dg+(i*n), norm_a_db+(i*n), grad_add2_out, embed_add+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), batch_size, seq_len, state);
            time_check_end();
            time_check_begin();
            layernorm_grad_dx(norm_a_dx, embed_add+(i*m*n), norm_a_mean+(i*m), norm_a_rstd+(i*m), grad_add2_out, norm_a_g+(i*n), batch_size, seq_len, state);
            time_check_end();
            grad_sum[3+i*16] = gradients_add(norm_a_dg+(i*n), 1, n);
            grad_sum[2+i*16] = gradients_add(norm_a_db+(i*n), 1, n);
            time_check_begin();
            tensor_add_tensor(logits_dx+(i*m*n), grad_add1, norm_a_dx, batch_size, seq_len, state);
            time_check_end();
        }

        dropout(x_dropout_dx, logits_dx, x_mask, probe, seq_len*state);
    
        add_grad(p_embed_grad, x_dropout_dx, batch_size, seq_len, state);

        dropout(p_dropout_dw, p_embed_grad, p_mask, probe, 5120);
        grad_sum[1] = gradients_add(p_dropout_dw, seq_len, state);

        embedding_lookup_grad(embed_grad, x_dropout_dx, xs_int, batch_size, seq_len, vocab_size, state);

        tensor_add_tensor(embed_grad_add_out, logits_dw_trans, embed_grad, 1, vocab_size, state);
        grad_sum[0] = gradients_add(embed_grad_add_out, vocab_size, state);
        
        grad_sum_sum = 0.0;
        for(int i=0;i<(num_layers*16+2);i++){
            // printf("%.25f \n", grad_sum[i]);
            grad_sum_sum += grad_sum[i];
        }

        global_norm = sqrt(grad_sum_sum);
        norm_scale = clip_by_global_norm(global_norm, clip_norm);

        printf("global_norm:%.10f\n", global_norm);
        printf("norm_sacle:%.10f\n", norm_scale);

        lr = global_step * (1.0/1000) < 1 ? global_step * (1.0/1000) : 1;
        lr *= learning_rate;

        beta1_power = adam_got_beta_power(beta1, global_step);
        beta2_power = adam_got_beta_power(beta2, global_step);

        float lr_2 = adam_got_lr(lr, beta1_power, beta2_power);
        
        adam_apply_gradients(norm_a_g, norm_a_dg, norm_a_g_mt, norm_a_g_vt, beta1, beta2, beta1_power, beta2_power, 
                            lr_2, epsilon, norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(norm_a_b, norm_a_db, norm_a_b_mt, norm_a_b_vt, beta1, beta2, beta1_power, beta2_power,
                            lr_2, epsilon, norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(norm_m_g, norm_m_dg, norm_m_g_mt, norm_m_g_vt, beta1, beta2, beta1_power, beta2_power, 
                            lr_2, epsilon, norm_scale, grad_scale, clip_sigma, n*num_layers);  
        adam_apply_gradients(norm_m_b, norm_m_db, norm_m_b_mt, norm_m_b_vt, beta1, beta2, beta1_power, beta2_power,
                            lr_2, epsilon, norm_scale, grad_scale, clip_sigma, n*num_layers);  
        adam_apply_gradients(k_w, k_dw, k_w_mt, k_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*n*num_layers);
        adam_apply_gradients(q_w, q_dw, q_w_mt, q_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*n*num_layers);
        adam_apply_gradients(v_w, v_dw, v_w_mt, v_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*n*num_layers);
        adam_apply_gradients(k_b, k_db, k_b_mt, k_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(q_b, q_db, q_b_mt, q_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(v_b, v_db, v_b_mt, v_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(a_b, a_db, a_b_mt, a_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(m1_b, m1_db, m1_b_mt, m1_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*4*num_layers);
        adam_apply_gradients(m2_b, m2_db, m2_b_mt, m2_b_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*num_layers);
        adam_apply_gradients(a_w, a_dw, a_w_mt, a_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*n*num_layers);
        adam_apply_gradients(m1_w, m1_dw, m1_w_mt, m1_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, n*n*4*num_layers);
        adam_apply_gradients(m2_w, m2_dw, m2_w_mt, m2_w_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                                norm_scale, grad_scale, clip_sigma, n*n*4*num_layers);
        

        adam_apply_gradients(p_embed, p_dropout_dw, p_embed_mt, p_embed_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, seq_len*state);
        adam_apply_gradients(x_embed, embed_grad_add_out, x_embed_mt, x_embed_vt, beta1, beta2, beta1_power, beta2_power, lr_2, epsilon,
                            norm_scale, grad_scale, clip_sigma, vocab_size*state);
    

        global_step++;
    }
    
    
    return 0;

}
