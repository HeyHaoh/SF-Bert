#include <stdio.h>
#include <cmath>
#include "common.h"

#define ex2 1.4426950408889634f
#define PI 3.141592654
#define URAND_SCALE 2.3283064365386962891e-10f

//gpu done
void bias(float* output, const float* input, const float* bias, int n, int m){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            output[i*n+j] = 0;
            output[i*n+j] = input[i*n+j] + bias[j];
        }
    }
}

//gpu done
//db = sum(dy)
void bias_grad_db(float* db, float* dy, const int batch_size, const int seq_len, const int state){

    for(int i=0; i<state; i++){
        db[i] = 0.0;
        for(int j=0; j<batch_size*seq_len; j++){
            db[i] += dy[j*state + i];
        }
    }
}

//gpu done
void gelu(float *C, float *A, const int batch_size, const int seq_len, const int state){

    for(int i=0;i<(batch_size*seq_len*state);i++){
        float ex = 0.0;
        float rcp = 0.0;
        ex =  exp(-(1.702 * A[i]));
        rcp  = 1.0/ (ex+1.0);
        C[i] = 0;
        C[i] = A[i] * rcp;
    }

}

// gpu done
void gelu_grad(float *out, float* dy, float* x, float* b, const int batch_size, const int seq_len, const int state){

    for(int i=0; i<batch_size * seq_len; i++){
        float sig = 0.0;
        float sig_sqr = 0.0;
        for(int j=0; j<state; j++){
            sig = exp(-1.702 * (x[i*state+j] + b[j])) + 1.0;
            sig  = 1.0 / sig;
            sig_sqr = sig * sig;

            out[i*state+j] = dy[i*state+j] * sig;
            out[i*state+j] += (1.702 * (x[i*state+j]+b[j]) * dy[i*state+j] * (sig - sig_sqr));
        }
    }
}

//gpu done
template<typename T1, typename T2>
void softmax(T1* C, T2* A, float scaler, const int seq_len, const int batch_size, const int head_num){
    for(int i=0;i<batch_size;i++)
    {
        for(int j=0;j<head_num;j++){
            for(int k=0;k<seq_len;k++)
            {
                float max = -1000000.0;
                float sum_of_C = 0.0;
                for(int m=0;m<seq_len;m++)
                {   
                    if(k>=m){
                        if(A[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m] > max)
                            max = A[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m];
                    }
                }

                for(int m=0;m<seq_len;m++)
                {      
                    C[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m] = 0.0;
                    if(k>=m)
                    {
                        C[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m] = 
                        exp((A[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m] - max) * scaler);
                        sum_of_C += C[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m]; 
                    }  
                }

                for(int m=0;m<seq_len;m++)
                {
                    C[i*head_num*seq_len*seq_len + j*seq_len*seq_len + k*seq_len + m] /= (sum_of_C + 1e-6f);
                }
            }  
        }    
    }
}

//
template<typename T1, typename T2, typename T3>
void softmax_grad(T1* out, T2* dy, T3* y, float scaler, const int batch_size, const int head_num, const int seq_len){

    float *sum_dyy = (float*)malloc(sizeof(float) * seq_len * batch_size * head_num);
    for(int i=0;i<batch_size*seq_len*head_num;i++){
        sum_dyy[i] = 0.0;
    }
    for(int i=0;i<batch_size*head_num;i++){
        for(int j =0;j<seq_len;j++){
            sum_dyy[i*seq_len + j] = 0;
            for(int k=0; k<seq_len; k++){

                sum_dyy[i*seq_len + j] += (dy[i*seq_len*seq_len + j*seq_len + k] * y[i*seq_len*seq_len + j*seq_len + k]);
            }

            for(int k=0;k<seq_len;k++){
                out[i*seq_len*seq_len + j*seq_len + k] = (dy[i*seq_len*seq_len + j*seq_len + k] - sum_dyy[i*seq_len + j]) 
                                                            * y[i*seq_len*seq_len + j*seq_len + k] * scaler;
            }    
        }
    }
    free(sum_dyy);
}

//gpu done
void gendropoutmask(unsigned int *mask, float keep_prob, int size){

    for(int i=0;i<size;i++){
        unsigned int* temp = (unsigned int*)malloc(sizeof(unsigned int) * 32);
        for(int j=0;j<32;j++){
            int lfsr0, lfsr1, lfsr2;
            lfsr0 = rand();
            lfsr1 = rand();
            lfsr2 = rand();
            lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
            lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
            lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
            unsigned int urand = lfsr0 ^ lfsr1 ^ lfsr2;
            bool keep  = (float)urand * URAND_SCALE < keep_prob;
            unsigned int anti = 0;
            anti =  1 << (32-j);
            anti = ~anti;
            temp[j] = keep << (32-j);
            temp[j] = temp[j] | anti;
        }

        for(int j=1;j<32;j++){
            temp[j] &= temp[j-1];         
        }

        mask[i] = temp[31];
    }
}

//gpu done
void dropout(float* C, float* A, unsigned int* M, const float probe, const int size_of_mask){
    for(int i=0; i<size_of_mask;i++){
        unsigned int mask = M[i];
        for(int j=0; j<32; j++){
            if(((1 << j) & mask) == 0)  
            {   
                C[i*32+j] = 0.0;

            }else{

                C[i*32+j] = A[i*32+j] * (1.0 /probe);
            }
        }
    }
}

void dropout1(float* C, float* A, unsigned int* M, const float probe, const int size_of_mask, const int batch_size, const int seq_len, const int state){
    for(int j=0;j<seq_len;j++){
        for(int k=0;k<state;k++){
            unsigned int mask = M[j*state+k];
            for(int i=0;i<batch_size;i++){
                if(((1 << i) & mask) == 0){

                    C[i*(seq_len*state)+j*state+k] = 0.0;

                }else{
                    
                    C[i*(seq_len*state)+j*state+k] = A[i*(seq_len*state)+j*state+k] * (1.0/probe);

                }
            }
        }
    }
}

void dropout2(float* C, float* A, unsigned int* M, const float probe, const int size_of_mask, const int state){

    for(int j=0;j<10;j++){
        for(int k=0;k<512;k++){
            unsigned int mask = M[j*state+k];
            for(int i=0;i<32;i++){
                if(((1 << i) & mask) == 0){

                    C[i*(10*state)+j*state+k] = 0.0;

                }else{
                    
                    C[i*(10*state)+j*state+k] = A[i*(10*state)+j*state+k] * (1.0/probe);

                }
            }
        }
    }
}

// void dropout_mask(int* mask_int, unsigned int* M, const int size_of_mask){
//     for(int i=0;i<size_of_mask;i++){
//         unsigned int mask = M[i];
//         for(int j=0;j<32;j++){
//             if(((1 << j) & mask) == 0){
//                 mask_int[i*32+j] = 0;
//             }else{
//                 mask_int[i*32+j] = 1;
//             }
//         }
//     }
// }


// void dropout_cpu(float* C, float* A, unsigned int* M, float probe, const int dim_0, const int dim_1, const int dim_2){
//     for(int i=0;i<dim_0*dim_1*dim_2;i++){
//             C[i] = 0.0;
//             C[i] = A[i] * M[i] * (1.0 / probe);
//     }
// }

//gpu done
float softmax_cross_entropy_with_logits(float *D, float *A, int *B, const int batch_size, const int seq_len, const int vocab_size){

    float *loss = (float*)malloc(sizeof(float)*batch_size*seq_len);
    float sum_of_loss = 0;

    for(int i=0;i<(batch_size*seq_len);i++){
        float sum_of_row = 0.0;
        loss[i] = 0; 
        for(int j=0;j<vocab_size;j++){

            A[i*vocab_size + j] = exp(A[i*vocab_size + j]);
            sum_of_row += A[i*vocab_size + j];
        }

        for(int j=0;j<vocab_size;j++){
            D[i*vocab_size + j] = A[i*vocab_size + j] / sum_of_row;
            A[i*vocab_size + j] = log(D[i*vocab_size + j]);
            D[i*vocab_size + j] -= B[i*vocab_size+j];
            loss[i] += -(A[i*vocab_size +j] * B[i*vocab_size+j]);
        }
        sum_of_loss += loss[i];
        
    }
    sum_of_loss = sum_of_loss / (batch_size * seq_len);

    free(loss);

    return sum_of_loss;
}

//gpu done
void cross_entropy_grad(float *C, float *A, const int batch_size, const int seq_len, const int vocab_size){

    for(int i=0;i<batch_size;i++){
        for(int j=0;j<seq_len;j++){
            for(int k=0;k<vocab_size;k++){
                C[i*seq_len*vocab_size + j*vocab_size + k] = A[i*seq_len*vocab_size + j*vocab_size + k] / (batch_size*seq_len);
            }
        }
    }
    
}

void tensor_add_matrix(float* C, float* A, float* B, const int dim_0, const int dim_1, const int dim_2){
    for(int i=0;i<dim_0;i++){
        for(int j=0;j<(dim_1 * dim_2);j++){
            C[i*(dim_1 * dim_2) + j] = A[i*(dim_1 * dim_2) + j] + B[j]; 
        }
    }
}

void add_grad(float* C, float* A, const int dim_0, const int dim_1, const int dim_2){

     for(int j=0;j<dim_1*dim_2;j++){
        C[j] = 0; 
        for(int i=0;i<dim_0;i++){
            C[j] += A[i*(dim_1 * dim_2) + j];
        }
            
    }
}

//gpu done
void tensor_add_tensor(float *C, float *A, float* B, const int dim_0, const int dim_1, const int dim_2){
    for(int i=0;i<(dim_0 * dim_1 * dim_2);i++){
        C[i] = 0.0;
        C[i] = A[i] + B[i];
    }
}

// (01) to (10)
void transpose_01to10(float* src, float* des, const int dim_0, const int dim_1){
    for(int i=0;i<dim_0;i++){
        for(int j=0;j<dim_1;j++){
            des[j*dim_0+i] = 0.0;
            des[j*dim_0+i] = src[i*dim_1+j];
        }
    }
}

//(0123) to (0213)
void transpose_0123to0213(float* src, float* des, const int dim_0, const int dim_1, const int dim_2, const int dim_3){

    for(int i=0;i<dim_0;i++){
        for(int j=0;j<dim_1;j++){
            for(int k=0;k<dim_2;k++){
                for(int m=0;m<dim_3;m++){
                    int index_src = 0;
                    index_src = i * (dim_1 * dim_2 * dim_3) + j * (dim_2 * dim_3) + k * dim_3 + m;
                    int index_des = 0;
                    index_des = i * (dim_1 * dim_2 * dim_3) + k * (dim_1 * dim_3) + j * dim_3 + m;
                    des[index_des] = 0.0; 
                    des[index_des] = src[index_src];
                }
            }
        }
    }

}

//(0123) to (0132)
void transpose_0123to0132(float* src, float* des, const int dim_0, const int dim_1, const int dim_2, const int dim_3){
    for(int i=0;i<dim_0;i++){
        for(int j=0;j<dim_1;j++){
            for(int k=0;k<dim_2;k++){
                for(int m=0;m<dim_3;m++){
                    int index_src = 0;
                    index_src = i * dim_1 * dim_2 * dim_3 + j * dim_2 * dim_3 + k * dim_3 + m;
                    int index_des = 0;
                    index_des = i * dim_1 * dim_2 * dim_3 + j * dim_2 * dim_3 + m * dim_2 + k;
                    des[index_des] = 0.0;
                    des[index_des] = src[index_src];
                }
            }
        }
    }
}


void to_one_hot(int *B, int *A, const int batch_size, const int seq_len, const int vocab_size){
    for(int i=0;i<(batch_size*seq_len);i++){
        for(int j=0;j<vocab_size;j++){
            B[i*vocab_size+j] = 0;
        }
        int num = A[i];
        B[i*vocab_size + num] = 1;
    }

}

float gradients_add(float* A, int dim_0, int dim_1){
    float C = 0.0;
    for(int i=0;i<(dim_0 * dim_1);i++){
            C += (A[i] * A[i]);
    }
    return C;
}


float add_1(float* A, int dim_0, int dim_1){
    float C = 0;
    for(int i=0;i<dim_0*dim_1;i++){
            C += (A[i]);
    }
    return C;
}

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
      data[i] = (float)(rand()%BERT_RAND_MAX) / (float)BERT_RAND_MAX;
}


void constantInit(VALUE_TYPE *data, VALUE_TYPE constant, int size){

    for(int i=0;i<size;i++){
        data[i] = constant;
    }
}

void random_normalInit(VALUE_TYPE *data, VALUE_TYPE mean, VALUE_TYPE stdv, int size){
    static int phase = 0;
    double U1,U2,V1,V2,S,Z;
    for(int i=0;i<size;i++){
        do{
            U1 = (double)((double)(rand()) / (double)(RAND_MAX));
            U2 = (double)((double)(rand()) / (double)(RAND_MAX));
            V1 = 2*U1-1;
            V2 = 2*U2-1;
            S = V1*V1 + V2*V2;
        }while(S>=1 || S==0);

        if(phase == 0){
            Z = V1 * sqrt(-2 * log(S) / S);
        }else{
            Z = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;
        data[i] = Z * stdv + mean;
    }

}

void random_binomal(unsigned int* C, int N, float keep_probe, int size){
    for(int i=0;i<size;i++){
        unsigned int rnd = 0;
        for(int j=0;j<N;j++){
            double pV = (double)(rand())/(double)RAND_MAX;
            if(pV<keep_probe){
                rnd++;
            }
        }
        C[i] = rnd;
    }
}

void generate_csr_full_mask(int *Rowptr, int *Colidx, const int batch_size, const int head, const int seq_len){
    int nnz_of_row = 0;
    Rowptr[0] = 0;
    for(int i=0;i<batch_size*head*seq_len;i++){
        for(int j=0; j<seq_len; j++){
            if(j<=(i%(seq_len))){
                Colidx[nnz_of_row] = j;
                nnz_of_row++;
                
            }
        }
        Rowptr[i+1] = nnz_of_row;
    }
}

void genenrate_csr_full_mask_v2(int *Rowptr, int *Colidx, const int batch_size, const int head, const int seq_len){
    int nnz_of_row = 0;
    Rowptr[0] = 0;
    for(int i=0;i<batch_size*head*seq_len;i++){
        for(int j=0;j<seq_len;j++){
            if(j<=(i%seq_len)){
                nnz_of_row++;
                Colidx[i*seq_len+j] = j;
            }else{
                Colidx[i*seq_len+j] = -1;
            }
        }
    }

}

//strided
void generate_qk_mask(int *mask, int local_atten_ctx, int seq_len){

    for(int i=0;i<seq_len;i++){
        for(int j=0;j<=local_atten_ctx;j++){
            if((i+j)<seq_len){
                mask[(i+j)*seq_len+i] = 1;
            }
        }
    }

    int *q = (int*)malloc(sizeof(int)*seq_len*seq_len);
    int *k = (int*)malloc(sizeof(int)*seq_len*seq_len);

    for(int i=0;i<seq_len;i++){
        for(int j=0;j<seq_len;j++){
            q[i*seq_len+j] = i;
            k[i*seq_len+j] = j;
        }
    }

    int c1, c2;
    int c;
    for(int i=0; i<seq_len*seq_len;i++){
        if(q[i] >= k[i]){
            c1 = 1;
        }else{
            c1 = 0;
        }

        if(((q[i] - k[i])%32) == 0){
            c2 = 1; 
        }else{
            c2 = 0;
        }

        c = c1 && c2;
        mask[i] = mask[i] || c; 
    }

    free(q);
    free(k);
}

//big bird
void generate_qk_mask_big_bird(int *mask, const int seq_len, const int window, const int global){
    //window
    for(int i=0;i<seq_len;i++){
        for(int j=0;j<=window;j++){
            if(((i+j)<seq_len)){
                mask[(i+j)*seq_len+i] = 1;
            }

            if((i-j) >= 0){
                mask[(i-j)*seq_len+i] = 1;
            }
        }
    }
    
    
    int *q = (int*)malloc(sizeof(int)*seq_len*seq_len);   
    //global
    for(int i=0;i<global;i++){
        for(int j=0; j<seq_len;j++){
            q[i*seq_len+j] = 1;
        }
    }

    for(int j=0;j<global;j++){
        for(int i=0;i<seq_len;i++){
            q[i*seq_len+j] = 1;
        }
    }


    int *k = (int*)malloc(sizeof(int)*seq_len*seq_len); 
    //random

    int randomnum1 = 0;
    int randomnum2 = randomnum1;

    for(int i=0;i<seq_len;i++){
        randomnum1 = rand() % 10;
    
        do{
            randomnum2 = rand() % 10;

        }while(randomnum2 != randomnum1);

        for(int j=0;j<32;j++){

            k[i*seq_len+randomnum1*32+j] = 1;
            k[i*seq_len+randomnum2*32+j] = 1;

        }
    }

    for(int i=0;i<seq_len*seq_len;i++){
        mask[i] = mask[i] || q[i] || k[i];
    }

}

//fixed
void generate_qk_mask_fixed(int *mask, const int seq_len, const int strided){

    for(int i=0;i<seq_len;i++){
        for(int j=0;j<32;j++){
            mask[i*seq_len+j] = 1;
        }

        for(int j=strided*32;j<(strided+1)*32;j++){
            mask[i*seq_len+j] = 1;
        }

        for(int j=strided*2*32;j<(strided*2+1)*32;j++){
            mask[i*seq_len+j] = 1;
        }
    }

    int *q = (int*)malloc(sizeof(int)*seq_len*seq_len); 

    for(int i=0;i<256;i++){
        int blk_num = i / 32;
        for(int j=blk_num*32;j<(blk_num+strided)*32;j++)
            q[i*seq_len+j] = 1;
    }

    for(int i=256;i<seq_len;i++){
        int blk_num = i/32;
        for(int j=blk_num*32;j<seq_len;j++){
            q[i*seq_len+j] = 1;
        }
    }
    
    for(int i=0; i<seq_len;i++){
        for(int j=0;j<seq_len;j++){
            mask[i*seq_len+j] = mask[i*seq_len+j] || q[i*seq_len+j];
            if(i<=j){
                mask[i*seq_len+j] = 0;
            }
        }
    }

    free(q);

}

void generate_csr_fixed_mask(int *RowptrM, int *ColidxM, const int seq_len, const int strided){

    int *mask = (int*)malloc(sizeof(int)*seq_len*seq_len);
    for(int i=0;i<seq_len;i++){
        for(int j=0;j<32;j++){
            mask[i*seq_len+j] = 1;
        }

        for(int j=strided*32;j<(strided+1)*32;j++){
            mask[i*seq_len+j] = 1;
        }

        for(int j=strided*2*32;j<(strided*2+1)*32;j++){
            mask[i*seq_len+j] = 1;
        }
    }

    int *q = (int*)malloc(sizeof(int)*seq_len*seq_len); 

    for(int i=0;i<256;i++){
        int blk_num = i / 32;
        for(int j=blk_num*32;j<(blk_num+strided)*32;j++)
            q[i*seq_len+j] = 1;
    }

    for(int i=256;i<seq_len;i++){
        int blk_num = i/32;
        for(int j=blk_num*32;j<seq_len;j++){
            q[i*seq_len+j] = 1;
        }
    }
    
    int nnz = 0;
    for(int i=0; i<seq_len;i++){
        for(int j=0;j<seq_len;j++){
            mask[i*seq_len+j] = mask[i*seq_len+j] || q[i*seq_len+j];
            if(i<=j){
                mask[i*seq_len+j] = 0;
            }
            if(mask[i*seq_len+j] !=0){
                ColidxM[nnz] = j;
                nnz++;
            }
        }
        RowptrM[i+1] = nnz;
    }

    free(q);
    free(mask);

}

// void generate_nt_lut(uint2 *nt_lut, const uint ctx_blks_a){
//     int count = 0;
//     for(int i=0;i<ctx_blks_a;i++){
//         for(int j=0;j<ctx_blks_a;j++){
//             if(i >= j){
//                 nt_lut[count].x = i;
//                 nt_lut[count].y = j;
//                 count++;
//             }
//         }
//     }

// }

// void generate_tn_lut(uint2 *tn_lut, uint blocks, uint ctx_blks_b_){

//     int offset_lut = 10;
//     int sum = 10;
//     int count = 0;
//     for(int i=0;i<ctx_blks_b_;i++){
        
//         tn_lut[i].x = sum;  
//         tn_lut[i].y = offset_lut;
//         sum+=offset_lut;
//         offset_lut--;

//     }

//     for(int col = 0;col<ctx_blks_b_;col++){
//         for(int row = col;row<ctx_blks_b_;row++){

//             int bid = 0;
//             for(int k = row; k>0; k--){
//                 bid+=k;
//             }
//             tn_lut[ctx_blks_b_+count].x = bid+col;
//             tn_lut[ctx_blks_b_+count].y = row;
//             count++;
//         }
//     }
// }

// void generate_nn_lut(uint2 *nn_lut, uint blokcs, uint ctx_blks_b_){

//     int offset_lut = 1;
//     int sum = 10;
//     for(int i=0;i<ctx_blks_b_;i++){
        
//         nn_lut[i].x = sum;
//         nn_lut[i].y = offset_lut;
//         sum+=offset_lut;
//         offset_lut++;
        
//     }

//     int count = 0;
//     for(int row = 0; row<ctx_blks_b_; row++){
//         for(int col = 0; col<=row; col++){

//             nn_lut[ctx_blks_b_+count].x = count;
//             nn_lut[ctx_blks_b_+count].y = col;
//             count++;
//         }
//     }

// }

// void generate_sm_mask(uint *mask, uint blocks_, uint blk_size_, uint2 *nt_lut){
    
//     bool *mask_cpu = (bool*)malloc(sizeof(bool)*blocks_*blk_size_*blk_size_);
    
//     for(int l = 0;l<blocks_;l++){
//         if((nt_lut[l].x) == (nt_lut[l].y)){
//             for(int i=0;i<blk_size_;i++){
//                 for(int j=0;j<blk_size_;j++){
//                     if(i>=j){
//                         mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 1;
//                     }else{
//                         mask_cpu[l*(blk_size_*blk_size_)+i*blk_size_+j] = 0;
//                     }
//                 }
//             }
//         }else{
//             for(int i=0;i<(blk_size_*blk_size_);i++){
//                 mask_cpu[(l*blk_size_*blk_size_)+ i] = 1;
//             }
//         }
        
//     }

//     unsigned int *mask_np = (unsigned int*)malloc(sizeof(unsigned int)*blocks_*blk_size_);
//     for(int i=0;i<(blocks_*blk_size_);i++){
//         mask_np[i] = 0;
//         for(int j=0;j<32;j++){
//             bool keep = mask_cpu[i*blk_size_+j];
//             unsigned int temp;
//             temp = keep << (j);
//             mask_np[i] = mask_np[i] | temp;
            
//         }
//     }

//     for(int i=0; i<blocks_; i++){
//         for(int j=0; j<blk_size_; j++){
//             mask[j*blocks_+i] = mask_np[i*blk_size_+j];
//         }
//     }

//     free(mask_cpu);
//     free(mask_np);

// }