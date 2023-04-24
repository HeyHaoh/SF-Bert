#include <stdio.h>
#include <cmath>

// gpu done
void embedding_lookup(float* C, float* A, int* B, const int batch_size, const int seq_len, const int head_num, const int size_per_head){
    for(int i=0;i<(batch_size*seq_len);i++){
        int num = B[i];
        for(int j=0;j<(head_num*size_per_head);j++){
            C[i*(head_num*size_per_head) + j] = A[num*(head_num*size_per_head) + j];
        }
    }
}

//gpu done
void embedding_lookup_grad(float* out, float* dy, int* B, const int batch_size, const int seq_len, const int vocab_size, const int state){
    
    for(int i=0;i<vocab_size*state;i++){
        out[i] = 0.0;
    }

    for(int i=0;i<batch_size;i++){
        for(int j=0;j<seq_len;j++){
            int num = B[(i*seq_len)+j];
            for(int k=0;k<state;k++){
                out[(num*state)+k] += dy[(i*seq_len*state)+(j*state)+k];

            }
        }
    }

}