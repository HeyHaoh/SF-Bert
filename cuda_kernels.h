#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <sys/time.h>
#include <common.h>
#include "ew_op_gpu.h"

#define FINAL_MASK 0xffffffff
#define PI 3.141592654
#define URAND_SCALE 2.3283064365386962891e-10f
 
// __device__ __forceinline__ float _exp_approx(float x)
// {
//     x *= 1.4426950408889634f;
//     asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
//     return x;
// }

// __device__ __forceinline__ float _rcp_approx(float x)
// {
//     asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(x) :);
//     return x;
// }

// __device__ __forceinline__ float _rsqrt_approx(float x)
// {
//     asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(x) :);
//     return x;
// }

// __device__ __forceinline__ float _log_approx(float x)
// {
//     asm("lg2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
//     x *= 0.6931471824645996f;
//     return x;
// }

__device__ VALUE_TYPE warpReduceSum(VALUE_TYPE val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
} 

__device__ double warpReduceSum_d(double val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
} 

__device__ VALUE_TYPE warpReduceMax(VALUE_TYPE val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
} 

__device__ double warpReduceMax_d(double val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

__device__ VALUE_TYPE blockReduceSum(VALUE_TYPE val){
    
    static __shared__ VALUE_TYPE shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum(val);

    __syncthreads();

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : VALUE_TYPE(0.0f);
    val = warpReduceSum(val);

    __syncthreads();
                                
    return val;

}

__device__ double blockReduceSum_d(double val){
    
    static __shared__ double shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum_d(val);

    __syncthreads();

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : VALUE_TYPE(0.0f);
    val = warpReduceSum_d(val);

    __syncthreads();
                                
    return val;

}

__device__ VALUE_TYPE blockReduceMax(VALUE_TYPE val){

    static __shared__ VALUE_TYPE shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
    
}

__device__ double blockReduceMax_d(VALUE_TYPE val){

    static __shared__ double shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax_d(val);

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
    val = warpReduceMax_d(val);

    return val;
    
}



template<typename T>
__global__
void transpose_0123to0213_gpu(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
        + seq_id * size_per_head + threadIdx.x] = src[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
        + head_id * size_per_head + threadIdx.x];
}

template<typename T>
__global__
void transpose_0213to0123_gpu(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;

    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
        + head_id * size_per_head + threadIdx.x] = src[batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
        + seq_id * size_per_head + threadIdx.x];
}

template<typename T>
__global__
void transpose_0123to0132_gpu(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int id_in_batch_head = blockIdx.x / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int tid = threadIdx.x;

    dst[id_in_batch_head * (seq_len * size_per_head) + seq_id * size_per_head + tid] = src[id_in_batch_head * (seq_len * size_per_head) + tid * seq_len + seq_id];
}

//bid = 320;
//tid = 512;
__global__ void add_grad_gpu(VALUE_TYPE *out, VALUE_TYPE *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for(int i=0;i<batch_size;i++){
        VALUE_TYPE temp = 0.0;
        temp = input[i*(seq_len*state)+bid*state+tid];
        out[bid*state+tid] += temp; 
    }
}


__global__ void embedding_lookup_gpu(VALUE_TYPE *out, VALUE_TYPE *input, int *label, const int batch_size, const int seq_len, const int state, const int vocab_size){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    VALUE_TYPE temp = 0.0;
    int idx = 0;
    idx = label[bid];
    temp = input[idx * state + tid];
    out[bid*state + tid] = temp;
    
}

__global__ void softmax_gpu(VALUE_TYPE *output, VALUE_TYPE *input, VALUE_TYPE scaler, const int batch_size, const int head_num, const int seq_len){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ VALUE_TYPE sum;
    __shared__ VALUE_TYPE max;
    int offset = 0;

    for(int i=0;i<seq_len;i++){
        
        VALUE_TYPE temp = 0.0;
        // VALUE_TYPE temp_exp = 0.0;

        if(threadIdx.x <= i){
            temp = input[bid*seq_len*seq_len+i*seq_len+tid];
        }else{
            temp = -1e20f;
        }
        
        VALUE_TYPE max_val = 0.0;
        max_val = blockReduceMax(temp);

        if(threadIdx.x == 0){
            max = max_val;
        }
        __syncthreads();

        if(threadIdx.x <= i){

            // temp_exp = _exp_approx((temp-max)*scaler);
            temp = _exp_approx((temp-max)*scaler);
            
        }else{

            temp = 0.0;
        }

        VALUE_TYPE sum_val = 0.0;

        sum_val = blockReduceSum(temp);

        if(threadIdx.x == 0){
            sum = _rcp_approx(sum_val + 1e-6f);
        }
        __syncthreads(); 


        if(threadIdx.x <= i){
            output[(bid*seq_len+offset)*seq_len+tid] = temp*sum ;
        }
        offset++;
    }
}


__global__ void softmax_gpu_with_mask(VALUE_TYPE *output, VALUE_TYPE *input, int *mask, VALUE_TYPE scaler, const int batch_size, const int head_num, const int seq_len){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ VALUE_TYPE sum;
    __shared__ VALUE_TYPE max;

    int offset = 0;

    for(int i=0;i<seq_len;i++){
        
        VALUE_TYPE temp = 0.0;

        if(mask[i*seq_len+tid] != 0){
            temp = input[bid*seq_len*seq_len+i*seq_len+tid];
        }else{
            temp = -1e20f;
        }
        
        VALUE_TYPE max_val = 0.0;
        max_val = blockReduceMax(temp);

        if(threadIdx.x == 0){
            max = max_val;
        }
        __syncthreads();

        if(mask[i*seq_len+tid] != 0){

            temp = _exp_approx((temp-max)*scaler);
            
        }else{

            temp = 0.0;
        }

        VALUE_TYPE sum_val = 0.0;

        sum_val = blockReduceSum(temp);

        if(threadIdx.x == 0){
            sum = _rcp_approx(sum_val + 1e-6f);
        }
        __syncthreads(); 


        if(mask[i*seq_len+tid] != 0){
            output[(bid*seq_len+offset)*seq_len+tid] = temp*sum ;
        }
        offset++;
    }

}

__global__ void layernorm_gpu(VALUE_TYPE* out, VALUE_TYPE* mean, VALUE_TYPE* rstd, const VALUE_TYPE* input, const VALUE_TYPE* gain, const VALUE_TYPE* bias, const int batch_size, const int seq_len, const int state){
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ VALUE_TYPE shared_mean[1], shared_s_mean[1], shared_rstd[1];

    VALUE_TYPE tmp = input[bid*state+tid];
    VALUE_TYPE tmp_sqr = input[bid*state+tid]*input[bid*state+tid];

    VALUE_TYPE sum_val = 0.0;
    sum_val = blockReduceSum(tmp);
    __syncthreads();

    VALUE_TYPE sum_sqr_val = 0.0;
    sum_sqr_val = blockReduceSum(tmp_sqr);
    __syncthreads();

    if(threadIdx.x == 0){
        shared_mean[0] = sum_val / state;
        shared_s_mean[0] = sum_sqr_val / state;
    }
    __syncthreads();

    mean[bid] = shared_mean[0];

    __syncthreads();

    if(threadIdx.x == 0){
        shared_rstd[0] = _rsqrt_approx(shared_s_mean[0] - (shared_mean[0] * shared_mean[0]) + 1e-5f);
    }
    __syncthreads();

    rstd[bid] = shared_rstd[0];

    __syncthreads();

    out[bid*state+tid] = (input[bid*state+tid]-shared_mean[0]) * shared_rstd[0] * gain[tid] + bias[tid]; 

}

__global__ void softmax_cross_entropy_with_logits_gpu(VALUE_TYPE *output, VALUE_TYPE *loss, VALUE_TYPE *input, int *label, const int batch_size, const int seq_len, const int vocab_size){

    __shared__ VALUE_TYPE sum[1], sh_loss[1], max[1];

    VALUE_TYPE temp =0.0;
    VALUE_TYPE exp_input = 0.0;

    temp = input[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)];

    VALUE_TYPE max_val = 0.0;
    max_val = blockReduceMax(temp);

    if(threadIdx.x == 0){
        max[0] = max_val;
    }
    __syncthreads();

    temp -= max[0];
    temp = _exp_approx(temp);
    exp_input = temp;
    
    VALUE_TYPE sum_val = 0.0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum[0] = sum_val;
    }
    __syncthreads();

    temp = exp_input / sum[0];


    output[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)] = (temp - label[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)]);

    VALUE_TYPE log_out = 0.0;
    log_out = -((_log_approx(temp)) * (label[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)]));

    VALUE_TYPE loss_val = 0.0;
    loss_val = blockReduceSum(log_out);

    if(threadIdx.x == 0){
        sh_loss[0] = loss_val;
    }
    __syncthreads();

    loss[(int)blockIdx.x] = sh_loss[0];
}

__global__ void loss_add(VALUE_TYPE* loss_out, VALUE_TYPE* loss_input, const int size){

    __shared__ VALUE_TYPE loss;
    int group  = 0;
    group = size / 512;

    int offset = 0;
    VALUE_TYPE sum_of_group = 0.0;

    for(int i=0;i<group;i++){

        VALUE_TYPE temp = loss_input[((int)threadIdx.x) + offset];
        sum_of_group += temp;
    }

    __syncthreads();

    VALUE_TYPE loss_val = 0.0;
    loss_val = blockReduceSum(sum_of_group);
    if(threadIdx.x == 0){
        loss = loss_val;
    }
    __syncthreads();

    loss_out[0] = loss/size; 
}

__global__ void tensor_add_tensor_gpu(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *B, const int dim0, const int dim1, const int dim2){

    int size = dim2 * dim1 * dim0;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if((bid*512+tid) < size){
        C[bid*512+tid] =  (double)A[bid*512+tid] + (double)B[bid*512+tid];
    }
}

__global__ void tensor_add_vector_gpu(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    VALUE_TYPE bias_value = bias[tid];
    if((bid*dim2+tid) < size)
    {
        C[bid*blockDim.x+tid] = A[bid*blockDim.x+tid] + bias_value;
    }
}

__global__ void tensor_add_vector_gpu_2048(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    int group = 4;
    int offset = 0;
    #pragma unroll 4
    for(int i=0;i<group;i++)
    {   
        VALUE_TYPE bias_value = bias[tid + offset];
        if((bid*2048+tid+offset) < size)
        {
            C[bid*2048+tid+offset] = A[bid*2048+tid+offset] + bias_value;
        }
        offset += 512;
    }

}

__global__ void tensor_add_vector_gpu_2048_v2(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int group = 2048;
    int row = bid*blockDim.x+tid;
    // #pragma unroll
    for(int i=0;i<group;i++){
        C[row*2048+i] = A[row*2048+i] + bias[i];
    }

}

__global__ void tensor_add_vector_gpu_2048_v3(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    int group = dim2/blockDim.x;
    int offset = 0;

    for(int i=0;i<group;i++)
    {   
        VALUE_TYPE bias_value = bias[tid + offset];
        if((bid*2048+tid+offset) < size)
        {
            C[bid*2048+tid+offset] = A[bid*2048+tid+offset] + bias_value;
        }
        offset += blockDim.x;
    }

}



__global__ void tensor_add_matrix_gpu(VALUE_TYPE *C, VALUE_TYPE *A, VALUE_TYPE *B, const int dim0, const int dim1, const int dim2){

    int size = dim0 * dim1 * dim2;
    if(((int)blockIdx.x * 512) + ((int)threadIdx.x) < size){
        int index = ((int)blockIdx.x % dim1) * 512 + (int)threadIdx.x;
        C[(((int)blockIdx.x * 512) + (int)threadIdx.x)] = A[(((int)blockIdx.x * 512) + (int)threadIdx.x)] 
                                                        + B[index];
    }

}

__global__ void gelu_gpu(VALUE_TYPE *C, VALUE_TYPE *A, const int dim0, const int dim1, const int dim2)
{
    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;


    for(int i=0; i<group; i++){

        if(((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset)) < size){
            
            VALUE_TYPE a_value = A[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))];
            VALUE_TYPE cdx = 0.0;
            cdx = 1.0 / ((__expf(-1.702 * a_value)) + 1.0);
            C[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))] = cdx * a_value;

        }

        offset+=512;
    }
}



__global__ void gelu_gpu_v2(VALUE_TYPE *C, VALUE_TYPE *A, const int dim0, const int dim1, const int dim2){

    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;

    for(int i=0; i<group; i++){

        if(((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset)) < size){
            
            VALUE_TYPE a_value = A[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))];
            VALUE_TYPE cdx = 0.0;
            cdx = _rcp_approx((_exp_approx(-1.702 * a_value)) + 1.0);
            C[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))] = cdx * a_value;

        }

        offset+=512;
    }

}

__global__ void GendropoutMask(int *Entropy, unsigned int *Mask, const VALUE_TYPE keep_probe, const int mask_size){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid*blockDim.x + tid;

    int uint_group = idx/32;

    if(uint_group < mask_size){

        int lfsr0 = Entropy[idx * 0 + idx];
        int lfsr1 = Entropy[idx * 1 + idx];
        int lfsr2 = Entropy[idx * 2 + idx];

        lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
        lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
        lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);

        unsigned int urand = lfsr0 ^ lfsr1 ^ lfsr2;
        bool keep  = (VALUE_TYPE)urand * URAND_SCALE < keep_probe;
        unsigned int mask  = __ballot_sync(0xffffffff, keep);

        __syncthreads();

        if ((tid & 31) == 0){
            Mask[uint_group] = mask;
        }
        __syncthreads();

        Entropy[idx * 0 + idx] = lfsr0;
        Entropy[idx * 1 + idx] = lfsr1;
        Entropy[idx * 2 + idx] = lfsr2;

    }
}

__global__ void ApplydropoutMask(VALUE_TYPE *out, VALUE_TYPE *input, unsigned int *mask, const VALUE_TYPE keep_probe, const int mask_size){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = 0;

    if((bid*blockDim.x+tid) < mask_size){
        unsigned int mask_val = mask[bid*blockDim.x+tid];
        for(int i=0;i<32;i++)
        {   
            if(((1 << i) & mask_val) == 0)  
            {   
                out[(bid+offset)*blockDim.x+tid] = 0.0;

            }else{

                out[(bid+offset)*blockDim.x+tid] = input[(bid+offset)*blockDim.x+tid] * (1.0 / keep_probe);
            }

            offset+=320;

        }
    }    
}

__global__ void ApplydropoutMask_SmallSize(VALUE_TYPE *out, VALUE_TYPE *input, unsigned int *mask, const VALUE_TYPE keep_probe, const int mask_size){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = 0;

    if((bid*blockDim.x+tid) < mask_size){
        unsigned int mask_val = mask[bid*blockDim.x+tid];

        for(int i=0;i<32;i++)
        {   
            if(((1 << i) & mask_val) == 0)  
            {   
                out[(bid+offset)*blockDim.x+tid] = 0.0;

            }else{

                out[(bid+offset)*blockDim.x+tid] = input[(bid+offset)*blockDim.x+tid] * (1.0 /keep_probe);
            }

            offset+=10;

        }
    }    
}

__global__ void embedding_lookup_grad_gpu(VALUE_TYPE *out, VALUE_TYPE *dy, int *label, const int batch_size, const int seq_len, const int state, const int vocab_size){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = 0;
    idx = label[bid];

    VALUE_TYPE temp = 0.0;
    temp = dy[bid*state+tid];

    atomicAdd(out+(idx*state+tid), temp);
}

__global__ void bias_grad_db_gpu(VALUE_TYPE *out, VALUE_TYPE *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid * batch_size;
    int col = tid;

    VALUE_TYPE sum_of_temp = 0.0;
    for(int i=0;i<batch_size;i++){
        sum_of_temp+=input[row*state+col];
        row++;
    }
    atomicAdd(out+tid,sum_of_temp);
}

__global__ void bias_grad_db_2048_gpu(VALUE_TYPE *out, VALUE_TYPE *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid * batch_size;
    VALUE_TYPE sum_of_temp ;
    for(int col=tid;col<state;col+=512){
        sum_of_temp = 0.0;
        row = bid * batch_size;
        for(int i=0;i<batch_size;i++){
            sum_of_temp+=input[row*state+col];
            row++;
        }
        atomicAdd(out+col,sum_of_temp);
    }

}

__global__ void gelu_grad_gpu(VALUE_TYPE *output, VALUE_TYPE *input, VALUE_TYPE *x, VALUE_TYPE *b, const int dim0, const int dim1, const int dim2){

    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;
    int bid = blockIdx.x;
    int tid = threadIdx.x; 

    for(int i=0; i<group; i++){

        if(((bid * dim2) + (tid + offset)) < size){
            
            VALUE_TYPE temp = 0.0;
            temp = input[bid*dim2+tid+offset];
            VALUE_TYPE x_val = 0.0;
            x_val = x[bid*dim2+tid+offset];
            VALUE_TYPE b_val = 0.0;
            b_val = b[tid+offset];
            VALUE_TYPE sig = _exp_approx(-1.702 * (x_val + b_val)) + 1.0;
            sig = 1.0/sig;
            VALUE_TYPE sig_sqr = 0.0;
            sig_sqr = sig*sig;
            output[bid*dim2+tid+offset] = (temp*sig) + (1.702 * (x_val+b_val) * temp * (sig - sig_sqr));
        }

        offset+=512;
    }

}

__global__ void cross_entropy_grad_gpu(VALUE_TYPE *output, VALUE_TYPE *input, const int batch_size, const int seq_len, const int vocab_size){
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    output[bid * blockDim.x + tid]=input[bid * blockDim.x + tid] / (batch_size*seq_len);
       
}

__global__ void adam_apply_gradients_gpu(VALUE_TYPE *d_C, VALUE_TYPE* d_dw, VALUE_TYPE* d_mean, VALUE_TYPE* d_var, double beta1, double beta2, double beta_power1, double beta_power2, double learning_rate, double epsilon, VALUE_TYPE norm_scale, VALUE_TYPE grad_scale, VALUE_TYPE clip_sigma, const int size){
    
    
    if(norm_scale !=0)
    {
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        int index= bid*blockDim.x+tid;
        
        if(index<size)
        {
            VALUE_TYPE dw_now=d_dw[index];
            dw_now *= grad_scale * norm_scale;
            d_var[index] = beta2 * d_var[index] + ((1.0f - beta2) * dw_now * dw_now);
            VALUE_TYPE sigma = sqrt(d_var[index]);
            d_mean[index] = beta1 * d_mean[index] + (1.0f - beta1) * dw_now;
            d_C[index] -=  learning_rate * d_mean[index] *_rcp_approx(sigma + epsilon);
            d_dw[index]=dw_now;
        }   
    }
       
}


__global__ void layernorm_dg_db_gpu(VALUE_TYPE *dg, VALUE_TYPE *db, VALUE_TYPE *dy, VALUE_TYPE *x, VALUE_TYPE *mean, VALUE_TYPE *rstd, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = batch_size*seq_len*state;


    VALUE_TYPE sum_of_dg = 0.0;
    VALUE_TYPE sum_of_db = 0.0;
    for(int i=0;i<batch_size;i++){
        VALUE_TYPE temp = 0.0;
        temp = (x[(i*seq_len+bid)*state+tid] - mean[i*seq_len+bid]) * rstd[i*seq_len+bid];
        VALUE_TYPE dy_val = dy[(i*seq_len+bid)*state+tid];
        sum_of_dg += (dy_val * temp); 
        sum_of_db += dy_val;
    }
    atomicAdd(dg+tid, sum_of_dg);
    atomicAdd(db+tid, sum_of_db);
    

}

__global__ void layernorm_grad_dx_gpu(VALUE_TYPE* dx, VALUE_TYPE* x, VALUE_TYPE* mean, VALUE_TYPE* rstd, VALUE_TYPE* dy, VALUE_TYPE* gamma, const int batch_size, const int seq_len, const int state){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int size = batch_size * seq_len * state;
    __shared__ VALUE_TYPE sum1, sum2;

    if((bid*blockDim.x + tid) < size){

        VALUE_TYPE temp = 0.0;
        temp = (x[bid*blockDim.x + tid] - mean[bid]) * rstd[bid];
        VALUE_TYPE dy_val = 0.0;
        dy_val = dy[bid*blockDim.x + tid];
        dy_val *= gamma[tid];
        VALUE_TYPE sum1_temp = temp * dy_val;
        
        VALUE_TYPE sum1_val = 0.0;
        VALUE_TYPE sum2_val = 0.0;

        sum1_val = blockReduceSum(sum1_temp);
        if(threadIdx.x == 0){
            sum1 = sum1_val;
        }

        __syncthreads();

        sum2_val = blockReduceSum(dy_val);
        if(threadIdx.x == 0){
            sum2 = sum2_val;
        }
        __syncthreads();

        dx[bid*blockDim.x + tid] = (dy_val - ((temp * sum1 + sum2) * (1.0 / state))) * rstd[bid];

    }
}

__global__ void softmax_grad_gpu(VALUE_TYPE *out, VALUE_TYPE *dy, VALUE_TYPE *y, VALUE_TYPE scaler, const int batch_size, const int head_num, const int seq_len){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = batch_size*head_num*seq_len*seq_len;
    __shared__ VALUE_TYPE sum_dyy;

    for(int i=0;i<seq_len;i++)
    {
        if((bid*seq_len*seq_len+i*seq_len+tid) < size){
            VALUE_TYPE temp = 0.0;
            VALUE_TYPE temp_y = 0.0;
            temp = dy[bid*seq_len*seq_len+i*seq_len+tid];
            temp_y = y[bid*seq_len*seq_len+i*seq_len+tid];

            VALUE_TYPE temp_dyy = 0.0;
            temp_dyy = temp * temp_y;

            VALUE_TYPE sum_dyy_val = 0.0;
            sum_dyy_val = blockReduceSum(temp_dyy);
            if(threadIdx.x == 0){
                sum_dyy = sum_dyy_val;
            }

            __syncthreads();
            out[bid*seq_len*seq_len+i*seq_len+tid] = (temp - sum_dyy)*temp_y*scaler;
        }   

    }
        
}

__global__ void gradients_add_gpu_512(VALUE_TYPE *gradients_sum, VALUE_TYPE *gradients, int size){

    __shared__ VALUE_TYPE sum;

    // int bid = blockIdx.x;
    int tid = threadIdx.x;

    VALUE_TYPE temp = 0.0;
    temp = gradients[tid];
    temp *= temp;

    VALUE_TYPE sum_val =0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0)
        sum = sum_val;

    __syncthreads();
    
    gradients_sum[0] = sum;

}

__global__ void gradients_add_gpu_2048(VALUE_TYPE *gradients_sum, VALUE_TYPE *gradients, int size){

    __shared__ VALUE_TYPE sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    VALUE_TYPE temp = gradients[bid*512+tid];
    temp *= temp;

    VALUE_TYPE sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}


__global__ void gradients_add_gpu_512_2048(VALUE_TYPE *gradients_sum, VALUE_TYPE *gradients, int size){
        
    __shared__ VALUE_TYPE sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    VALUE_TYPE temp = gradients[bid*512+tid];
    temp *= temp;

    VALUE_TYPE sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}


__global__ void gradients_add_gpu_512_512(VALUE_TYPE *gradients_sum, VALUE_TYPE *gradients, int size){
        
    __shared__ VALUE_TYPE sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    VALUE_TYPE temp = gradients[bid*512+tid];
    temp *= temp;

    VALUE_TYPE sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}

__global__ void tensor_add_tensor_add_tensor_gpu(VALUE_TYPE *D, VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int index = bid*blockDim.x+tid;
    int size = dim0*dim1*dim2;
    if((bid*blockDim.x+tid) < size){
        D[index] =  A[index]+B[index]+C[index];
    }
    
}


__global__ void csr_sddmm(const int *RowPtrM, const int *ColidxM,
    int *RowptrC, int *ColidxC, VALUE_TYPE *ValC, 
    const double alpha, VALUE_TYPE *A, const double beta, VALUE_TYPE *B,
    const int seq_len, const int size_per_head){
    

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int Colidx;
    VALUE_TYPE sum = 0;

    int offset_matrix_rowptr = (seq_len)*(bid);
    int offset_matrix_colidx = (RowPtrM[seq_len-1])*(bid);

    for(int i=RowPtrM[offset_matrix_rowptr+tid];i<RowPtrM[offset_matrix_rowptr+tid+1];i++){
        
        Colidx = ColidxM[offset_matrix_colidx + i];
        sum = 0.0;
        for(int p=0; p<size_per_head; p++){

            sum += A[tid*size_per_head + p] * B[Colidx*size_per_head + p];
            
        } 
        ValC[offset_matrix_colidx+i] =  sum;
        ColidxC[offset_matrix_colidx+i] = Colidx;
    }

    RowptrC[offset_matrix_rowptr+tid+1] = RowPtrM[offset_matrix_rowptr+tid+1];
}


__global__ void csr_softmax(const int *RowptrC, const int *ColidxC,  VALUE_TYPE *ValC, const int batch_size, const int seq_len, const int head_num, const float scaler){
    

    int tid = threadIdx.x;
    int bid = blockIdx.x;


    int offset_matrix_rowptr = (seq_len)*(bid);
    int offset_matrix_colidx = (RowptrC[seq_len-1])*(bid);

    VALUE_TYPE temp, max = 1e-20f, sum = 0.0;

    for(int i=RowptrC[offset_matrix_rowptr+tid];i<RowptrC[offset_matrix_rowptr+tid+1];i++){
        temp = ValC[offset_matrix_colidx + i];
        if(temp > max)
            max = temp;
    }

    for(int i=RowptrC[offset_matrix_rowptr+tid];i<RowptrC[offset_matrix_rowptr+tid+1];i++){
        ValC[offset_matrix_colidx + i] = exp((ValC[offset_matrix_colidx + i] - max)*scaler);
        sum += ValC[offset_matrix_colidx + i];
         
    }
    

    for(int i=RowptrC[offset_matrix_rowptr+tid];i<RowptrC[offset_matrix_rowptr+tid+1];i++){
        ValC[offset_matrix_colidx + i] /= (sum + 1e-6f);
    }

}


__global__ void csr_spmm(const int *RowptrA, const int *ColidxA, const VALUE_TYPE *ValA, VALUE_TYPE *B, VALUE_TYPE *C, const int seq_len, const int size_per_head){
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;


    int offset_matrix_rowptr = (seq_len)*(bid);
    int offset_matrix_colidx = (RowptrA[seq_len-1])*(bid);

    int offset_dense = bid*(seq_len*size_per_head);

    for(int i=0;i<size_per_head;i++){
        for(int l=RowptrA[offset_matrix_rowptr+tid];l<RowptrA[offset_matrix_rowptr+tid+1];l++){
            C[offset_dense+tid*size_per_head+i] += ValA[offset_matrix_colidx+l] * B[offset_dense+ColidxA[offset_matrix_colidx+l]*size_per_head+i]; 
        }
    }
}

__global__ void fused_attention(const int *RowptrM, const int *ColidxM, float *ValM, float *q, float *k, float* v, float*sv, 
                                const int batch_size, const int head_num, const int seq_len, const int size_per_head, const float scaler){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int Colidx;
    VALUE_TYPE sum = 0;

    int offset_matrix_rowptr = (seq_len)*(bid);
    int offset_matrix_colidx = (RowptrM[seq_len-1])*(bid);
    VALUE_TYPE max = 1e-20f, sum_of_softmax = 0.0;
    int offset_dense = bid*(seq_len*size_per_head);

    int start = RowptrM[offset_matrix_rowptr+tid];
    int end = RowptrM[offset_matrix_rowptr+tid+1];

    for(int i=start;i<end;i++){
        
        Colidx = ColidxM[offset_matrix_colidx + i];
        sum = 0.0;
        for(int p=0; p<size_per_head; p++){

            sum += q[tid*size_per_head + p] * k[Colidx*size_per_head + p];
            
        }
        ValM[offset_matrix_colidx + i] = sum; 

        if(sum > max){
            max = sum;
        }
    }

    for(int i=start;i<end;i++){
        ValM[offset_matrix_colidx + i] = exp((ValM[offset_matrix_colidx + i] - max)*scaler);
        sum_of_softmax += ValM[offset_matrix_colidx + i];
         
    }
    
    for(int i=start;i<end;i++){
        ValM[offset_matrix_colidx + i] /= (sum_of_softmax + 1e-6f);
    }
    

    for(int i=0;i<size_per_head;i++){
        for(int l=RowptrM[offset_matrix_rowptr+tid];l<RowptrM[offset_matrix_rowptr+tid+1];l++){
            sv[offset_dense+tid*size_per_head+i] += ValM[offset_matrix_colidx+l] * v[offset_dense+ColidxM[offset_matrix_colidx+l]*size_per_head+i]; 
        }
    }

}


__global__ void sddmm_v2(const int *ColidxM, float *q, float *k, float *qk, const int batch_size, const int head_num, const int seq_len, const int size_per_head){

    int bid  = blockIdx.x;
    int tid = threadIdx.x;

    int matrix_num = batch_size*head_num;
    int offset_matirx_dense_size = seq_len*seq_len;


    for(int i=0;i<matrix_num;i++){

        VALUE_TYPE sum = 0.0;
        int temp = 0;

        temp = ColidxM[bid*seq_len+tid+(i*offset_matirx_dense_size)];

        if(temp != -1){

            for(int j=0;j<size_per_head;j++){

                sum += q[bid*size_per_head+j] * k[tid*size_per_head+j];
            }

            qk[i*offset_matirx_dense_size + bid*seq_len+tid] = sum;

        }
    }

}

__global__ void softmax_v2(const int *ColidxM, float *softmax, float *qk, const int batch_size, const int head_num, const int seq_len, const float scaler){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ VALUE_TYPE sum_softmax;
    __shared__ VALUE_TYPE max;

    int matrix_num = batch_size*head_num;

    int offset_matirx_dense_size = seq_len*seq_len;

    for(int i=0;i<matrix_num;i++){

        int temp = 0;
        VALUE_TYPE res =0.0, sum = 0.0;
        VALUE_TYPE max_temp = 0.0, sum_temp = 0.0;

        temp = ColidxM[bid*seq_len+tid+(i*offset_matirx_dense_size)];

        if(temp == -1){
            sum = 1e-20f;
            res = 0.0;
        }else{
            sum = qk[i*offset_matirx_dense_size+bid*seq_len+tid];
        }

        max_temp = blockReduceMax(sum);

        if(tid == 0){

            max = max_temp;
        }

        __syncthreads();

        if(temp != -1){
            res = _exp_approx((sum-max) * scaler);
        }

        sum_temp = blockReduceSum(res);

        if(tid == 0){
            sum_softmax = _rcp_approx(sum_temp + 1e-6f);
        }

        __syncthreads();

        res *= sum_softmax;

        softmax[i*offset_matirx_dense_size+bid*seq_len+tid] = res;
    }


}

__global__ void spmm_v2(const int *ColidxM, float *softmax, float *v, float *sv, const int batch_size, const int head_num, const int seq_len, const int size_per_head){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
        
    __shared__ VALUE_TYPE sv_out;

    int matrix_num = batch_size*head_num;
    int offset_matirx_dense_size = seq_len*seq_len;
    int offset_matrix_dense_size_128 = seq_len*size_per_head;



    for(int i = 0;i<matrix_num;i++){

        int temp = 0;
        VALUE_TYPE sv_temp = 0.0;
        VALUE_TYPE sv_sum = 0.0;

        temp = ColidxM[bid*seq_len+tid+(i*offset_matirx_dense_size)];
        for(int j=0;j<size_per_head;j++){
            if(temp != -1){
                sv_temp = softmax[i*offset_matirx_dense_size+bid*seq_len+tid] * v[temp*size_per_head+j+(i*offset_matrix_dense_size_128)];

            }

            sv_sum = blockReduceSum(sv_temp);

            if(tid == 0){
                sv_out = sv_sum;
            }

            __syncthreads();

            sv[bid*size_per_head+j+(i*offset_matrix_dense_size_128)] = sv_out;

        }
    }
}

//nblock=320, nthreads=320
__global__ void fused_attention_v2(const int *ColidxM, float *q, float *k, float* v, float*sv, 
                                const int batch_size, const int head_num, const int seq_len, const int size_per_head, const float scaler){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ VALUE_TYPE sum_softmax;
    __shared__ VALUE_TYPE max;
    __shared__ VALUE_TYPE sv_out;

    int matrix_num = batch_size*head_num;

    int offset_matirx_dense_size = seq_len*seq_len;
    int offset_matrix_dense_size_128 = seq_len*size_per_head;

    for(int i=0;i<matrix_num;i++){

        VALUE_TYPE sum = 0, res = 0;
        int temp = 0;
        VALUE_TYPE max_temp = 0, sum_temp = 0;
        VALUE_TYPE sv_temp = 0, sv_sum=0;

        temp = ColidxM[bid*seq_len+tid+(i*offset_matirx_dense_size)];

        if(temp == -1){
            sum = 1e-20f;
            res = 0.0;
            sv_temp = 0.0;
        }

        if(temp != -1){

            for(int j=0;j<size_per_head;j++){

                sum += q[bid*size_per_head+j] * k[tid*size_per_head+j];
            }

        }

        max_temp = blockReduceMax(sum);

        if(tid == 0){

            max = max_temp;
        }

        __syncthreads();

        if(temp != -1){
            res = _exp_approx((sum-max) * scaler);
        }

        sum_temp = blockReduceSum(res);

        if(tid == 0){
            sum_softmax = _rcp_approx(sum_temp + 1e-6f);
        }

        __syncthreads();

        res *= sum_softmax;

        

        for(int j=0;j<size_per_head;j++){
            if(temp != -1){
                sv_temp = res * v[temp*size_per_head+j+(i*offset_matrix_dense_size_128)];

            }

            sv_sum = blockReduceSum(sv_temp);

            if(tid == 0){
                sv_out = sv_sum;
            }

            __syncthreads();

            sv[bid*size_per_head+j+(i*offset_matrix_dense_size_128)] = sv_out;

        }

    }

}





void gpu_save_float(VALUE_TYPE *gpu_array, char *file_name, int size){

    VALUE_TYPE *temp = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE)*size);
    cudaMemcpy(temp, gpu_array, sizeof(VALUE_TYPE)*size, cudaMemcpyDeviceToHost);
    FILE *fp = NULL;

    strcat(file_name, ".txt");
    fp = fopen(file_name, "w");
    if(fp == NULL){
            printf("open error in %s !\n", file_name);
    }
    for(int j=0;j<size;j++){
        fprintf(fp, "%.25f ", temp[j]);
    }
    fclose(fp);
    free(temp);

}


void cpu_save_float(VALUE_TYPE *cpu_array, char *file_name, int size){
    
    FILE *fp = NULL;

    strcat(file_name, ".txt");
    fp = fopen(file_name, "w");
    if(fp == NULL){
            printf("open error in %s !\n", file_name);
    }
    for(int j=0;j<size;j++){
        fprintf(fp, "%.8f ", cpu_array[j]);
    }
    fclose(fp);


}



