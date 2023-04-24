#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <sys/time.h>

#define FINAL_MASK 0xffffffff
#define PI 3.141592654
#define URAND_SCALE 2.3283064365386962891e-10f

#define VALUE_TYPE float
 
__device__ __forceinline__ float _exp_approx(float x)
{
    x *= 1.4426950408889634f;
    asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}

__device__ __forceinline__ float _rcp_approx(float x)
{
    asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}

__device__ __forceinline__ float _rsqrt_approx(float x)
{
    asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    return x;
}

__device__ __forceinline__ float _log_approx(float x)
{
    asm("lg2.approx.ftz.f32 %0, %0;" : "+f"(x) :);
    x *= 0.6931471824645996f;
    return x;
}

__device__ float warpReduceSum(float val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
} 

__device__ double warpReduceSum_d(double val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
} 

__device__ float warpReduceMax(float val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
} 

__device__ double warpReduceMax_d(double val){

    for(int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

__device__ float blockReduceSum(float val){
    
    static __shared__ float shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum(val);

    __syncthreads();

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : float(0.0f);
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

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : float(0.0f);
    val = warpReduceSum_d(val);

    __syncthreads();
                                
    return val;

}

__device__ float blockReduceMax(float val){

    static __shared__ float shared[32];
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

__device__ double blockReduceMax_d(float val){

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
__global__ void add_grad_gpu(float *out, float *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for(int i=0;i<batch_size;i++){
        float temp = 0.0;
        temp = input[i*(seq_len*state)+bid*state+tid];
        out[bid*state+tid] += temp; 
    }
}


__global__ void embedding_lookup_gpu(float *out, float *input, int *label, const int batch_size, const int seq_len, const int state, const int vocab_size){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float temp = 0.0;
    int idx = 0;
    idx = label[bid];
    temp = input[idx * state + tid];
    out[bid*state + tid] = temp;
    
}

__global__ void softmax_gpu(float *output, float *input, float scaler, const int batch_size, const int head_num, const int seq_len){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float sum;
    __shared__ float max;
    int offset = 0;

    for(int i=0;i<seq_len;i++){
        
        float temp = 0.0;
        // float temp_exp = 0.0;

        if(threadIdx.x <= i){
            temp = input[bid*seq_len*seq_len+i*seq_len+tid];
        }else{
            temp = -1e20f;
        }
        
        float max_val = 0.0;
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

        float sum_val = 0.0;

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


__global__ void softmax_gpu_with_mask(float *output, float *input, int *mask, float scaler, const int batch_size, const int head_num, const int seq_len){

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float sum;
    __shared__ float max;

    int offset = 0;

    for(int i=0;i<seq_len;i++){
        
        float temp = 0.0;

        if(mask[i*seq_len+tid] != 0){
            temp = input[bid*seq_len*seq_len+i*seq_len+tid];
        }else{
            temp = -1e20f;
        }
        
        float max_val = 0.0;
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

        float sum_val = 0.0;

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

__global__ void layernorm_gpu(float* out, float* mean, float* rstd, const float* input, const float* gain, const float* bias, const int batch_size, const int seq_len, const int state){
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float shared_mean[1], shared_s_mean[1], shared_rstd[1];

    float tmp = input[bid*state+tid];
    float tmp_sqr = input[bid*state+tid]*input[bid*state+tid];

    float sum_val = 0.0;
    sum_val = blockReduceSum(tmp);
    __syncthreads();

    float sum_sqr_val = 0.0;
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

__global__ void softmax_cross_entropy_with_logits_gpu(float *output, float *loss, float *input, int *label, const int batch_size, const int seq_len, const int vocab_size){

    __shared__ float sum[1], sh_loss[1], max[1];

    float temp =0.0;
    float exp_input = 0.0;

    temp = input[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)];

    float max_val = 0.0;
    max_val = blockReduceMax(temp);

    if(threadIdx.x == 0){
        max[0] = max_val;
    }
    __syncthreads();

    temp -= max[0];
    temp = _exp_approx(temp);
    exp_input = temp;
    
    float sum_val = 0.0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum[0] = sum_val;
    }
    __syncthreads();

    temp = exp_input / sum[0];


    output[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)] = (temp - label[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)]);

    float log_out = 0.0;
    log_out = -((_log_approx(temp)) * (label[((int)blockIdx.x * vocab_size) + ((int)threadIdx.x)]));

    float loss_val = 0.0;
    loss_val = blockReduceSum(log_out);

    if(threadIdx.x == 0){
        sh_loss[0] = loss_val;
    }
    __syncthreads();

    loss[(int)blockIdx.x] = sh_loss[0];
}

__global__ void loss_add(float* loss_out, float* loss_input, const int size){

    __shared__ float loss;
    int group  = 0;
    group = size / 512;

    int offset = 0;
    float sum_of_group = 0.0;

    for(int i=0;i<group;i++){

        float temp = loss_input[((int)threadIdx.x) + offset];
        sum_of_group += temp;
    }

    __syncthreads();

    float loss_val = 0.0;
    loss_val = blockReduceSum(sum_of_group);
    if(threadIdx.x == 0){
        loss = loss_val;
    }
    __syncthreads();

    loss_out[0] = loss/size; 
}

__global__ void tensor_add_tensor_gpu(float *C, float *A, float *B, const int dim0, const int dim1, const int dim2){

    int size = dim2 * dim1 * dim0;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if((bid*512+tid) < size){
        C[bid*512+tid] =  (double)A[bid*512+tid] + (double)B[bid*512+tid];
    }
}

__global__ void tensor_add_vector_gpu(float *C, float *A, float *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    float bias_value = bias[tid];
    if((bid*dim2+tid) < size)
    {
        C[bid*blockDim.x+tid] = A[bid*blockDim.x+tid] + bias_value;
    }
}

__global__ void tensor_add_vector_gpu_2048(float *C, float *A, float *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    int group = 4;
    int offset = 0;
    #pragma unroll 4
    for(int i=0;i<group;i++)
    {   
        float bias_value = bias[tid + offset];
        if((bid*2048+tid+offset) < size)
        {
            C[bid*2048+tid+offset] = A[bid*2048+tid+offset] + bias_value;
        }
        offset += 512;
    }

}

__global__ void tensor_add_vector_gpu_2048_v2(float *C, float *A, float *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int group = 2048;
    int row = bid*blockDim.x+tid;
    // #pragma unroll
    for(int i=0;i<group;i++){
        C[row*2048+i] = A[row*2048+i] + bias[i];
    }

}

__global__ void tensor_add_vector_gpu_2048_v3(float *C, float *A, float *bias, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = dim0 * dim1 * dim2;
    int group = dim2/blockDim.x;
    int offset = 0;

    for(int i=0;i<group;i++)
    {   
        float bias_value = bias[tid + offset];
        if((bid*2048+tid+offset) < size)
        {
            C[bid*2048+tid+offset] = A[bid*2048+tid+offset] + bias_value;
        }
        offset += blockDim.x;
    }

}



__global__ void tensor_add_matrix_gpu(float *C, float *A, float *B, const int dim0, const int dim1, const int dim2){

    int size = dim0 * dim1 * dim2;
    if(((int)blockIdx.x * 512) + ((int)threadIdx.x) < size){
        int index = ((int)blockIdx.x % dim1) * 512 + (int)threadIdx.x;
        C[(((int)blockIdx.x * 512) + (int)threadIdx.x)] = A[(((int)blockIdx.x * 512) + (int)threadIdx.x)] 
                                                        + B[index];
    }

}

__global__ void gelu_gpu(float *C, float *A, const int dim0, const int dim1, const int dim2)
{
    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;


    for(int i=0; i<group; i++){

        if(((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset)) < size){
            
            float a_value = A[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))];
            float cdx = 0.0;
            cdx = 1.0 / ((__expf(-1.702 * a_value)) + 1.0);
            C[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))] = cdx * a_value;

        }

        offset+=512;
    }
}



__global__ void gelu_gpu_v2(float *C, float *A, const int dim0, const int dim1, const int dim2){

    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;

    for(int i=0; i<group; i++){

        if(((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset)) < size){
            
            float a_value = A[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))];
            float cdx = 0.0;
            cdx = _rcp_approx((_exp_approx(-1.702 * a_value)) + 1.0);
            C[((((int)blockIdx.x) * 2048) + ((int)threadIdx.x + offset))] = cdx * a_value;

        }

        offset+=512;
    }

}

__global__ void GendropoutMask(int *Entropy, unsigned int *Mask, const float keep_probe, const int mask_size){

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
        bool keep  = (float)urand * URAND_SCALE < keep_probe;
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

__global__ void ApplydropoutMask(float *out, float *input, unsigned int *mask, const float keep_probe, const int mask_size){

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

__global__ void ApplydropoutMask_SmallSize(float *out, float *input, unsigned int *mask, const float keep_probe, const int mask_size){

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

__global__ void embedding_lookup_grad_gpu(float *out, float *dy, int *label, const int batch_size, const int seq_len, const int state, const int vocab_size){

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = 0;
    idx = label[bid];

    float temp = 0.0;
    temp = dy[bid*state+tid];

    atomicAdd(out+(idx*state+tid), temp);
}

__global__ void bias_grad_db_gpu(float *out, float *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid * batch_size;
    int col = tid;

    float sum_of_temp = 0.0;
    for(int i=0;i<batch_size;i++){
        sum_of_temp+=input[row*state+col];
        row++;
    }
    atomicAdd(out+tid,sum_of_temp);
}

__global__ void bias_grad_db_2048_gpu(float *out, float *input, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid * batch_size;
    float sum_of_temp ;
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

__global__ void gelu_grad_gpu(float *output, float *input, float *x, float *b, const int dim0, const int dim1, const int dim2){

    int size = dim0*dim1*dim2;
    int group = dim2/512;
    int offset = 0;
    int bid = blockIdx.x;
    int tid = threadIdx.x; 

    for(int i=0; i<group; i++){

        if(((bid * dim2) + (tid + offset)) < size){
            
            float temp = 0.0;
            temp = input[bid*dim2+tid+offset];
            float x_val = 0.0;
            x_val = x[bid*dim2+tid+offset];
            float b_val = 0.0;
            b_val = b[tid+offset];
            float sig = _exp_approx(-1.702 * (x_val + b_val)) + 1.0;
            sig = 1.0/sig;
            float sig_sqr = 0.0;
            sig_sqr = sig*sig;
            output[bid*dim2+tid+offset] = (temp*sig) + (1.702 * (x_val+b_val) * temp * (sig - sig_sqr));
        }

        offset+=512;
    }

}

__global__ void cross_entropy_grad_gpu(float *output, float *input, const int batch_size, const int seq_len, const int vocab_size){
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    output[bid * blockDim.x + tid]=input[bid * blockDim.x + tid] / (batch_size*seq_len);
       
}

__global__ void adam_apply_gradients_gpu(float *d_C, float* d_dw, float* d_mean, float* d_var, double beta1, double beta2, double beta_power1, double beta_power2, double learning_rate, double epsilon, float norm_scale, float grad_scale, float clip_sigma, const int size){
    
    
    if(norm_scale !=0)
    {
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        int index= bid*blockDim.x+tid;
        
        if(index<size)
        {
            float dw_now=d_dw[index];
            dw_now *= grad_scale * norm_scale;
            d_var[index] = beta2 * d_var[index] + ((1.0f - beta2) * dw_now * dw_now);
            float sigma = sqrt(d_var[index]);
            d_mean[index] = beta1 * d_mean[index] + (1.0f - beta1) * dw_now;
            d_C[index] -=  learning_rate * d_mean[index] *_rcp_approx(sigma + epsilon);
            d_dw[index]=dw_now;
        }   
    }
       
}


__global__ void layernorm_dg_db_gpu(float *dg, float *db, float *dy, float *x, float *mean, float *rstd, const int batch_size, const int seq_len, const int state){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = batch_size*seq_len*state;
    int row = bid * batch_size;
    int col = tid;

    if((row*state+col) < size){
        float sum_of_dg = 0.0;
        float sum_of_db = 0.0;
        for(int i=0;i<batch_size;i++){
            float temp = 0.0;
            temp = (x[row*state+col] - mean[row]) * rstd[row];
            float dy_val = dy[row*state+col];
            sum_of_dg += dy_val * temp; 
            sum_of_db += dy_val;
            row++;
        }
        atomicAdd(dg+tid, sum_of_dg);
        atomicAdd(db+tid, sum_of_db);
    }

}

__global__ void layernorm_grad_dx_gpu(float* dx, float* x, float* mean, float* rstd, float* dy, float* gamma, const int batch_size, const int seq_len, const int state){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int size = batch_size * seq_len * state;
    __shared__ float sum1, sum2;

    if((bid*blockDim.x + tid) < size){

        float temp = 0.0;
        temp = (x[bid*blockDim.x + tid] - mean[bid]) * rstd[bid];
        float dy_val = 0.0;
        dy_val = dy[bid*blockDim.x + tid];
        dy_val *= gamma[tid];
        float sum1_temp = temp * dy_val;
        
        float sum1_val = 0.0;
        float sum2_val = 0.0;

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

__global__ void softmax_grad_gpu(float *out, float *dy, float *y, float scaler, const int batch_size, const int head_num, const int seq_len){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int size = batch_size*head_num*seq_len*seq_len;
    __shared__ float sum_dyy;

    for(int i=0;i<seq_len;i++)
    {
        if((bid*seq_len*seq_len+i*seq_len+tid) < size){
            float temp = 0.0;
            float temp_y = 0.0;
            temp = dy[bid*seq_len*seq_len+i*seq_len+tid];
            temp_y = y[bid*seq_len*seq_len+i*seq_len+tid];

            float temp_dyy = 0.0;
            temp_dyy = temp * temp_y;

            float sum_dyy_val = 0.0;
            sum_dyy_val = blockReduceSum(temp_dyy);
            if(threadIdx.x == 0){
                sum_dyy = sum_dyy_val;
            }

            __syncthreads();
            out[bid*seq_len*seq_len+i*seq_len+tid] = (temp - sum_dyy)*temp_y*scaler;
        }   

    }
        
}

__global__ void gradients_add_gpu_512(float *gradients_sum, float *gradients, int size){

    __shared__ float sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float temp = 0.0;
    temp = gradients[tid];
    temp *= temp;

    float sum_val =0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0)
        sum = sum_val;

    __syncthreads();
    
    gradients_sum[0] = sum;

}

__global__ void gradients_add_gpu_2048(float *gradients_sum, float *gradients, int size){

    __shared__ float sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float temp = gradients[bid*512+tid];
    temp *= temp;

    float sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}


__global__ void gradients_add_gpu_512_2048(float *gradients_sum, float *gradients, int size){
        
    __shared__ float sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float temp = gradients[bid*512+tid];
    temp *= temp;

    float sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}


__global__ void gradients_add_gpu_512_512(float *gradients_sum, float *gradients, int size){
        
    __shared__ float sum;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float temp = gradients[bid*512+tid];
    temp *= temp;

    float sum_val = 0;
    sum_val = blockReduceSum(temp);

    if(threadIdx.x == 0){
        sum = sum_val;
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(gradients_sum, sum);

}

__global__ void tensor_add_tensor_add_tensor_gpu(float *D, float *A, float *B, float *C, const int dim0, const int dim1, const int dim2){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int index = bid*blockDim.x+tid;
    int size = dim0*dim1*dim2;
    if((bid*blockDim.x+tid) < size){
        D[index] =  A[index]+B[index]+C[index];
    }
    
}

void gpu_save_float(float *gpu_array, char *file_name, int size){

    float *temp = (float*)malloc(sizeof(float)*size);
    cudaMemcpy(temp, gpu_array, sizeof(float)*size, cudaMemcpyDeviceToHost);
    FILE *fp = NULL;

    strcat(file_name, ".txt");
    fp = fopen(file_name, "w");
    if(fp == NULL){
            printf("open error in %s !\n", file_name);
    }
    for(int j=0;j<size;j++){
        fprintf(fp, "%.8f ", temp[j]);
    }
    fclose(fp);
    free(temp);

}


void cpu_save_float(float *cpu_array, char *file_name, int size){
    
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
