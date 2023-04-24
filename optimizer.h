#include <stdio.h>
#include <cmath>


float clip_by_global_norm(float global_norm, float clip_norm){
    
    float norm_scale = 0.0;
    if(global_norm > clip_norm){
        norm_scale = clip_norm/ global_norm;
    }else{
        norm_scale = 1.0;
    }
    return norm_scale;
}

double adam_got_beta_power(double beta, float global_step){
    
    double beta_power = 1.0;
    for(int i=0;i<global_step;i++)
    {
        beta_power *= beta;
    }
    return beta_power;
}

double adam_got_lr(double learning_rate, double beta1_power, double beta2_power){
    
    double lr = 0.0;
    lr = learning_rate * sqrt(1.0 - beta2_power) / (1.0 - beta1_power);
    return lr;
}   

//gpu done
void adam_apply_gradients(float *C, float* dw, float* mean, float* var, float beta1, float beta2, float beta_power1, float beta_power2, 
                          float learning_rate, float epsilon, float norm_scale, float grad_scale, float clip_sigma, 
                          const int size){ 

    if(norm_scale !=0){

        for(int i=0; i<size;i++){

            dw[i] *= grad_scale * norm_scale;
            var[i] = beta2 * var[i] + ((1.0f - beta2) * dw[i] * dw[i]);
            mean[i] = beta1 * mean[i] + (1.0f - beta1) * dw[i];
            C[i] -=  learning_rate * mean[i] * (1.0f / (sqrt(var[i]) + epsilon));

        }
    }

}

