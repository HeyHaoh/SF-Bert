#include <stdio.h>
#include <cmath>

#define ex2 1.4426950408889634f
#define PI 3.141592654

//gpu done
// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// y     = xhat*g + b
void layernorm(float* out, float* mean, float* rstd, const float* input, const float* gain, const float* bias, const int batch_size, const int seq_len, const int state){

    for(int i=0; i<batch_size*seq_len; i++){
        double sum_of_row = 0.0;
        double sum_of_row_sqr = 0.0;
        double mean_of_row_sqr = 0.0;

        for(int j=0; j<state; j++){
            
            sum_of_row += input[i*state + j];
            sum_of_row_sqr += (input[i*state+j] * input[i*state+j]);
        }

        mean[i] = sum_of_row / state;
        // mean[i] = sum_of_row;
        mean_of_row_sqr  = sum_of_row_sqr / state;
        
        rstd[i] = 1.0f / (double)sqrt(mean_of_row_sqr - (mean[i] * mean[i]) + 1e-5);

        for(int j=0; j<state; j++){
            out[i*state+j] = 0;
            out[i*state+j] = ((input[i*state+j]-mean[i]) * rstd[i]) * gain[j] + bias[j];
        }
    }
    
}
// dg = sum(dy * xhat(x), axis=0)
// db = sum(dy, axis=0)

// xmean = x - mean(x, axis=1)
// xvar  = var(x, axis=1)
// xstdr = rcp(sqrt(xvar + epsilon))
// xhat  = xmean * xstdr
// dy    = dy * g
// sum1  = sum(xhat * dy, axis=1)
// sum2  = sum(dy, axis=1)
// dx    = (dy - ((xhat * sum1 + sum2) * rcpK)) * xstd

// gpu done
void layernorm_dg_db(float *dg, float *db, float *dy, float *x, float *mean, float *rstd, const int batch_size, const int seq_len, const int state){
    
    float *x_hat = (float*)malloc(sizeof(float) * batch_size * seq_len * state);
    for(int i=0;i<batch_size*seq_len*state;i++){
        x_hat[i] = 0.0;
    }
    
    for(int i=0; i<state; i++){
        dg[i] = 0.0;
        db[i] = 0.0;
        for(int j=0; j<batch_size*seq_len; j++){
            x_hat[j*state + i] = (x[j*state+i] - mean[j]) * rstd[j];
            dg[i] += dy[j*state+i] * x_hat[j*state+i];
            db[i] += dy[j*state+i];
        }
    }

    free(x_hat);

}

// not check 
void layernorm_grad_dx(float* dx, float* x, float* mean, float* rstd, float* dy, float* gamma, const int batch_size, const int seq_len, const int state){

    float *sum1 = (float*)malloc(sizeof(float) * batch_size * seq_len);
    float *sum2 = (float*)malloc(sizeof(float) * batch_size * seq_len);

    for(int i=0;i<batch_size*seq_len;i++){
        sum1[i] = 0.0;
        sum2[i] = 0.0;
    }

    for(int i=0; i<batch_size*seq_len; i++){
        for(int j=0; j<state; j++){
            dx[i*state+j] = x[i*state+j] - mean[i];
            dx[i*state+j] *= rstd[i];
            dy[i*state+j] *= gamma[j];
            sum1[i] += (dx[i*state+j] * dy[i*state+j]);
            sum2[i] += dy[i*state+j];
        }
        for(int j=0; j<state; j++){
            dx[i*state+j] =  (dy[i*state+j] - ((dx[i*state+j] * sum1[i] + sum2[i]) * (1.0 / state))) * rstd[i];
        }
    }

    free(sum1);
    free(sum2);
}


