#include <stdio.h>
#include <cmath>

//nA=mB
template<typename T1, typename T2, typename T3>
void multiheadmatmulNN(T1* C, T2* A, T3* B, const int batch_size, const int head_num, const int mA, const int nA, const int nB){
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<head_num;j++){
            for(int k=0;k<mA;k++){
                for(int m=0;m<nB;m++){
                    float sum = 0;
                    for(int n=0;n<nA;n++){
                        sum += (A[i*mA*nA*head_num + j*nA*mA + k*nA + n] * B[i*nA*nB*head_num + j*nB*nA + n*nB + m]);
                    }
                    C[i*mA*nB*head_num + j*nB*mA + k*nB + m] = sum;
                }
            }
        }
    }
    // int batch_step_x = head_num * mA * nA;
	// int batch_step_y = head_num * nA * nB;
	// int batch_step_z = head_num * mA * nB;

	// int heads_step_x = mA * nA;
	// int heads_step_y = nA * nB;
	// int heads_step_z = mA * nB;

    
	// for (int batch = 0; batch < batch_size; batch++)
	// {
	// 	for (int heads = 0; heads < head_num; heads++)
	// 	{
	// 		for (int i = 0; i < mA; i++)
	// 		{
	// 			for (int j = 0; j < nB; j++)
	// 			{
	// 				float sum = 0;
	// 				for (int k = 0; k < nA; k++)
	// 				{
	// 					sum += (A[batch * batch_step_x + heads * heads_step_x + i * nA + k] * B[batch * batch_step_y + heads * heads_step_y + k * nB + j]);
	// 				}
	// 				C[batch * batch_step_z + heads * heads_step_z + i * nB + j] = sum;
	// 			}
	// 		}
	// 	}
	// }
}
//generate a mask

//nA=nB
template<typename T1, typename T2, typename T3>
void multiheadmatmulNT(T1* C, T2* A, T3* B, const int batch_size, const int head_num, const int mA, const int nA, const int mB){
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++){
        for(int j=0;j<head_num;j++){
            for(int k=0;k<mA;k++){
                for(int m=0;m<mB;m++){
                    float sum=0;
                    for(int n=0;n<nA;n++){
                        sum += (A[i*mA*nA*head_num + j*mA*nA + k*nA + n] * B[i*mB*nA*head_num + j*mB*nA + m*nA + n]);
                    }
                    C[i*mA*mB*head_num + j*mA*mB + k*mB + m] = 0.0;
                    
                    if(k>=m)
                    {
                        C[i*mA*mB*head_num + j*mA*mB + k*mB + m] = sum;

                    }else{

                        C[i*mA*mB*head_num + j*mA*mB + k*mB + m] = 0.0;
                        // C[i*mA*mB*head_num + j*mA*mB + k*mB + m] = sum;
                    }
                }
            }
        }
    }
}

//mA=mB
template<typename T1, typename T2, typename T3>
void multiheadmatmulTN(T1* C, T2* A, T3* B, const int batch_size, const int head_num, const int mA, const int nA, const int nB){
    #pragma omp parallel for
    for(int i=0; i<batch_size;i++){
        for(int j=0;j<head_num;j++){
            for(int k=0;k<nA;k++){
                for(int m=0;m<nB;m++){
                    float sum = 0.0;
                    for(int n=0;n<mA;n++){
                        sum += (A[i*mA*nA*head_num + j*mA*nA + n*nA + k] * B[i*nB*mA*head_num + j*mA*nB + n*nB + m]);
                    }
                    C[i*nA*nB*head_num + j*nB*nA + k*nB + m] = 0.0;
                    C[i*nA*nB*head_num + j*nB*nA + k*nB + m] = sum;
                }
            }
        }
    }
}

//nA=mB
void matrixMulNNCPU(float *C, const float *A, const float *B, const int mA, const int nA, const int nB)
{   
    #pragma omp parallel for
    for (unsigned int i = 0; i < mA; ++i)
        for (unsigned int j = 0; j < nB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < nA; ++k)
            {
                double a = A[i * nA + k];
                double b = B[k * nB + j];
                sum += a * b;
            }
            C[i * nB + j] = 0.0;
            C[i * nB + j] = (float)sum;
        }
}

//nA=nB
void matrixMulNTCPU(float *C, const float *A, const float *B, const int mA, const int nA, const int mB){
    #pragma omp parallel for
    for(int i=0;i<mA;i++){
        for(int j=0;j<mB;j++){
            float sum = 0;
            for(int k=0;k<nA;k++){
                sum += (A[i*nA + k] * B[j*nA + k]);
            }
            C[i*mB + j] = sum;        
        }
    }
}

//mA=mB
void matrixMulTNCPU(float *C, const float *A, const float *B, const int mA, const int nA, const int nB){
    #pragma omp parallel for
    for(int i=0;i<nA;i++){
        for(int j=0;j<nB;j++){
            float sum =0;
            for(int k=0;k<mA;k++){
                sum += A[k*nA + i] * B[k*nB + j];
            }
            C[i*nB + j] = sum;
        }
    }

}
