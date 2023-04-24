#ifndef VALIDATE_H

#include <stdio.h>
#include <cmath>

#define VALIDATE_H

void readmask(char* filename, unsigned int* store_array, int size){
    FILE *file;
    file = fopen(filename,"rb");
    if(file == NULL){
        printf("error!");
    }
    fread(store_array,sizeof(unsigned int),size,file);
}

void readbinary(char* filename, float* store_array, int size){
    FILE *file;
    file = fopen(filename,"rb");
    if(file == NULL){
        printf("error!");
    }
    fread(store_array,sizeof(float),size,file);
}

void readbinary_char(char* filename, unsigned char* store_array, int size){
    FILE *file;
    file = fopen(filename,"rb");
    if(file == NULL){
        printf("error!");
    }
    fread(store_array,1,size,file);
}

void readtxt(char* filename, float* store_array, int size){
    FILE *file;
    file = fopen(filename, "rb");
    if(file == NULL){
        printf("error!");
    }

    for(int i=0;i<size;i++){
        fscanf(file,"%f", &store_array[i]);
    }
    
}

void ifitsright(float* result_array, float* store_array, int size){
    int error_num = 0;
    float sum =0.0;
    float value_sum = 0.0;
    for(int i=0;i<size;i++){
        printf("calculate : %.15f, read: %.15f, %d\n",result_array[i],store_array[i], i);
        if(result_array[i] != store_array[i]){
            sum += fabs(result_array[i] - store_array[i]);
            value_sum += fabs(store_array[i]);
            // if(fabs(result_array[i] - store_array[i] < 0.0001))
            // {
            //     printf("calculate : %.15f, read: %.15f, %d\n",result_array[i],store_array[i], i);
            // }
            error_num++;
        }
    }
    if(error_num != 0){
        printf("error elements difference sum: %.15f,\navg difference: %.15f,\nerror count: %d,\nvalue: %.15f,\ndiff/value: %.15f\n", sum, sum/error_num, error_num, value_sum/size, (sum/error_num)/(value_sum/size));
    }else{
        printf("All elements (%d) are same with the intermediate results calculated by BST.\n", size);
    }
    printf("\n");
    
}

#endif


