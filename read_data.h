#define n_train 90000000
#define n_valid 5000000
#define n_test 5000000

#include <stdlib.h>
#include <time.h> 
#include <string.h>
#include <iostream>
#include <fstream>

using namespace std;


void enwik8(unsigned char *X, unsigned char *trX, unsigned char *vaX, unsigned char *teX)
{   
    int n_total = n_train + n_valid + n_test;
    
    unsigned char temp_uc;
    ifstream srcFile("enwik8", ios::in); 

    if (!srcFile) { 
        cout << "error opening source file." << endl;
        exit(0);
    }

    for(int i = 0;i < n_total;i++){
        srcFile >> temp_uc;
        X[i] = temp_uc;
    }
    srcFile.close();
    
    memcpy(trX, X, sizeof(unsigned char)*n_train); 
    memcpy(vaX, X+sizeof(unsigned char)*n_train, sizeof(unsigned char)*n_valid);
    memcpy(teX, X+sizeof(unsigned char)*(n_train + n_valid),sizeof(unsigned char)*n_test);
}


void my1_print_rank0(int num)
{

    cout<<"Not including "<<num<<" sequences"<<endl;
    
    return;
}

void my2_print_rank0(int num)
{
 
    cout<<"Number of minibatches this epoch:" <<num<<endl;
    
    return;
}
