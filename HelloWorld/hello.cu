#include <cuda.h>
#include <stdio.h>

__global__ void mykernel(void){
    printf("Hello World from GPU by thread %d!\n", threadIdx.x);
}

int main(void){
    mykernel<<<1,1>>>();
    cudaDeviceSynchronize();
    mykernel<<<1,4>>>();
    cudaDeviceSynchronize();
    return 0;
}