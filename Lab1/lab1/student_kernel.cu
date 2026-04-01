/*
 * ============================================================
 *  并行计算课程作业：CUDA 前缀和（Inclusive Scan）
 *  刘英哲 2300012753
 * ============================================================
 *
 * 任务：实现一个高效的 CUDA inclusive prefix sum kernel。
 *
 * 给定长度为 n 的整型数组 d_in，计算：
 *   d_out[i] = d_in[0] + d_in[1] + ... + d_in[i]
 *
 * 要求：
 *   1. 只修改本文件（可添加辅助 kernel / 设备函数）
 *   2. 不得修改 student_prefix_sum 的函数签名
 *   3. 不得使用 Thrust / cuBLAS / cub 等高级库
 *   4. 需正确处理任意长度 n（不保证是 2 的幂次）
 *
 * 提交：将本文件上传至提交系统，文件名保持 student_kernel.cu
 * ============================================================
 */

#include <cuda_runtime.h>

// ------------------------------------------------------------
// 可在此处添加辅助 kernel 或 __device__ 函数
// ------------------------------------------------------------

#define BLOCK_SIZE 512

__global__ void block_scan(const int* d_in, int* d_out, int* d_block_sums, int n) {
    __shared__ int temp[BLOCK_SIZE][2];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;

    int val = (idx < n) ? d_in[idx] : 0;
    temp[tid][0] = val;
    __syncthreads();
    int src = 0;
    int dst = 1;

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        if(tid>=offset) temp[tid][dst] = temp[tid][src] + temp[tid-offset][src];
        else temp[tid][dst] = temp[tid][src];
        src = dst;
        dst = dst^1;
        __syncthreads();
    }

    if (idx < n) {
        d_out[idx] = temp[tid][src];
    }

    if (d_block_sums && tid == BLOCK_SIZE - 1) {
        d_block_sums[bid] = temp[BLOCK_SIZE - 1][src];
    }
}


__global__ void add_block_sums(int* d_out, int* d_block_sums_scanned, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;
    if (bid > 0 && idx < n) {
        d_out[idx] += d_block_sums_scanned[bid - 1];
    }
}

// ------------------------------------------------------------
// 主接口：不得修改函数签名
//   d_in  - device 端输入数组（长度 n, int）
//   d_out - device 端输出数组（长度 n, int）
//   n     - 元素个数
// ------------------------------------------------------------
void student_prefix_sum(const int* d_in, int* d_out, int n) {
    if (n <= 0) return;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 1) {

        block_scan<<<1, BLOCK_SIZE>>>(d_in, d_out, nullptr, n);
        cudaDeviceSynchronize();
        return;
    }


    int* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));

    block_scan<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    int* d_block_sums_scanned;
    cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int));

    student_prefix_sum(d_block_sums, d_block_sums_scanned, num_blocks);
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scanned);

    return;
}
