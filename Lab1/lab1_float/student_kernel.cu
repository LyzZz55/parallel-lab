/*
 * ============================================================
 *  并行计算课程作业：CUDA 前缀和（Inclusive Scan）
 * ============================================================
 *
 * 任务：实现一个高效的 CUDA inclusive prefix sum kernel。
 *
 * 给定长度为 n 的浮点数组 d_in，计算：
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

#define BLOCK_SIZE 512   // 块大小，适配 K20c 架构

// 块内扫描内核：计算每个块的排他前缀和，并记录块总和
__global__ void block_scan(float* d_in, float* d_out, float* d_block_sums, int n) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;

    // 加载数据，超范围补0
    float val = (idx < n) ? d_in[idx] : 0.0f;
    temp[tid] = val;
    __syncthreads();

    // Kogge-Stone 扫描（单缓冲区）
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        float t = 0.0f;
        if (tid >= offset) t = temp[tid - offset];
        __syncthreads();
        if (tid >= offset) temp[tid] += t;
        __syncthreads();
    }

    float inclusive = temp[tid];   // 包含当前元素的扫描和

    // 输出排他前缀和（全局暂时排他，后续加块间偏移）
    if (idx < n) {
        if (tid == 0)
            d_out[idx] = 0.0f;
        else
            d_out[idx] = temp[tid - 1];
    }

    // 存储块总和（供下一级递归使用）
    if (d_block_sums && tid == BLOCK_SIZE - 1) {
        d_block_sums[bid] = inclusive;
    }
}


// 将块前缀和加到各个元素上
__global__ void add_block_sums(float* d_out, float* d_block_sums_scanned, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * BLOCK_SIZE + tid;
    if (bid > 0 && idx < n) {
        d_out[idx] += d_block_sums_scanned[bid - 1];
    }
}

__global__ void convert_to_inclusive(float* d_out, float* d_in, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) d_out[idx] += d_in[idx];
}


// ------------------------------------------------------------
// naive 示例：单线程顺序扫描（仅供参考，性能极差）
// 请用高效并行实现替换 student_prefix_sum 的函数体。
// ------------------------------------------------------------
__global__ void naive_scan_kernel(const float* in, float* out, int n) {
    // 只用 1 个线程顺序计算，等价于 CPU 实现
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = in[0];
        for (int i = 1; i < n; ++i)
            out[i] = out[i - 1] + in[i];
    }
}

// ------------------------------------------------------------
// 主接口：不得修改函数签名
//   d_in  - device 端输入数组（长度 n，float）
//   d_out - device 端输出数组（长度 n，float）
//   n     - 元素个数
// ------------------------------------------------------------
void student_prefix_sum(float* d_in, float* d_out, int n) {
    // TODO: 用高效并行实现替换下面这行代码
    //
    // naive_scan_kernel<<<1, 1>>>(d_in, d_out, n);
    // return;


    if (n <= 0) return;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 情况1：只需单个块，直接计算
    if (num_blocks == 1) {
        block_scan<<<1, BLOCK_SIZE>>>(d_in, d_out, nullptr, n);
        cudaDeviceSynchronize();
        // 将排他前缀和转换为 inclusive
        convert_to_inclusive<<<1, BLOCK_SIZE>>>(d_out, d_in, n);
        cudaDeviceSynchronize();
        return;
    }

    // 情况2：需要多级处理
    // 分配块总和数组
    float* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    // 第一遍扫描：块内前缀和 + 块总和
    block_scan<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    // 分配空间存放块总和的前缀和结果
    float* d_block_sums_scanned;
    cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(float));

    // 递归计算块总和的前缀和
    student_prefix_sum(d_block_sums, d_block_sums_scanned, num_blocks);

    // 将块前缀和加到各块的结果上
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);

    convert_to_inclusive<<<num_blocks, BLOCK_SIZE>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    // 释放临时内存
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scanned);
}
