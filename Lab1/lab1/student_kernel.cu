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
#define WARP_SIZE 32
#define WARP_PER_BLOCK BLOCK_SIZE/WARP_SIZE
// 使用 warp shuffle 的高效 block 级 scan kernel
// 每个 block 计算局部的 inclusive prefix sum，并可选地输出 block 总和
__global__ void warp_block_scan(const int* d_in, int* d_out, int* d_block_sums, int n) {
    __shared__ int warp_scan[WARP_PER_BLOCK];

    int tid = threadIdx.x;
    int lane_id = tid & (WARP_SIZE - 1);   // 线程在 warp 内的编号 (0~31)
    int warp_id = tid / WARP_SIZE;         // warp 索引
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. 加载数据，超出范围的补 0
    int val = (idx < n) ? d_in[idx] : 0;

    // 2. 在 warp 内部进行 inclusive scan（利用 shuffle 指令）
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n_val;
    }

    // 3. 每个 warp 的最后一个线程将 warp 总和存入共享内存
    if (lane_id == WARP_SIZE - 1) {
        warp_scan[warp_id] = val;
    }
    __syncthreads();

    // 4. 对 warp 总和进行 exclusive scan（使用第一个 warp）
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        int sum = (lane_id < num_warps) ? warp_scan[lane_id] : 0;
        // 在 warp 内进行 inclusive scan
        #pragma unroll
        for (int offset = 1; offset < WARP_PER_BLOCK; offset <<= 1) {
            int n_sum = __shfl_up_sync(0xffffffff, sum, offset);
            if (lane_id >= offset) sum += n_sum;
        }
        if (lane_id < num_warps) {
            warp_scan[lane_id] = (lane_id!=0)*__shfl_up_sync(0xffffffff, sum, 1);
        }
    }
    __syncthreads();

    // 5. 将当前 warp 内的前缀和加上前面 warp 的累计和，得到最终前缀和
    int prefix = warp_scan[warp_id];
    val += prefix;

    // 6. 写回结果（仅有效线程）
    if (idx < n) {
        d_out[idx] = val;
    }

    // 7. 如果需要，由最后一个线程输出 block 的总和
    if (d_block_sums && tid == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = val;
    }
}

// 辅助 kernel：为每个元素加上对应 block 的前缀偏移量（来自 block sums 的 scan 结果）
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

    // 单 block 情况：直接 scan 即可
    if (num_blocks == 1) {
        warp_block_scan<<<1, BLOCK_SIZE>>>(d_in, d_out, nullptr, n);
        cudaDeviceSynchronize();
        return;
    }

    // 多 block 情况：先计算每个 block 的局部 scan 和 block 总和
    int* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    warp_block_scan<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    // 递归计算 block 总和的 scan（得到每个 block 之前的前缀和）
    int* d_block_sums_scanned;
    cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int));
    student_prefix_sum(d_block_sums, d_block_sums_scanned, num_blocks);

    // 将 block 偏移量加到对应的输出元素上
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scanned);
}