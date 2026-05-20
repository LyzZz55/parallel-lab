/*
 * ============================================================
 *  并行计算课程作业3：CUDA 单 Kernel 固定 Block 前缀和
 * ============================================================
 *
 * 任务：实现一个高效的 CUDA inclusive prefix sum。
 *
 * 给定长度为 n 的整型数组 d_in，计算：
 *   d_out[i] = d_in[0] + d_in[1] + ... + d_in[i]
 *
 * 要求：
 *   1. 只修改本文件
 *   2. 不得修改 student_prefix_sum_13block_kernel / student_prefix_sum_26block_kernel 的 kernel 签名
 *   3. 不得使用 Thrust / cuBLAS / cub 等高级库
 *   4. 需正确处理任意长度 n（不保证是 2 的幂次）
 *   5. 本文件中不要写 kernel launch，评测程序会固定用 <<<13, 256>>> 和 <<<26, 256>>> 调用
 *
 * 提交：将本文件上传至提交系统，文件名保持 student_kernel.cu
 * ============================================================
 */

#include <cuda_runtime.h>

// ------------------------------------------------------------
// 全局变量：用于 13-block kernel 的同步与中间存储
// ------------------------------------------------------------
__device__ int g_mutex13 = 0;
__device__ int g_sense13 = 0;
__device__ int g_block_flags13[13];
__device__ int g_block_sums13[13];
__device__ int g_block_offsets13[13];
__device__ int g_prefix_offset_13 = 0;

// ------------------------------------------------------------
// 全局变量：用于 26-block kernel 的同步与中间存储
// ------------------------------------------------------------
__device__ int g_mutex26 = 0;
__device__ int g_sense26 = 0;
__device__ int g_block_flags26[26];
__device__ int g_block_sums26[26];
__device__ int g_block_offsets26[26];
__device__ int g_prefix_offset_26 = 0;

// ------------------------------------------------------------
// 网格同步宏（13 block 版本）
// ------------------------------------------------------------
#define GRID_SYNC_13() \
do { \
    if (threadIdx.x == 0) { \
        atomicExch(&g_block_flags13[blockIdx.x], 0); \
    } \
    __syncthreads(); \
    if (threadIdx.x == 0) { \
        __threadfence(); \
        int token = atomicAdd(&g_mutex13, 1); \
        if (token == gridDim.x - 1) { \
            g_mutex13 = 0; \
            int new_sense = 1 - atomicAdd(&g_sense13, 0); \
            atomicExch(&g_sense13, new_sense); \
            __threadfence(); \
        } else { \
            int sense = atomicAdd(&g_sense13, 0); \
            while (atomicAdd(&g_sense13, 0) == sense) {} \
            __threadfence(); \
        } \
        __threadfence(); \
        atomicExch(&g_block_flags13[blockIdx.x], 1); \
    } else { \
        while (atomicAdd(&g_block_flags13[blockIdx.x], 0) == 0) {} \
    } \
    __syncthreads(); \
} while(0)

// ------------------------------------------------------------
// 网格同步宏（26 block 版本）
// ------------------------------------------------------------
#define GRID_SYNC_26() \
do { \
    if (threadIdx.x == 0) { \
        atomicExch(&g_block_flags26[blockIdx.x], 0); \
    } \
    __syncthreads(); \
    if (threadIdx.x == 0) { \
        __threadfence(); \
        int token = atomicAdd(&g_mutex26, 1); \
        if (token == gridDim.x - 1) { \
            g_mutex26 = 0; \
            int new_sense = 1 - atomicAdd(&g_sense26, 0); \
            atomicExch(&g_sense26, new_sense); \
            __threadfence(); \
        } else { \
            int sense = atomicAdd(&g_sense26, 0); \
            while (atomicAdd(&g_sense26, 0) == sense) {} \
            __threadfence(); \
        } \
        __threadfence(); \
        atomicExch(&g_block_flags26[blockIdx.x], 1); \
    } else { \
        while (atomicAdd(&g_block_flags26[blockIdx.x], 0) == 0) {} \
    } \
    __syncthreads(); \
} while(0)

#define BLOCK_SIZE 256
#define EPT 16
#define TILE_SIZE (BLOCK_SIZE*EPT)
#define WARP_SIZE 32

// ------------------------------------------------------------
// 13-block kernel 实现（warp 级协作 + 减少全局同步）
// ------------------------------------------------------------
__global__ void student_prefix_sum_13block_kernel(const int* in, int* out, int n) {

    if (n <= 0) return;

    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    int num_iters = (num_tiles + gridDim.x - 1) / gridDim.x;

    __shared__ int s_warp_sums[8];
    __shared__ int s_warp_prefix[8];

    for (int iter = 0; iter < num_iters; ++iter) {
        int tile_start = iter * gridDim.x;
        int cur_tiles = num_tiles - tile_start;
        if (cur_tiles > gridDim.x) cur_tiles = gridDim.x;
        int my_tile = tile_start + blockIdx.x;
        bool valid_tile = (blockIdx.x < cur_tiles);

        int vals[EPT];           // 提升作用域，供阶段3使用
        int block_exclusive = 0;

        // ---- 阶段 1: 局部 inclusive scan 与块总和 (warp shuffle) ----
        if (valid_tile) {
            int tile_base = my_tile * TILE_SIZE;
            int base_idx = tile_base + threadIdx.x * EPT;

            int thread_sum = 0;
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                int idx = base_idx + i;
                int v = (idx < n) ? in[idx] : 0;
                thread_sum += v;
                vals[i] = thread_sum;
            }

            int lane = threadIdx.x % WARP_SIZE;
            int warp_id = threadIdx.x / WARP_SIZE;
            int warp_scan = thread_sum;

            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
                int n = __shfl_up_sync(0xffffffff, warp_scan, offset);
                if (lane >= offset) warp_scan += n;
            }

            int warp_exclusive = warp_scan - thread_sum;
            int warp_total = __shfl_sync(0xffffffff, warp_scan, WARP_SIZE - 1);

            if (lane == WARP_SIZE - 1) {
                s_warp_sums[warp_id] = warp_total;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                int accum = 0;
                int num_warps = blockDim.x / WARP_SIZE;
                for (int i = 0; i < num_warps; ++i) {
                    s_warp_prefix[i] = accum;
                    accum += s_warp_sums[i];
                }
                g_block_sums13[blockIdx.x] = accum;
            }
            __syncthreads();

            block_exclusive = s_warp_prefix[warp_id] + warp_exclusive;
        }

        // ---- 全局同步 1 ----
        GRID_SYNC_13();

        // ---- 阶段 2: block 0 计算块间偏移 ----
        if (blockIdx.x == 0) {
            int accum = g_prefix_offset_13;
            for (int i = 0; i < cur_tiles; ++i) {
                g_block_offsets13[i] = accum;
                accum += g_block_sums13[i];
            }
            g_prefix_offset_13 = accum;
        }

        // ---- 全局同步 2 ----
        GRID_SYNC_13();

        // ---- 阶段 3: 加上全局偏移并写出最终结果 ----
        if (valid_tile) {
            int global_offset = g_block_offsets13[blockIdx.x];
            int tile_base = my_tile * TILE_SIZE;
            int base_idx = tile_base + threadIdx.x * EPT;

            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                int idx = base_idx + i;
                if (idx < n) {
                    out[idx] = vals[i] + block_exclusive + global_offset;
                }
            }
        }
    }

    // 所有迭代完成后做一次全局同步，安全重置状态
    GRID_SYNC_13();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_mutex13 = 0;
        g_sense13 = 0;
        g_prefix_offset_13 = 0;
    }
}


__global__ void student_prefix_sum_26block_kernel(const int* in, int* out, int n) {
    if (n <= 0) return;

    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    int num_iters = (num_tiles + gridDim.x - 1) / gridDim.x;

    __shared__ int s_warp_sums[8];
    __shared__ int s_warp_prefix[8];

    for (int iter = 0; iter < num_iters; ++iter) {
        int tile_start = iter * gridDim.x;
        int cur_tiles = num_tiles - tile_start;
        if (cur_tiles > gridDim.x) cur_tiles = gridDim.x;
        int my_tile = tile_start + blockIdx.x;
        bool valid_tile = (blockIdx.x < cur_tiles);

        int vals[EPT];
        int block_exclusive = 0;

        if (valid_tile) {
            int tile_base = my_tile * TILE_SIZE;
            int base_idx = tile_base + threadIdx.x * EPT;

            int thread_sum = 0;
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                int idx = base_idx + i;
                int v = (idx < n) ? in[idx] : 0;
                thread_sum += v;
                vals[i] = thread_sum;
            }

            int lane = threadIdx.x % WARP_SIZE;
            int warp_id = threadIdx.x / WARP_SIZE;
            int warp_scan = thread_sum;

            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
                int n = __shfl_up_sync(0xffffffff, warp_scan, offset);
                if (lane >= offset) warp_scan += n;
            }

            int warp_exclusive = warp_scan - thread_sum;
            int warp_total = __shfl_sync(0xffffffff, warp_scan, WARP_SIZE - 1);

            if (lane == WARP_SIZE - 1) {
                s_warp_sums[warp_id] = warp_total;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                int accum = 0;
                int num_warps = blockDim.x / WARP_SIZE;
                for (int i = 0; i < num_warps; ++i) {
                    s_warp_prefix[i] = accum;
                    accum += s_warp_sums[i];
                }
                g_block_sums26[blockIdx.x] = accum;
            }
            __syncthreads();

            block_exclusive = s_warp_prefix[warp_id] + warp_exclusive;
        }

        GRID_SYNC_26();

        if (blockIdx.x == 0) {
            int accum = g_prefix_offset_26;
            for (int i = 0; i < cur_tiles; ++i) {
                g_block_offsets26[i] = accum;
                accum += g_block_sums26[i];
            }
            g_prefix_offset_26 = accum;
        }

        GRID_SYNC_26();

        if (valid_tile) {
            int global_offset = g_block_offsets26[blockIdx.x];
            int tile_base = my_tile * TILE_SIZE;
            int base_idx = tile_base + threadIdx.x * EPT;

            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                int idx = base_idx + i;
                if (idx < n) {
                    out[idx] = vals[i] + block_exclusive + global_offset;
                }
            }
        }
    }

    GRID_SYNC_26();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_mutex26 = 0;
        g_sense26 = 0;
        g_prefix_offset_26 = 0;
    }
}