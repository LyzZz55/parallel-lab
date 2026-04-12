#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARP_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define ITEMS_PER_THREAD 2

__global__ void warp_block_scan_items(const int* d_in, int* d_out, int* d_block_sums, int n) {
    __shared__ int warp_sums[WARP_PER_BLOCK];

    int tid = threadIdx.x;
    int lane_id = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int base_idx = (blockIdx.x * blockDim.x + tid) * ITEMS_PER_THREAD;

    int val[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        val[i] = (idx < n) ? __ldg(&d_in[idx]) : 0;
    }

    #pragma unroll
    for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
        val[i] += val[i-1];
    }
    int thread_sum = val[ITEMS_PER_THREAD-1];

    // warp 内 exclusive scan on thread_sum
    int warp_incl = thread_sum;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n_val = __shfl_up_sync(0xffffffff, warp_incl, offset);
        warp_incl += (lane_id >= offset) * n_val;
    }
    int warp_excl = warp_incl - thread_sum;

    // 应用 warp 内偏移
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        val[i] += warp_excl;
    }

    // 每个 warp 的总和（由最后一个线程存储）
    int warp_total = __shfl_sync(0xffffffff, warp_incl, WARP_SIZE-1);

    if (lane_id == WARP_SIZE - 1) {
        warp_sums[warp_id] = warp_total;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        int sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0;
        // inclusive scan
        int incl = sum;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, incl, offset);
            incl += (lane_id >= offset) * n;
        }
        int excl = incl - sum;   // exclusive prefix
        if (lane_id < num_warps) {
            warp_sums[lane_id] = excl;
        }
    }
    __syncthreads();

    // 加上 warp 级别的偏移
    int warp_offset = warp_sums[warp_id];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        val[i] += warp_offset;
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < n) d_out[idx] = val[i];
    }

    // 输出 block 总和
    if (d_block_sums && tid == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = val[ITEMS_PER_THREAD-1];
    }
}

__global__ void add_block_sums_items(int* d_out, const int* d_block_sums_scanned, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int base_idx = (bid * blockDim.x + tid) * ITEMS_PER_THREAD;
    int offset = (bid > 0) ? __ldg(&d_block_sums_scanned[bid - 1]) : 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < n) d_out[idx] += offset;
    }
}

void student_prefix_sum(const int* d_in, int* d_out, int n) {
    if (n <= 0) return;
    int items_per_block = BLOCK_SIZE * ITEMS_PER_THREAD;
    int num_blocks = (n + items_per_block - 1) / items_per_block;

    if (num_blocks == 1) {
        warp_block_scan_items<<<1, BLOCK_SIZE>>>(d_in, d_out, nullptr, n);
        cudaDeviceSynchronize();
        return;
    }

    int* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    warp_block_scan_items<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);
    cudaDeviceSynchronize();

    int* d_block_sums_scanned;
    cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int));
    student_prefix_sum(d_block_sums, d_block_sums_scanned, num_blocks);

    add_block_sums_items<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scanned);
}