#include <cuda_runtime.h>

#define ITEMS_13 24
#define ITEMS_26 16
#define TILE_SIZE_13 (256 * ITEMS_13)   // 6144
#define TILE_SIZE_26 (256 * ITEMS_26)   // 4096
#define MAX_TILES   (1 << 20)


#define MAKE_DATA(stat, agg) (unsigned int)(((stat) << 30) | ((agg) & 0x3FFFFFFF))
#define STAT(data) ((unsigned int)(data) >> 30)
#define AGG(data)  ((int)((unsigned int)(data) & 0x3FFFFFFF))

// ---------- 13 block 全局变量 ----------
__device__ unsigned int tile_data_13[MAX_TILES];
__device__ int          g_counter_13;
__device__ unsigned int g_goal_13[13];
__device__ volatile unsigned int g_in_13[13];
__device__ volatile unsigned int g_out_13[13];

// ---------- 26 block 全局变量 ----------
__device__ unsigned int tile_data_26[MAX_TILES];
__device__ int          g_counter_26;
__device__ unsigned int g_goal_26[26];
__device__ volatile unsigned int g_in_26[26];
__device__ volatile unsigned int g_out_26[26];

// ---------- 全局软件栅栏 ----------
__device__ __forceinline__ void grid_barrier(unsigned int* goal_arr,
                             volatile unsigned int* in_arr,
                             volatile unsigned int* out_arr,
                             int num_blocks) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ unsigned int s_goal;
    if (tid == 0) {
        unsigned int g = goal_arr[bid] + 1u;
        goal_arr[bid] = g;
        s_goal = g;
    }
    __syncthreads();
    unsigned int goal = s_goal;

    // 到达
    if (tid == 0) {
        atomicExch((unsigned int*)&in_arr[bid], goal);
    }

    // 释放
    if (bid == 0 && tid == 0) {
        for (int i = 0; i < num_blocks; i++) {
            while (atomicAdd((unsigned int*)&in_arr[i], 0) != goal) { }
        }
        for (int i = 0; i < num_blocks; i++) {
            atomicExch((unsigned int*)&out_arr[i], goal);
        }
    }

    // 等待释放
    if (bid != 0 || tid != 0) {
        while (atomicAdd((unsigned int*)&out_arr[bid], 0) != goal) { }
    }
    __syncthreads();   // 保证块内所有线程在栅栏处对齐
}


template <int ITEMS_PER_THREAD, int TILE_SIZE>
__device__ void prefix_sum_impl(const int* __restrict__ in,
                                int* __restrict__ out,
                                int n,
                                unsigned int* tile_data,
                                int* counter,
                                unsigned int* goal_arr,
                                volatile unsigned int* in_arr,
                                volatile unsigned int* out_arr,
                                int num_blocks) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bdim = blockDim.x;            // 固定 256
    const int num_warps = bdim / 32;        // 8
    const int VEC = ITEMS_PER_THREAD / 4;   // int4 向量化因子

    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    if (num_tiles > MAX_TILES) num_tiles = MAX_TILES;
    if (num_tiles == 0) return;

    // 初始化：清零所有 tile 状态，重置计数器，并用栅栏保证完成
    for (int t = bid * bdim + tid; t < num_tiles; t += num_blocks * bdim) {
        tile_data[t] = 0;
    }
    if (bid == 0 && tid == 0) {
        *counter = 0;
    }
    __syncthreads();
    grid_barrier(goal_arr, in_arr, out_arr, num_blocks);

    // 共享内存：用于向量化加载/写回 + 块内扫描
    __shared__ int smem[TILE_SIZE];
    __shared__ int s_warp_sums[32];
    __shared__ int exclusive_prefix;
    __shared__ int s_tile_id;

    int4* smem4 = reinterpret_cast<int4*>(smem);

    // 主循环：动态领取 tile
    while (true) {
        if (tid == 0) {
            s_tile_id = atomicAdd(counter, 1);
        }
        __syncthreads();
        int tile_id = s_tile_id;
        if (tile_id >= num_tiles) break;

        int start = tile_id * TILE_SIZE;
        bool full_tile = (start + TILE_SIZE <= n);

        // 1. 向量化加载到共享内存
        if (full_tile) {
            const int4* in4 = reinterpret_cast<const int4*>(in + start);
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                smem4[i * bdim + tid] = in4[i * bdim + tid];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int idx   = i * bdim + tid;
                int g_idx = start + idx;
                smem[idx] = (g_idx < n) ? in[g_idx] : 0;
            }
        }
        __syncthreads();

        // 2. 从共享内存读取到寄存器（线程本地连续排列）
        int val[ITEMS_PER_THREAD];
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int4 q = smem4[tid * VEC + v];
            val[v * 4 + 0] = q.x;
            val[v * 4 + 1] = q.y;
            val[v * 4 + 2] = q.z;
            val[v * 4 + 3] = q.w;
        }

        // 3. 线程内 inclusive scan
        #pragma unroll
        for (int k = 1; k < ITEMS_PER_THREAD; ++k) {
            val[k] += val[k - 1];
        }
        int thread_sum = val[ITEMS_PER_THREAD - 1];

        // 4. warp 级 inclusive scan
        int warp_id = tid / 32;
        int lane_id = tid & 31;
        int warp_sum = thread_sum;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n_val = __shfl_up_sync(0xffffffff, warp_sum, offset);
            if (lane_id >= offset) warp_sum += n_val;
        }
        if (lane_id == 31) {
            s_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        // 5. 跨 warp 前缀
        if (warp_id == 0) {
            int w_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0;
            unsigned mask = (1 << num_warps) - 1;
            #pragma unroll
            for (int offset = 1; offset < num_warps; offset <<= 1) {
                int n_val = __shfl_up_sync(mask, w_sum, offset);
                if (lane_id >= offset) w_sum += n_val;
            }
            if (lane_id < num_warps) {
                s_warp_sums[lane_id] = w_sum - s_warp_sums[lane_id]; // 排他前缀
            }
        }
        __syncthreads();

        int warp_exclusive = (warp_id < num_warps) ? s_warp_sums[warp_id] : 0;
        int block_exclusive = warp_exclusive + (warp_sum - thread_sum);

        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            val[k] += block_exclusive;
        }

        if (tid == bdim - 1) {
            int local_tile_sum = val[ITEMS_PER_THREAD - 1];

            // 原来 tid == 0 的状态更新与 Look-back
            atomicExch(&tile_data[tile_id], MAKE_DATA(1, local_tile_sum));
            __threadfence();

            int ex = 0;
            int j = tile_id - 1;
            while (j >= 0) {
                unsigned int raw = atomicAdd(&tile_data[j], 0);
                int stat = STAT(raw);
                int agg  = AGG(raw);

                if (stat == 0) {
                    __threadfence_block();
                    continue;
                } else if (stat == 1) {
                    ex += agg;
                    j--;
                } else { // stat == 2
                    ex += agg;
                    break;
                }
            }
            exclusive_prefix = ex;

            atomicExch(&tile_data[tile_id], MAKE_DATA(2, ex + local_tile_sum));
            __threadfence();
        }
        __syncthreads();  // 保证所有线程看到 exclusive_prefix

        // 7. 写回结果（先写共享内存，再向量化存储）
        int my_ex = exclusive_prefix;
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            val[k] += my_ex;
        }

        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            int4 q;
            q.x = val[v * 4 + 0];
            q.y = val[v * 4 + 1];
            q.z = val[v * 4 + 2];
            q.w = val[v * 4 + 3];
            smem4[tid * VEC + v] = q;
        }
        __syncthreads();

        if (full_tile) {
            int4* out4 = reinterpret_cast<int4*>(out + start);
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                out4[i * bdim + tid] = smem4[i * bdim + tid];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int idx   = i * bdim + tid;
                int g_idx = start + idx;
                if (g_idx < n) out[g_idx] = smem[idx];
            }
        }
        __syncthreads(); // 确保写回完成，下一轮可安全复用 smem
    }
}

__global__ void student_prefix_sum_13block_kernel(const int* d_in, int* d_out, int n) {
    prefix_sum_impl<ITEMS_13, TILE_SIZE_13>(d_in, d_out, n,
        tile_data_13, &g_counter_13, g_goal_13, g_in_13, g_out_13, 13);
}

__global__ void student_prefix_sum_26block_kernel(const int* d_in, int* d_out, int n) {
    prefix_sum_impl<ITEMS_26, TILE_SIZE_26>(d_in, d_out, n,
        tile_data_26, &g_counter_26, g_goal_26, g_in_26, g_out_26, 26);
}