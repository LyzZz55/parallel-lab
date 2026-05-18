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
// 可在此处添加 __device__ 函数、全局状态变量和 kernel 定义。
//
// 注意：
//   - 必须分别实现 13-block 和 26-block 两个版本。
//   - 本文件只提供 __global__ kernel / __device__ 辅助函数 / device 全局状态。
//   - 不得在本文件中调用 cudaMalloc/cudaMemcpy/cudaDeviceSynchronize 等运行时 API。
//   - 推荐使用 persistent blocks + kernel 内跨 block 同步完成全局 scan。
// ------------------------------------------------------------



// ------------------------------------------------------------
// naive 示例：固定 block 数 / 256 threads 的单 kernel 顺序扫描。
// 该示例只用于说明接口和 launch 形式，性能极差。
// 请用高效并行实现替换下面两个 kernel 的函数体。
// ------------------------------------------------------------
__global__ void student_prefix_sum_13block_kernel(const int* in, int* out, int n) {
    // 只用 1 个线程顺序计算，等价于 CPU 实现；请勿把它作为最终答案提交。
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n <= 0) return;
        out[0] = in[0];
        for (int i = 1; i < n; ++i) {
            out[i] = out[i - 1] + in[i];
        }
    }
}

__global__ void student_prefix_sum_26block_kernel(const int* in, int* out, int n) {
    // 只用 1 个线程顺序计算，等价于 CPU 实现；请勿把它作为最终答案提交。
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n <= 0) return;
        out[0] = in[0];
        for (int i = 1; i < n; ++i) {
            out[i] = out[i - 1] + in[i];
        }
    }
}
