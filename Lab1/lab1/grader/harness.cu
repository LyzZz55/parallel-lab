/* hrness.cu — 评测主程序
 *
 * 用法：./harness <N> <repeats>
 *   N       - 数组长度
 *   repeats - 计时重复次数（默认 100）
 *
 * 输出（单行，供 grade.py 解析）：
 *   PASS <avg_ms>
 *   FAIL <reason>
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 学生实现的接口声明
extern void student_prefix_sum(float* d_in, float* d_out, int n);

// ---- CUDA 错误检查宏 ----
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            printf("FAIL cuda_error\n");                                    \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ---- GPU 上逐元素比较，写出第一个错误的索引（sentinel=n 表示全部正确）----
__global__ void check_kernel(const float* out, const float* ref,
                              int n, float eps, int* err_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (fabsf(out[i] - ref[i]) > eps) {
        atomicMin(err_idx, i);
    }
}

int main(int argc, char** argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : 1048576;
    int repeats = (argc > 2) ? atoi(argv[2]) : 100;

    // ---- 生成随机输入（host）----
    float* h_in = new float[n];
    srand(42);
    for (int i = 0; i < n; ++i)
        h_in[i] = (float)(rand() % 100) / 10.0f;  // [0, 10)

    // ---- CPU 串行前缀和作为参考答案（inclusive scan）----
    float* h_ref = new float[n];
    h_ref[0] = h_in[0];
    for (int i = 1; i < n; ++i)
        h_ref[i] = h_ref[i - 1] + h_in[i];

    // ---- device 内存 ----
    float *d_in, *d_out, *d_ref;
    int   *d_err;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in,  h_in,  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref, n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Warmup ----
    student_prefix_sum(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 正确性验证（全在 GPU 上完成）----
    int sentinel = n;  // 初始值超出范围，表示"无错误"
    CUDA_CHECK(cudaMemcpy(d_err, &sentinel, sizeof(int), cudaMemcpyHostToDevice));

    const float eps = 1e-3f * n;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    check_kernel<<<blocks, threads>>>(d_out, d_ref, n, eps, d_err);
    CUDA_CHECK(cudaDeviceSynchronize());

    int err_idx;
    CUDA_CHECK(cudaMemcpy(&err_idx, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_idx < n) {
        float got, expected;
        CUDA_CHECK(cudaMemcpy(&got,      d_out + err_idx, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&expected, d_ref + err_idx, sizeof(float), cudaMemcpyDeviceToHost));
        printf("FAIL wrong_answer index=%d got=%.6f expected=%.6f\n",
               err_idx, got, expected);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_ref); cudaFree(d_err);
        delete[] h_in;
        delete[] h_ref;
        return 0;
    }

    // ---- 计时 ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r)
        student_prefix_sum(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / repeats;

    printf("PASS %.6f\n", avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_ref);
    cudaFree(d_err);
    delete[] h_in;
    delete[] h_ref;
    return 0;
}