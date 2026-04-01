/* harness.cu — 整型前缀和评测程序（CPU 串行参考）
 *
 * 用法：./harness <N> <repeats>
 *   N       - 数组长度
 *   repeats - 计时重复次数（默认 100）
 *
 * 输出（单行，供 grade.py 解析）：
 *   PASS <avg_ms>
 *   FAIL wrong_answer index=...
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

// ---- 学生实现的接口声明 ----
extern void student_prefix_sum(const int* d_in, int* d_out, int n);

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

int main(int argc, char** argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : 1048576;
    int repeats = (argc > 2) ? atoi(argv[2]) : 100;

    // ---- 生成随机输入（host），范围 0~10 ----
    int* h_in = new int[n];
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 10);
    for (int i = 0; i < n; ++i)
        h_in[i] = dist(rng);

    // ---- CPU 串行计算参考结果----
    int* h_ref = new int[n];
    h_ref[0] = h_in[0];
    for (int i = 1; i < n; ++i)
        h_ref[i] = h_ref[i - 1] + h_in[i];

    // ---- device 内存 ----
    int *d_in, *d_out, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ref, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in,  h_in,  n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, h_ref, n * sizeof(int), cudaMemcpyHostToDevice));

    // ---- Warmup ----
    student_prefix_sum(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 正确性检验 ----
    int* h_out = new int[n];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    int err_idx = -1;
    for (int i = 0; i < n; ++i) {
        if (h_out[i] != h_ref[i]) {
            err_idx = i;
            break;
        }
    }
    if (err_idx >= 0) {
        printf("FAIL wrong_answer index=%d got=%d expected=%d\n",
               err_idx, h_out[err_idx], h_ref[err_idx]);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_ref);
        delete[] h_in;
        delete[] h_ref;
        delete[] h_out;
        return 0;
    }
    delete[] h_out;

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
    delete[] h_in;
    delete[] h_ref;
    return 0;
}