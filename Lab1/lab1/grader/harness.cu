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
#include <random>
#include <cuda_runtime.h>

// ---- double 精度参考 scan（Hillis-Steele 块内 + 递归块间）----
#define REF_BLOCK 1024

__global__ void ref_block_scan(const float* in, double* out,
                                double* block_sums, int n)
{
    __shared__ double s[REF_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * REF_BLOCK + tid;

    s[tid] = (gid < n) ? (double)in[gid] : 0.0;
    __syncthreads();

    for (int stride = 1; stride < REF_BLOCK; stride <<= 1) {
        double add = (tid >= stride) ? s[tid - stride] : 0.0;
        __syncthreads();
        s[tid] += add;
        __syncthreads();
    }

    if (gid < n) out[gid] = s[tid];
    if (tid == REF_BLOCK - 1) block_sums[blockIdx.x] = s[REF_BLOCK - 1];
}

__global__ void ref_block_scan_d(const double* in, double* out,
                                  double* block_sums, int n)
{
    __shared__ double s[REF_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * REF_BLOCK + tid;

    s[tid] = (gid < n) ? in[gid] : 0.0;
    __syncthreads();

    for (int stride = 1; stride < REF_BLOCK; stride <<= 1) {
        double add = (tid >= stride) ? s[tid - stride] : 0.0;
        __syncthreads();
        s[tid] += add;
        __syncthreads();
    }

    if (gid < n) out[gid] = s[tid];
    if (tid == REF_BLOCK - 1) block_sums[blockIdx.x] = s[REF_BLOCK - 1];
}

__global__ void ref_add_sums(double* out, const double* sums, int n)
{
    int gid = blockIdx.x * REF_BLOCK + threadIdx.x;
    if (blockIdx.x > 0 && gid < n) out[gid] += sums[blockIdx.x - 1];
}

// 递归：对 double 数组做 inclusive scan（原地）
static void ref_scan_d(double* arr, int n)
{
    int nb = (n + REF_BLOCK - 1) / REF_BLOCK;
    double* bsums; cudaMalloc(&bsums, nb * sizeof(double));
    double* tmp;   cudaMalloc(&tmp,   n  * sizeof(double));

    ref_block_scan_d<<<nb, REF_BLOCK>>>(arr, tmp, bsums, n);
    cudaMemcpy(arr, tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(tmp);

    if (nb > 1) {
        ref_scan_d(bsums, nb);
        ref_add_sums<<<nb, REF_BLOCK>>>(arr, bsums, n);
    }
    cudaFree(bsums);
}

static void ref_scan(const float* in, double* out, int n)
{
    int nb = (n + REF_BLOCK - 1) / REF_BLOCK;
    double* bsums; cudaMalloc(&bsums, nb * sizeof(double));

    ref_block_scan<<<nb, REF_BLOCK>>>(in, out, bsums, n);

    if (nb > 1) {
        ref_scan_d(bsums, nb);
        ref_add_sums<<<nb, REF_BLOCK>>>(out, bsums, n);
    }
    cudaFree(bsums);
}

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


int main(int argc, char** argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : 1048576;
    int repeats = (argc > 2) ? atoi(argv[2]) : 100;

    // ---- 生成随机输入（host）----
    float* h_in = new float[n];
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for (int i = 0; i < n; ++i)
        h_in[i] = dist(rng);

    // ---- device 内存 ----
    float  *d_in, *d_out;
    double *d_ref;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- GPU double 精度参考答案（并行 Hillis-Steele + 递归块间修正）----
    ref_scan(d_in, d_ref, n);

    // ---- Warmup ----
    student_prefix_sum(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 正确性验证：相对误差 ≤ 1e-3 ----
    float*  h_out   = new float[n];
    double* h_ref_d = new double[n];
    CUDA_CHECK(cudaMemcpy(h_out,   d_out, n * sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref_d, d_ref, n * sizeof(double), cudaMemcpyDeviceToHost));

    // float 前缀和允许相对误差 ~sqrt(n)*eps_float，用 1e-3 作为宽松阈值
    const double rel_tol = 1e-1;
    int err_idx = -1;
    for (int i = 0; i < n; ++i) {
        double ref = h_ref_d[i];
        double rel_err = fabs((double)h_out[i] - ref) / ref;
        if (rel_err > rel_tol) {
            err_idx = i;
            break;
        }
    }
    if (err_idx >= 0) {
        printf("FAIL wrong_answer index=%d got=%.6f expected=%.6f\n",
               err_idx, h_out[err_idx], (float)h_ref_d[err_idx]);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_ref);
        delete[] h_in;
        delete[] h_ref_d;
        delete[] h_out;
        return 0;
    }
    delete[] h_out;
    delete[] h_ref_d;

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
    return 0;
}