/* harness.cpp — MPI 整型前缀和评测程序（CPU 串行参考）
 *
 * 用法：mpirun -np <P> ./harness <N> <repeats>
 *   P       - MPI 进程数
 *   N       - 全局数组长度
 *   repeats - 计时重复次数（默认 100）
 *
 * 输出（单行，供脚本解析）：
 *   PASS <avg_ms>
 *   FAIL wrong_answer index=... got=... expected=...
 */

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---- 学生实现的接口声明 ----
extern void student_prefix_sum(const int* h_in, int* h_out, int n, MPI_Comm comm);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = (argc > 1) ? std::atoi(argv[1]) : 1048576;
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 100;

    std::vector<int> h_in;
    std::vector<int> h_ref;
    std::vector<int> h_out;

    if (rank == 0) {
        h_in.resize(n);
        h_ref.resize(n);
        h_out.resize(n);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 10);
        for (int i = 0; i < n; ++i) {
            h_in[i] = dist(rng);
        }

        if (n > 0) {
            h_ref[0] = h_in[0];
            for (int i = 1; i < n; ++i) {
                h_ref[i] = h_ref[i - 1] + h_in[i];
            }
        }
    }

    // ---- Warmup ----
    student_prefix_sum(rank == 0 ? h_in.data() : NULL,
                       rank == 0 ? h_out.data() : NULL,
                       n,
                       MPI_COMM_WORLD);

    // ---- 正确性检验 ----
    int local_fail = 0;
    int err_idx = -1;
    int got_value = 0;
    int expected_value = 0;

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != h_ref[i]) {
                local_fail = 1;
                err_idx = i;
                got_value = h_out[i];
                expected_value = h_ref[i];
                break;
            }
        }
    }

    MPI_Bcast(&local_fail, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (local_fail) {
        if (rank == 0) {
            std::printf("FAIL wrong_answer index=%d got=%d expected=%d\n",
                        err_idx, got_value, expected_value);
        }
        MPI_Finalize();
        return 0;
    }

    // ---- 计时 ----
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int r = 0; r < repeats; ++r) {
        student_prefix_sum(rank == 0 ? h_in.data() : NULL,
                           rank == 0 ? h_out.data() : NULL,
                           n,
                           MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double local_ms = (end - start) * 1000.0 / repeats;
    double avg_ms = 0.0;
    MPI_Reduce(&local_ms, &avg_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("PASS %.6f\n", avg_ms);
    }

    MPI_Finalize();
    return 0;
}
