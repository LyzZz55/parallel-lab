// MPI_RUN_TEST1_NP: 12
// MPI_RUN_TEST2_NP: 12
// MPI_RUN_TEST3_NP: 12

#include <mpi.h>
#include <vector>
#include <cstddef>

namespace {

// 计算每个进程的数据块大小和起始偏移量（用于Scatterv/Gatherv）
void build_counts_and_displs(int n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);

    if (size == 0) return;
    const int base = n / size;
    const int rem = n % size;

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

} // anonymous namespace

/**
 * 并行前缀和（inclusive prefix sum）实现
 * 算法步骤：
 * 1. 使用MPI_Scatterv将输入数组分发给所有进程（非root可传NULL）
 * 2. 每个进程计算本地前缀和（inclusive），同时记录本地总和
 * 3. 通过MPI_Gather收集所有本地总和到root进程
 * 4. root进程计算每个进程的全局偏移量（exclusive prefix sum of local sums）
 * 5. 通过MPI_Scatter将偏移量分发回所有进程
 * 6. 每个进程将本地前缀和加上偏移量，得到最终结果
 * 7. 使用MPI_Gatherv将各进程的结果收集回root进程（其他进程传NULL）
 *
 * 性能优势：
 * - 通信次数为常数（2次全局集合通信），与进程数无关
 * - 避免了线性链式通信的O(P)串行延迟
 * - MPI_Gather/Scatter内部使用树形/优化拓扑，对中小规模P非常高效
 * - 正确处理n < size、局部块为空等边界情况
 */
void student_prefix_sum(const int* h_in, int* h_out, int n, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. 数据划分
    std::vector<int> counts, displs;
    build_counts_and_displs(n, size, counts, displs);

    const int local_n = counts[rank];
    std::vector<int> local_in(local_n);
    std::vector<int> local_out(local_n);

    // 2. 散布数据（root提供h_in，其他为NULL）
    MPI_Scatterv(rank == 0 ? h_in : nullptr,
                 counts.data(), displs.data(), MPI_INT,
                 local_n > 0 ? local_in.data() : nullptr, local_n, MPI_INT,
                 0, comm);

    // 3. 本地前缀和，同时计算本地总和
    int local_total = 0;
    for (int i = 0; i < local_n; ++i) {
        local_total += local_in[i];
        local_out[i] = local_total;
    }

    // 4. 收集所有进程的本地总和到root
    std::vector<int> all_totals;  // 仅在root有效
    int my_total = local_total;
    if (rank == 0) {
        all_totals.resize(size);
    }
    MPI_Gather(&my_total, 1, MPI_INT,
               rank == 0 ? all_totals.data() : nullptr, 1, MPI_INT,
               0, comm);

    // 5. root计算每个进程的全局偏移量（exclusive prefix of local totals）
    int offset = 0;          // 当前进程的偏移量
    if (rank == 0) {
        std::vector<int> offsets(size, 0);
        int running = 0;
        for (int i = 0; i < size; ++i) {
            offsets[i] = running;
            running += all_totals[i];
        }
        // 将偏移量分发给所有进程
        MPI_Scatter(offsets.data(), 1, MPI_INT,
                    &offset, 1, MPI_INT,
                    0, comm);
    } else {
        MPI_Scatter(nullptr, 1, MPI_INT,
                    &offset, 1, MPI_INT,
                    0, comm);
    }

    // 6. 将偏移量加到本地前缀和上
    if (offset != 0) {
        for (int i = 0; i < local_n; ++i) {
            local_out[i] += offset;
        }
    }

    // 7. 收集结果回root
    MPI_Gatherv(local_n > 0 ? local_out.data() : nullptr, local_n, MPI_INT,
                rank == 0 ? h_out : nullptr, counts.data(), displs.data(), MPI_INT,
                0, comm);
}