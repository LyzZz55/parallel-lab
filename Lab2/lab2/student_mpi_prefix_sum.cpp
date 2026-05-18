// MPI_RUN_TEST1_NP: 12
// MPI_RUN_TEST2_NP: 12
// MPI_RUN_TEST3_NP: 12


// 刘英哲 2300012753
#include <mpi.h>
#include <vector>
#include <cstddef>

namespace {

// 计算每个进程的数据块大小和起始偏移量
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


void student_prefix_sum(const int* h_in, int* h_out, int n, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 数据划分
    std::vector<int> counts, displs;
    build_counts_and_displs(n, size, counts, displs);

    const int local_n = counts[rank];
    std::vector<int> local_in(local_n);
    std::vector<int> local_out(local_n);

    // 分发数据
    MPI_Scatterv(rank == 0 ? h_in : nullptr,
                 counts.data(), displs.data(), MPI_INT,
                 local_n > 0 ? local_in.data() : nullptr, local_n, MPI_INT,
                 0, comm);

    // 本地前缀和
    int local_total = 0;
    for (int i = 0; i < local_n; ++i) {
        local_total += local_in[i];
        local_out[i] = local_total;
    }

    // 收集所有进程的本地总和到root
    std::vector<int> all_totals; 
    int my_total = local_total;
    if (rank == 0) {
        all_totals.resize(size);
    }
    MPI_Gather(&my_total, 1, MPI_INT,
               rank == 0 ? all_totals.data() : nullptr, 1, MPI_INT,
               0, comm);

    // root计算每个进程的偏移量
    int offset = 0;         
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

    // 将偏移量加到本地前缀和上
    if (offset != 0) {
        for (int i = 0; i < local_n; ++i) {
            local_out[i] += offset;
        }
    }

    // 收集结果回root
    MPI_Gatherv(local_n > 0 ? local_out.data() : nullptr, local_n, MPI_INT,
                rank == 0 ? h_out : nullptr, counts.data(), displs.data(), MPI_INT,
                0, comm);
}