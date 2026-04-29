// MPI_RUN_TEST1_NP: 4
// MPI_RUN_TEST2_NP: 8
// MPI_RUN_TEST3_NP: 12

#include <mpi.h>

#include <vector>

namespace {

void build_counts_and_displs(int n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);

    const int base = (size == 0) ? 0 : (n / size);
    const int rem = (size == 0) ? 0 : (n % size);

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

}  // namespace

// ------------------------------------------------------------
// naive 示例：函数内部自行完成分块、scatter/gather，并使用
// rank 链式传递偏移量。仅供参考，正确但性能较差。
// ------------------------------------------------------------
void student_prefix_sum(const int* h_in, int* h_out, int n, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<int> counts;
    std::vector<int> displs;
    build_counts_and_displs(n, size, counts, displs);

    const int local_n = counts[rank];
    std::vector<int> local_in(local_n);
    std::vector<int> local_out(local_n);

    MPI_Scatterv(rank == 0 ? h_in : NULL,
                 counts.data(),
                 displs.data(),
                 MPI_INT,
                 local_n > 0 ? local_in.data() : NULL,
                 local_n,
                 MPI_INT,
                 0,
                 comm);

    int local_sum = 0;
    for (int i = 0; i < local_n; ++i) {
        local_sum += local_in[i];
        local_out[i] = local_sum;
    }

    int offset = 0;
    if (size > 1) {
        if (rank == 0) {
            if (rank + 1 < size) {
                MPI_Send(&local_sum, 1, MPI_INT, rank + 1, 0, comm);
            }
        } else {
            int prefix_before_me = 0;
            MPI_Recv(&prefix_before_me, 1, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            offset = prefix_before_me;

            const int prefix_through_me = prefix_before_me + local_sum;
            if (rank + 1 < size) {
                MPI_Send(&prefix_through_me, 1, MPI_INT, rank + 1, 0, comm);
            }
        }
    }

    if (offset != 0) {
        for (int i = 0; i < local_n; ++i) {
            local_out[i] += offset;
        }
    }

    MPI_Gatherv(local_n > 0 ? local_out.data() : NULL,
                local_n,
                MPI_INT,
                rank == 0 ? h_out : NULL,
                counts.data(),
                displs.data(),
                MPI_INT,
                0,
                comm);
}
