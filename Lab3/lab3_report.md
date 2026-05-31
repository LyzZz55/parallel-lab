## 刘英哲 2300012753


- `student_prefix_sum_13block_kernel`：使用 **13 个线程块**，每线程处理 24 个元素，Tile 大小为 6144。
- `student_prefix_sum_26block_kernel`：使用 **26 个线程块**，每线程处理 16 个元素，Tile 大小为 4096。

两套实现共享同一个模板函数 `prefix_sum_impl`，通过模板参数 `ITEMS_PER_THREAD` 和 `TILE_SIZE` 区分行为。

---

## 优化技术

### 动态 Tile 分配（原子计数器分发）

传统前缀和常将输入按线程块数做静态划分。本实现使用**全局原子计数器 `counter`** 动态分配任务：

```cpp
if (tid == 0) {
    s_tile_id = atomicAdd(counter, 1);
}
__syncthreads();
int tile_id = s_tile_id;
if (tile_id >= num_tiles) break;
```

每个线程块在完成一个 Tile 后原子地获取下一个 Tile 的索引，天然支持任意数目线程块，且无需事先为每个块分配等长任务。

### 向量化全局内存访问（`int4` 加载与存储）

为最大化全局内存带宽利用率，代码将 Tile 数据以 `int4`（128 位）粒度进行读写：

- **加载**：若为完整 Tile，直接从 `in` 以 `const int4*` 形式向量化读入共享内存。
- **存储**：计算结果写回共享内存后，再以 `int4*` 形式向量化写回全局内存 `out`。

共享内存布局与线程的映射关系：`smem4[i * bdim + tid]` 保证同一次 `int4` 访问的线程间地址连续、且满足 128 位对齐要求，从而生成单次宽事务，减少指令发射次数并提高带宽利用率。

### 共享内存上的三级前缀扫描

1. **线程内 inclusive scan**  
   每线程将自己的 `ITEMS_PER_THREAD` 个元素做局部前缀求和，得出线程内总和 `thread_sum`。

2. **Warp 级 inclusive scan**  
   利用 `__shfl_up_sync` 进行 warp 内洗牌扫描，得到每个 warp 的局部总和 `warp_sum`。Warp 最后一个线程将总和写入共享内存数组 `s_warp_sums`。

3. **跨 Warp 前缀（block‑level scan）**  
   由 warp 0 的前 `num_warps` 个线程读取 `s_warp_sums`，再做一次洗牌扫描，计算出每个 warp 应加上的排他前缀 `warp_exclusive`。各 warp 再结合自己的 `warp_sum` 和 `thread_sum` 还原出线程级排他前缀 `block_exclusive`。

### 跨 Tile 前缀传播：Look‑back 与状态机

不同 Tile 之间的前缀依赖通过 **Look‑back** 技术解决，无需全局栅栏或递归调用。每个 Tile 计算完块内前缀后，由最后一个线程（`tid == bdim - 1`）负责处理跨 Tile 传播，其执行如下状态机：

```
状态 0 ：Tile 尚未计算或正在计算
状态 1 ：Tile 已完成本地前缀和，但尚未确定全局前缀
状态 2 ：Tile 的全局前缀已确定，并记录最终累加值
```

状态与聚合值通过宏 `MAKE_DATA` 打包进一个 `unsigned int`（高 2 位状态，低 30 位聚合值），并利用 `atomicExch` 写入全局数组 `tile_data`。

Look‑back 过程：
- 当前 Tile 计算完块内总和后，将自己的状态设为 **1**，并附加上局部总和。
- 随后向前遍历 `tile_id - 1`，通过 `atomicAdd(&tile_data[j], 0)` 原子读取前驱状态。
  - 若状态为 **0**（前驱尚未完成），则使用 `__threadfence_block` 后重试（忙等）。
  - 若状态为 **1**，累加其记录的局部总和，继续向前查找。
  - 若状态为 **2**，累加该记录值后立即终止，因为该 Tile 已包含所有前驱的全局前缀。
- 累加结果即为当前 Tile 的排他前缀 `exclusive_prefix`，将其广播给块内所有线程。
- 最后将该 Tile 状态更新为 **2**，并写入 **`exclusive_prefix + 局部总和`**，供后续 Tile 直接使用。

### 软件栅栏

初始化阶段需要保证所有线程块将 `tile_data` 清零、`counter` 复位后再开始动态分配。 `grid_barrier`利用原子操作来避免显式的 `__threadfence`：

- **到达 (`in_arr`)**：每个 block 的 thread 0 使用 `atomicExch` 写入本块的 `in_arr`。`atomicExch` 自带 **release** 语义，确保之前的所有内存写入（如清零操作）对后续观察者可见。
- **释放 (`out_arr`)**：block 0 的 thread 0 循环使用 `atomicAdd(&in_arr[i], 0)` 读取所有块的状态。该读取为原子操作，自带 **acquire** 语义，一旦看到对应值，即代表该块的所有写入已完成。随后通过 `atomicExch` 写各块的 `out_arr`（release 语义），通知所有块可以继续。
- **等待**：其余线程自旋于 `atomicAdd(&out_arr[bid], 0)` 等待对应释放信号，同样受益于 acquire 语义。


### 消除同步

除了 Look‑back 中必要的 `__threadfence` 保证状态写入的可见性外，其余同步均使用 `__syncthreads` 实现块内对齐。跨块依赖完全由原子操作和忙等解决，避免使用 `cudaDeviceSynchronize` 等全局同步手段，提升了可扩展性。

---

## 总体流程与数据流

```
初始化清零 tile_data / counter
  → 全局栅栏 (grid_barrier)
  → 主循环:
      原子获取 tile_id
      向量化加载到共享内存
      三级扫描 (线程内 → warp → 跨 warp)
      块内排他前缀
      Look‑back 查询/更新状态，得到 tile 排他前缀
      施加 tile 排他前缀后写回共享内存
      向量化存储到全局内存
      __syncthreads (准备复用共享内存)
  → 循环至所有 Tile 完成
```

