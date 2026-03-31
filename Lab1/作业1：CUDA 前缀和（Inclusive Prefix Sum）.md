# 作业1：CUDA 前缀和（Inclusive Prefix Sum）

## 作业背景

前缀和（Prefix Sum / Scan）是并行计算中最基础也最重要的原语之一，被广泛用于排序、压缩、稀疏矩阵运算等场景。本次作业要求你在 GPU 上实现一个**高效的 inclusive prefix sum kernel**。助教会在自动评测系统中将同学们的实现进行性能排名。

---

## 问题定义

给定长度为 `n` 的整型数组 `A`，计算其 inclusive prefix sum 数组 `B`：

```
B[0] = A[0]
B[1] = A[0] + A[1]
B[2] = A[0] + A[1] + A[2]
...
B[i] = A[0] + A[1] + ... + A[i]
```

### 示例

**输入：**
```
A = [3, 1, 4, 1, 5, 9, 2, 6]
```

**输出：**
```
B = [3, 4, 8, 9, 14, 23, 25, 31]
```

---

## 接口要求

你需要实现以下函数（**函数签名不得修改**）：

```cuda
void student_prefix_sum(int* d_in, int* d_out, int n);
```

| 参数 | 说明 |
|------|------|
| `d_in` | device 端输入数组，长度为 `n`，类型 `int` |
| `d_out` | device 端输出数组，长度为 `n`，类型 `int` |
| `n` | 数组元素个数，**不保证是 2 的幂次** |

> `d_in` 和 `d_out` 均已由评测系统分配好 device 内存，你只需完成计算，无需 `cudaMalloc` / `cudaMemcpy`。

---

## 测试用例

评测系统会依次用以下三个规模测试你的实现：

| 测试编号 | 数组长度 N | 约等于 | 计时重复次数 |
|----------|-----------|--------|-------------|
| Test 1 | 1,024 | 1K | 10 次 |
| Test 2 | 1,048,576 | 1M | 10 次 |
| Test 3 | 167,772,160 | 160M | 10 次 |

- 输入数据为随机 `int`，范围 `[0, 10)`，固定随机种子（seed=42）
- 正确性判断：与参考结果逐元素进行精准对比
- **排名依据 Test 3（160M 规模）的平均执行时间**，时间越短排名越靠前 (评测时运行的GPU为 Tesla K20c )

---

## 评测结果说明

| 状态 | 含义 |
|------|------|
| `PASS` | 正确性通过，显示各规模平均执行时间（ms） |
| `WRONG_ANSWER` | 输出结果与参考答案不符，不参与排名 |
| `COMPILE_ERROR` | 代码无法编译，不参与排名 |
| `TLE` | 单规模运行超过 60 秒，不参与排名 |
| `RUNTIME_ERROR` | 运行时崩溃（如非法内存访问），不参与排名 |

---

## 允许与禁止

**允许：**
- 在 `student_kernel.cu` 中添加任意数量的辅助 `__global__` kernel 和 `__device__` 函数
- 使用 `cudaMalloc` 分配临时 device 内存（记得在函数返回前 `cudaFree`）
- 使用 shared memory、warp intrinsics（`__shfl_down_sync` 等）
- 使用多个 kernel launch（分块 scan 需要多个 pass）
- 只能修改 `student_kernel.cu` 文件

**禁止：**

- 修改 `student_prefix_sum` 的函数签名
- 使用 Thrust、CUB、cuBLAS 等高级库
- 修改除 `student_kernel.cu` 以外的任何文件
- 在 `student_prefix_sum` 内部调用 `cudaMemcpy`（host/device 之间）
- 不能修改除 `student_kernel.cu` 的其他文件


## 提交要求

1. 代码：

**仅一个代码文件**：（请一定确保你的文件可以通过`nvcc -O2 -std=c++11 harness.cu [/path/student_kernel.cu] -o harness`方式编译成功）

```
student_kernel.cu
```

将文件放入共享服务器的相应位置，并以**你的中文姓名/学号**命名的目录中：

```
/home/pppd-stu/hw1-submissions/
  张三_2024001/student_kernel.cu
  李四_2024002/student_kernel.cu
```

> 子目录名即为排行榜上显示的名字，请使用"姓名\_学号"格式。

注意超过截止时间放入文件夹的代码不会被计入分数。

2. 优化策略详细解析（pdf文档）

对所实现优化策略的文字说明，需涵盖核心思路与关键设计决策，简明扼要。**须以 PDF 格式提交至教学网。**


### 注意事项

1. **文件名必须是 `student_kernel.cu`**，大小写敏感
2. 提交前请在本地确认代码能通过编译（`nvcc -O2 -std=c++11 harness.cu [/path/student_kernel.cu] -o harness` 不报错）
3. 确保 `student_prefix_sum` 函数签名与模板完全一致
4. 不要在代码中 `#include` 评测系统内部头文件（如 `harness.cu`）
5. 如使用临时 device 内存，务必在函数结束前 `cudaFree`，否则大规模测试可能 OOM

---

## 本地自测方法

如果你有 CUDA 环境，可以在提交前本地验证。

在本地或其他非课程机器上进行编译时，请自行查询你的硬件设备的计算能力，并据此调整编译指令。具体见下一节。

### nvcc 直接编译

```bash
# 在 grader/ 目录下执行，将 ../student_kernel.cu 替换为你的文件路径
nvcc -std=c++11 harness.cu ../student_kernel.cu -o harness -gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61

# 运行
./harness 1024
./harness 1048576
./harness 167772160
```

在进行本地测试时，以`NVIDIA GeForce RTX 3060 Laptop GPU`为例，其计算能力为`8.6`, 因此需要将编译指令变更为

```bash
nvcc -std=c++11 harness.cu ../student_kernel.cu -o harness -gencode arch=compute_86,code=sm_86
```

当然，为了简单起见，我们提供了一个简单的`Makefile`文件。你可以直接在`student_kernel.cu`所在目录下
```bash
# 编译
make

# 运行
make run
```


### 输出示例

```
PASS 4.823100
```

---

## 评分标准

| 评分项 | 占比 | 说明 |
|--------|------|------|
| 正确性 | 60% | 三个规模全部 PASS 得满分 |
| 性能排名 | 40% | 按 160M 规模执行时间排名 |

> 无法通过正确性验证（`WRONG_ANSWER` / `COMPILE_ERROR` / `TLE`）的提交，性能分为 0。

---

## 截止时间: 北京时间2026年4月17日晚上11点59分

**请在提交系统关闭前完成提交，逾期不予受理。**


---
## 提示

由于`float`浮点计算中累加顺序不同会带来较大误差，要求从`float`浮点型改为了`int`整型。大部分情况下已完成的代码可以简单地将代码中的`float`, `double`替换为`int`，然后重新进行测试即可。

如有疑问请在课程群中提问，或课后找助教答疑。