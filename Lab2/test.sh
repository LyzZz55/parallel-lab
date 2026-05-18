#!/bin/bash

# 设置基础目录
HW_DIR="./hw2-submissions"
GRADER_DIR="./grader"

# 检查 grader 目录是否存在
if [ ! -d "$GRADER_DIR" ]; then
    echo "错误: grader 目录不存在: $GRADER_DIR"
    exit 1
fi

# 检查 hw1-submission 目录是否存在
if [ ! -d "$HW_DIR" ]; then
    echo "错误: hw2-submission 目录不存在: $HW_DIR"
    exit 1
fi

# 遍历 hw1-submission 下的所有子文件夹
for student_dir in "$HW_DIR"/*/; do
    # 去除路径末尾的斜杠，获取文件夹名
    student_dir=${student_dir%/}
    student_name=$(basename "$student_dir")
    
    # 检查 student_kernel.cu 是否存在
    kernel_file="$student_dir/student_mpi_prefix_sum.cpp"
    if [ ! -f "$kernel_file" ]; then
        echo "警告: 在 $student_name 中未找到 student_mpi_prefix_sum.cpp，跳过..."
        continue
    fi
    
    echo "========================================="
    echo "处理学生: $student_name"
    echo "========================================="
    
    # 编译
    echo "编译中..."
    if mpicxx -O2 -std=c++11 grader/harness.cpp student_mpi_prefix_sum.cpp -o harness; then
        echo "编译成功"
    else
        echo "编译失败，跳过测试"
        continue
    fi
    
    echo "运行测试 - 参数: 167772160"
    mpirun -np 12 ./harness 167772160 5
    
    echo ""
done

echo "所有学生处理完成！"