remote:
	nvcc -std=c++11 ./grader/harness.cu ./student_kernel.cu -o ./grader/harness -gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61

local:
	nvcc -std=c++11 ./grader/harness.cu ./student_kernel.cu -o ./grader/harness -gencode arch=compute_86,code=sm_86

run:
	./grader/harness 1024
	./grader/harness 1048576
	./grader/harness 167772160

