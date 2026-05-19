remote:
	nvcc -std=c++11 ./grader/harness.cu ./student_kernel.cu -o ./grader/harness -gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61

local:
	nvcc -std=c++11 ./grader/harness.cu ./student_kernel.cu -o ./grader/harness -gencode arch=compute_86,code=sm_86

run:
	./grader/harness 1048576 10 13
	./grader/harness 1048576 10 26
	./grader/harness 167772160 10 13
	./grader/harness 167772160 10 26

push:
	cp student_kernel.cu ~/hw3-submissions/刘英哲_2300012753/

pull:
	cp -r ~/hw3-submissions ./

test:
	./test.sh