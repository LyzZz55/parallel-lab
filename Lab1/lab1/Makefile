all:
	cd grader && nvcc -O2 -std=c++11 harness.cu ../student_kernel.cu -o harness

run:
	mkdir result
	cd grader && ./harness 1024 > ../result/1024.txt
	cd grader && ./harness 1048576 > ../result/1048576.txt
	cd grader && ./harness 167772160 > ../result/167772160.txt

