all:
	nvcc -O2 -std=c++11 ./grader/harness.cu ./student_kernel.cu -o harness

run:
	rm -r result
	mkdir result
	./harness 1024 > ./result/1024.txt
	./harness 1048576 > ./result/1048576.txt
	./harness 167772160 > ./result/167772160.txt

