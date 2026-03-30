all:
	nvcc -O2 -std=c++11 ./grader/harness.cu ./student_kernel.cu -o ./harness

run:
	rm -r result
	mkdir result
	./harness 1024 1000
	./harness 1048576 1000
	./harness 167772160 1000

