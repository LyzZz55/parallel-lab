all:
	nvcc -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -o hello hello.cu

run:
	CUDA_VISIBLE_DEVICES=0 ./hello