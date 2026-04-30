all:
	mpicxx -O2 -std=c++11 grader/harness.cpp student_mpi_prefix_sum.cpp -o harness

run:
	mpirun -np 12 ./harness 10240 5
	mpirun -np 12 ./harness 10485760 5
	mpirun -np 12 ./harness 167772160 5

push:
	cp student_mpi_prefix_sum.cpp ~/hw2-submissions/刘英哲_2300012753/

pull:
	cp -r ~/hw2-submissions ./

test:
	./test.sh