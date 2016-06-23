qsort:
	nvcc -arch=sm_50 -lcurand -rdc=true quicksort_thrust.cu -o qsort.o	
