qsort:
	nvcc -arch=sm_50 -std=c++11 -lcurand -rdc=true quicksort_thrust.cu -o qsort.o	
cor:
	nvcc -arch=sm_30 -std=c++11 -lcublas -lcurand correlation.cu -o cor.o
clean:
	rm *.o	