#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cublas_v2.h>

#define BLKSIZE 256
#define ROWS 2
#define COLUMNS 5 
#define IDX2F(i, j, ld)((j)*ld+(i))

__global__ void cuComputeAvg(float *avg, float *matrix, int row, int col){
	int idx = blockDim.x*blockIdx.x +threadIdx.x;
	if(idx >= row){
		return;
	}
	float sum = 0;
	for(int i = 0; i< row; i++){
		sum += matrix[idx+ i*col];
	}

	avg[idx] = sum/row;

	return;
}

__global__ void cuMinusAverage(float *matrix, const float *avg, int row, int col){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= row*col){
		return;
	}
	int data_id = idx % col;
	//minus (n-1) and divided by (data size - 1)
	matrix[idx] = (matrix[idx] - avg[data_id])/(row-1);
	return;
}

int main(){
	float *h_matrix = (float *) malloc(sizeof(float)*ROWS*COLUMNS);

	for(int row = 0; row < ROWS ; row++){
		for(int col = 0 ; col < COLUMNS; col++){
			h_matrix[IDX2F(row, col, COLUMNS)] = (float) row*COLUMNS + col;
		}
	}

	float *d_matrix, *d_s_matrix, *ans_matrix, *d_avg;
	cudaMalloc(&d_matrix, sizeof(float)*ROWS*COLUMNS);
	cudaMemcpy(d_matrix, h_matrix, sizeof(float)*ROWS*COLUMNS, cudaMemcpyHostToDevice);
	cudaMalloc(&d_s_matrix, sizeof(float)*COLUMNS*COLUMNS);
	cudaMalloc(&ans_matrix, sizeof(float)*COLUMNS*COLUMNS);
	cudaMalloc(&d_avg, sizeof(float)*ROWS);
	// prepare and substract mean, divide (n-1)
	cuComputeAvg<<<(ROWS+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_avg, d_matrix, COLUMNS, ROWS);
	cuMinusAverage<<<(ROWS*COLUMNS+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_matrix, d_avg, COLUMNS, ROWS);
	
	thrust::device_ptr<float> d_ptr(d_matrix);

	thrust::copy(d_ptr, d_ptr + ROWS*COLUMNS, std::ostream_iterator<float>(std::cout, "\n"));
	// call cublas

	cudaFree(d_matrix);
	cudaFree(d_s_matrix);
	cudaFree(ans_matrix);
	cudaFree(d_avg);

	free(h_matrix);
}