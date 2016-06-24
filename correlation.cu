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

// __global__ void cuComputeAvg(float *avg, float *matrix, int row, int col){
// 	int idx = blockDim.x*blockIdx.x +threadIdx.x;
// 	if(idx >= row){
// 		return;
// 	}
// 	float sum = 0;
// 	for(int i = 0; i< row; i++){
// 		sum += matrix[idx+ i*col];
// 	}

// 	avg[idx] = sum/row;

// 	return;
// }

__global__ void cuComputeAvg(float *avg, float *matrix, int row, int col){
	int idx = blockDim.x*blockIdx.x +threadIdx.x;
	if(idx >= row){
		return;
	}
	float sum = 0;
	for(int i = 0; i< col; i++){
		sum += matrix[idx*col + i];
	}

	avg[idx] = sum/col;

	return;
}

// __global__ void cuAverage(float *matrix, const float *avg, int row, int col){
// 	int idx = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(idx >= row*col){
// 		return;
// 	}
// 	int data_id = idx % col;
// 	//minus (n-1)
// 	matrix[idx] = (matrix[idx] - avg[data_id]);
// 	return;
// }

__global__ void cuAverage(float *matrix, const float *avg, int row, int col){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= row*col){
		return;
	}
	int data_id = idx/col;
	//minus avg
	matrix[idx] = (matrix[idx] - avg[data_id]);
	return;
}

int IDX2F(int i, int j, int ld){ 
	return (j)*(ld)+(i); 
}
int MIDX(int i, int j, int ld){
	return i*ld +j;
}
// Cublas
void matrix_mul(float *ans, const float *a, const float *b, const int m, const int k, const int n){
	int lda = m, ldb = m, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
				m, m, k, alpha, a, lda, b, ldb, beta, ans, ldc);
	cublasDestroy(handle);
	return;
}

int main(int argc, char **argv){
	float *h_m = (float *)malloc(sizeof(float)*ROWS*COLUMNS);

	for(int row = 0; row < ROWS ; row++){
		for(int col = 0 ; col < COLUMNS; col++){
			h_m[MIDX(row, col, COLUMNS)] = (float) (row*COLUMNS + col);
			printf("%d: %f\n", MIDX(row, col, COLUMNS), h_m[MIDX(row, col, COLUMNS)]);
		}
	}
	float *d_m, *d_s_m, *ans_m, *d_avg;

	cudaMalloc(&d_m, sizeof(float)*ROWS*COLUMNS);
	cudaMemcpy(d_m, h_m, sizeof(float)*ROWS*COLUMNS, cudaMemcpyHostToDevice);
	cudaMalloc(&d_s_m, sizeof(float)*ROWS*ROWS);
	cudaMalloc(&ans_m, sizeof(float)*ROWS*ROWS);
	cudaMalloc(&d_avg, sizeof(float)*ROWS);
	
	// // prepare and substract mean, divide (n-1)
	cuComputeAvg<<<(ROWS+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_avg, d_m, ROWS, COLUMNS);

	cuAverage<<<(ROWS*COLUMNS+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_m, d_avg, ROWS, COLUMNS);
	
	thrust::device_ptr<float> d_ptr(d_m);

	thrust::copy(d_ptr, d_ptr + ROWS*COLUMNS, std::ostream_iterator<float>(std::cout, "\n"));
	// call cublas
	float *t_d_m = d_m;
	matrix_mul(d_s_m, d_m, t_d_m, ROWS, COLUMNS, ROWS);
	
	printf("======\n");
	thrust::device_ptr<float> d_ptr2(d_s_m);
	thrust::copy(d_ptr2, d_ptr2+ROWS*ROWS, std::ostream_iterator<float>(std::cout, "\n"));
	
	//
	cudaFree(d_m);
	cudaFree(d_s_m);
	cudaFree(ans_m);
	cudaFree(d_avg);

	free(h_m);
	return 0;
}