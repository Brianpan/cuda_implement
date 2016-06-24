#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cublas_v2.h>
#include <curand.h>

#include "Timer.h"

#define BLKSIZE 256
#define ROWS 2
#define COLUMNS 5 

///// cublas methods
__global__ void cuComputeAvg(double *avg, double *matrix, int row, int col){
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

__global__ void cuAverage(double *matrix, const double *avg, int row, int col){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= row*col){
		return;
	}
	int data_id = idx % col;
	//minus (n-1)
	matrix[idx] = (matrix[idx] - avg[data_id]);
	return;
}

__global__ void cuDivideSS(double *ans, double *ss, int row){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= ROWS*ROWS){
		return;
	}
	const int x_idx = idx % row;
	const int y_idx = idx / row;

	double denomitor_x = ss[x_idx*row + x_idx];
	double denomitor_y = ss[y_idx*row + y_idx];
	ans[idx] = ss[idx]/sqrtf(denomitor_x*denomitor_y);

	return;
}

int IDX2F(int i, int j, int ld){ 
	return (j)*(ld)+(i); 
}
int MIDX(int i, int j, int ld){
	return i*ld +j;
}
// Cublas
void matrix_mul(double *ans, const double *a, const double *b, const int m, const int k, const int n){
	int lda = m, ldb = m, ldc = m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
				m, m, k, alpha, a, lda, b, ldb, beta, ans, ldc);
	cublasDestroy(handle);
	return;
}

struct divide_val{
	int divideVal;
	divide_val(int _divideVal) : divideVal(_divideVal){}
	__host__ __device__ double operator()(const double val){
		return val/divideVal;
	}
};

__global__ void cuColumnBase(double *dest, double *data, const int rows, const int cols){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if(idx >= rows*cols){
		return;
	}
	const int x_idx = idx % cols;
	const int y_idx = idx / cols;

	dest[x_idx*rows+ y_idx] = data[idx];
	return;
}

void computeCov(double *ans_m, double *data, const int rows, const int cols){
	double *d_s_m, *d_avg, *d_dest;

	cudaMalloc(&d_dest, sizeof(double)*ROWS*COLUMNS);
	cudaMalloc(&d_s_m, sizeof(double)*rows*rows);
	cudaMalloc(&d_avg, sizeof(double)*rows);

	cuColumnBase<<<(rows*cols+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_dest, data, rows, cols);
	
	// prepare and substract mean, divide (n-1)
	cuComputeAvg<<<(rows+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_avg, d_dest, cols, rows);

	cuAverage<<<(rows*cols+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_dest, d_avg, cols, rows);
	cudaFree(d_avg);

	// thrust::device_ptr<float> d_ptr(d_dest);
	// thrust::copy(d_ptr, d_ptr + rows*cols, std::ostream_iterator<float>(std::cout, "\n"));
	
	// call cublas
	// columned-based
	// turn to vertical arrange
	matrix_mul(d_s_m, d_dest, d_dest, rows, cols, rows);
	
	cudaFree(d_dest);
	
	// divide by 1/n-1
	thrust::device_ptr<double> d_ptr2(d_s_m);
	thrust::transform(d_ptr2, d_ptr2+rows*rows, d_ptr2, divide_val(cols-1));
	// thrust::copy(d_ptr2, d_ptr2+ROWS*ROWS, std::ostream_iterator<float>(std::cout, "\n"));
	
	// do pairwise divide
	cuDivideSS<<<(rows*rows+BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(ans_m, d_s_m, rows);
	
	// thrust::device_ptr<double> d_aptr(ans_m);
	// thrust::copy(d_aptr, d_aptr+rows*rows, std::ostream_iterator<double>(std::cout, "\n"));

	cudaFree(d_s_m);

	return;
}

int main(int argc, char **argv){
	double *h_m = (double *)malloc(sizeof(double)*ROWS*COLUMNS);

	// for(int row = 0; row < ROWS ; row++){
	// 	for(int col = 0 ; col < COLUMNS; col++){
	// 		// h_m[MIDX(row, col, COLUMNS)] = (float) (row*COLUMNS + col);
	// 		// printf("%d: %f\n", MIDX(row, col, COLUMNS), h_m[MIDX(row, col, COLUMNS)]);
	// 		h_m[IDX2F(row, col, ROWS)] = (float) (row*COLUMNS + col);
	// 		printf("%d : %f \n", IDX2F(row, col, ROWS), h_m[IDX2F(row, col, ROWS)]);
	// 	}
	// }

	// float h_m2[10] = {0.3, 0.4, 0.1, 0.7, 0.5, 0.6, 0.2, 0.8, 0.9, 1.3};
	double h_m2[10] = {0.3, 0.1, 0.5, 0.2, 0.9, 0.4, 0.7, 0.6, 0.8, 1.3};
	memcpy(h_m, h_m2, sizeof(double)*10);
	
	double *d_m, *ans_m, *ans_m2;
	
	cudaMalloc(&d_m, sizeof(double)*ROWS*COLUMNS);
	
	// for(int i = 0 ; i < ROWS; i ++){
	// 	for(int j = 0 ; j < COLUMNS; j++){
	// 		h_m[i*COLUMNS+j] = rand();
	// 	}
	// }
	
	cudaMemcpy(d_m, h_m, sizeof(double)*ROWS*COLUMNS, cudaMemcpyHostToDevice);
	free(h_m);

	cudaMalloc(&ans_m, sizeof(double)*ROWS*ROWS);
	cudaMalloc(&ans_m2, sizeof(double)*ROWS*ROWS);
	// count time
	Timer cor_timer, cor_2_timer;
	cor_timer.Start();
	
	// generate random numbers
	computeCov(ans_m, d_m, ROWS, COLUMNS);
	
	cor_timer.Pause();
	
	printf_timer(cor_timer);
	// free memory
	cudaFree(d_m);
	cudaFree(ans_m);

	return 0;
}