#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include <curand.h>

#define FEATURES 30
#define MSIZE 2
#define BLKSIZE 256

template<class T>
__device__ void cuSelectsort(T *data, int lpt, int rpt){
	for(int pivot = lpt; pivot <= rpt; pivot++){
		T min_val = data[pivot];
		int min_idx = pivot;
		for(int j = pivot+1; j <= rpt; j++){
			if(min_val > data[j]){
				min_idx = j;
				min_val = data[j];
			}
		}

		if(min_idx != pivot){
			T tmp = data[pivot];
			data[pivot] = data[min_idx];
			data[min_idx] = tmp;
		}
	}
	return;
}

template<class T>
__global__ void cuQuicksort(T *data, int lpt, int rpt, int shell_no){
	
	if(shell_no > 4 || rpt - lpt <= 3){
		cuSelectsort(data, lpt, rpt);
		return;
	}

	// pivoting
	T *l_ptr = data+lpt;
	T *r_ptr = data+rpt;
	T pivot = data[(lpt+rpt)/2];

	while(l_ptr <= r_ptr){
		T l_val = *l_ptr;
		T r_val = *r_ptr;
		while(*l_ptr <= pivot){
			l_ptr++;
			l_val = *l_ptr;
		}
		while(*r_ptr >= pivot){
			r_ptr--;
			r_val = *r_ptr;
		}

		//swap
		if(r_ptr >= l_ptr){
			*l_ptr++ = r_val;
			*r_ptr-- = l_val; 
		}
	}
	// do dynamic parallelism
	int nright = r_ptr - data;
	int nleft = l_ptr - data;

	if(nright > lpt){
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		cuQuicksort<<<1, 1, 0, s1>>>(data, lpt, nright, shell_no+1);

		cudaStreamDestroy(s1);
	}
	if(nleft < rpt){
		cudaStream_t s2;
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		cuQuicksort<<<1, 1, 0, s2>>>(data, nleft, rpt, shell_no+1);
		
		cudaStreamDestroy(s2);
	}

	return;
}

template<class T>
__global__ void quickSortCuda(T *data, int sort_size, int features){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= sort_size){
		return;
	}

	cuQuicksort<<<1, 1>>>(data+features*idx, 0, features-1, 0);
	cudaDeviceSynchronize();

	return;
}
int mod_s = 0;
int m_idx(){
	return (mod_s++)/FEATURES;
}
template<class T>
void thrustSort(T *data, int sort_size, int features){
	thrust::device_ptr<T> d_ptr(data);
	thrust::device_vector<T> d_vec(d_ptr, d_ptr+FEATURES*MSIZE);
	thrust::host_vector<int> h_rank(FEATURES*MSIZE);
	mod_s = 0;
	thrust::generate(h_rank.begin(), h_rank.end(), m_idx);
	thrust::device_vector<int> d_rank = h_rank;

	thrust::stable_sort_by_key(d_vec.begin(), d_vec.end(), d_rank.begin());
	cudaDeviceSynchronize();
	thrust::stable_sort_by_key(d_rank.begin(), d_rank.end(), d_vec.begin());
	cudaDeviceSynchronize();
	// thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<T>(std::cout, "\n"));

	cudaMemcpy(data, thrust::raw_pointer_cast(d_vec.data()), sizeof(T)*FEATURES*MSIZE, cudaMemcpyDeviceToDevice);

	return;
}

template<class T>
bool validate(T *sorted1, T *sorted2){
	thrust::device_ptr<T> s1_ptr(sorted1), s2_ptr(sorted2);
	return thrust::equal(s1_ptr, s1_ptr+FEATURES*MSIZE, s2_ptr);
}

int main(int argc, char **argv){
	float *d_m, *d_t;
	cudaMalloc(&d_m, sizeof(float)*FEATURES*MSIZE);
	cudaMalloc(&d_t, sizeof(float)*FEATURES*MSIZE);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for(int iter = 0 ; iter < 1; iter++){
		curandSetPseudoRandomGeneratorSeed(gen, rand()/1000);
		curandGenerateUniform(gen, d_m, FEATURES*MSIZE);

		cudaMemcpy(d_t, d_m, sizeof(float)*FEATURES*MSIZE,cudaMemcpyDeviceToDevice);

		// quicksort
		quickSortCuda<<<MSIZE+BLKSIZE-1/BLKSIZE, BLKSIZE>>>(d_m, MSIZE, FEATURES);
		cudaDeviceSynchronize();
		
		// thrust
		thrustSort(d_t, MSIZE, FEATURES);
		// test successful or not
		bool success = validate(d_t, d_m);
		printf("success: %d \n", success);
		// thrust::device_ptr<float> m_ptr(d_m);
		// thrust::copy(m_ptr, m_ptr+FEATURES*MSIZE, std::ostream_iterator<float>(std::cout, "\n"));
	}
	cudaFree(d_m);
	cudaFree(d_t);
	curandDestroyGenerator(gen);
	return 0;
}