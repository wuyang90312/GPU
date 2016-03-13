#include <stdio.h>
#include <time.h>

const int M = 1024 * 1024;
const int thread_per_block = 512;

#define time_record_begin(start){ \
	cudaEventCreate(&start);	  \
	cudaEventRecord(start, 0);	  \
}
#define time_record_end(start, stop, time){ \
	cudaEventCreate(&stop);		\
	cudaEventRecord(stop, 0);	\
	cudaEventSynchronize(stop);	\
	cudaEventElapsedTime(&time, start, stop);\
}

__global__ void findMin(int *A){

    __shared__ int sdata[512];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = A[i];
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 32; s /= 2)
	{
		if(tid < s)
		{
			if(sdata[tid] > sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid < 32)
	{
		if(sdata[tid] > sdata[tid + 32])
			sdata[tid] = sdata[tid + 32];
		if(sdata[tid] > sdata[tid + 16])
			sdata[tid] = sdata[tid + 16];
		if(sdata[tid] > sdata[tid + 8])
			sdata[tid] = sdata[tid + 8];
		if(sdata[tid] > sdata[tid + 4])
			sdata[tid] = sdata[tid + 4];
		if(sdata[tid] > sdata[tid + 2])
			sdata[tid] = sdata[tid + 2];
		if(sdata[tid] > sdata[tid + 1])
			sdata[tid] = sdata[tid + 1];
	}

	if(tid == 0)
		A[blockIdx.x] = sdata[0];
}

__global__ void findMax(int *A){

    __shared__ int sdata[512];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = A[i];
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 32; s /= 2)
	{
		if(tid < s)
		{
			if(sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid < 32)
	{
		if(sdata[tid] < sdata[tid + 32])
			sdata[tid] = sdata[tid + 32];
		if(sdata[tid] < sdata[tid + 16])
			sdata[tid] = sdata[tid + 16];
		if(sdata[tid] < sdata[tid + 8])
			sdata[tid] = sdata[tid + 8];
		if(sdata[tid] < sdata[tid + 4])
			sdata[tid] = sdata[tid + 4];
		if(sdata[tid] < sdata[tid + 2])
			sdata[tid] = sdata[tid + 2];
		if(sdata[tid] < sdata[tid + 1])
			sdata[tid] = sdata[tid + 1];
	}

	__syncthreads();

	if(tid == 0)
		A[blockIdx.x] = sdata[0];
}

void random_number_generator(int *A, int size){
	time_t t;
	srand((unsigned) time(&t));
	for(int i = 0; i < size; i++){
		A[i] = rand();
	}
}

int main(int argc, char ** argv){
	int *A;
	int *d_A_MAX, *d_A_MIN;
	int SIZE[3] = {2, 8 ,32};

	int max_gpu = INT_MIN;
	int min_gpu = INT_MAX;
	int max_cpu = INT_MIN;
	int min_cpu = INT_MAX;

	for(int i = 0; i < 3; i++){

		int size = SIZE[i] * M;

		A = (int*) malloc(size * sizeof(int));
		random_number_generator(A, size);

		float total_max_average = 0;
		float total_min_average = 0;

		cudaMalloc((void**)&d_A_MAX, size * sizeof(int));
		cudaMalloc((void**)&d_A_MIN, size * sizeof(int));
		cudaMemcpy(d_A_MAX, A, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_A_MIN, A, size * sizeof(int), cudaMemcpyHostToDevice);

		dim3 dimGrid1(size / thread_per_block);
		dim3 dimBlock1(thread_per_block);

		dim3 dimGrid2(size / thread_per_block / thread_per_block);
		dim3 dimBlock2(512);

		dim3 dimGrid3(1);
		dim3 dimBlock3(size / thread_per_block / thread_per_block);


		cudaEvent_t start,stop;
		float time_findMax, time_findMin;
		float total_max_average_gpu = 0;
		float total_min_average_gpu = 0;

		for(int j = 0; j < 10; j++){

			int max = INT_MIN, min = INT_MAX;

			clock_t start_t, end_t;

			start_t = clock();
			for(int i = 0; i < size; i++){
				if(A[i] > max)
					max = A[i];
			}
			max_cpu = max;
			end_t = clock();
			total_max_average += (float)(end_t - start_t) / CLOCKS_PER_SEC;

			start_t = clock();
			for(int i = 0; i < size; i++){
				if(A[i] < min)
					min = A[i];
			}
			min_cpu = min;
			end_t = clock();
			total_min_average += (float)(end_t - start_t) / CLOCKS_PER_SEC;

			cudaMemcpy(d_A_MAX, A, size * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_A_MIN, A, size * sizeof(int), cudaMemcpyHostToDevice);

			time_record_begin(start);
			findMax<<<dimGrid1, dimBlock1>>>(d_A_MAX);
			findMax<<<dimGrid2, dimBlock2>>>(d_A_MAX);
			findMax<<<dimGrid3, dimBlock3>>>(d_A_MAX);
			time_record_end(start, stop, time_findMax);
			total_max_average_gpu += time_findMax / 1000;
			cudaMemcpy(&max_gpu, d_A_MAX, sizeof(int), cudaMemcpyDeviceToHost);

			time_record_begin(start);
			findMin<<<dimGrid1, dimBlock1>>>(d_A_MIN);
			findMin<<<dimGrid2, dimBlock2>>>(d_A_MIN);
			findMin<<<dimGrid3, dimBlock3>>>(d_A_MIN);
			time_record_end(start, stop, time_findMin);
			total_min_average_gpu += time_findMin / 1000;
			cudaMemcpy(&min_gpu, d_A_MIN, sizeof(int), cudaMemcpyDeviceToHost);

		}

		printf("N: %dM, GPUmax: %d, CPUmax: %d GPUtime: %f, CPUtime: %f, GPUSpeedup: %f   ", SIZE[i], max_gpu, max_cpu, total_max_average_gpu/10, total_max_average/10, total_max_average/total_max_average_gpu);
		printf("N: %dM, GPUmin: %d, CPUmin: %d GPUtime: %f, CPUtime: %f, GPUSpeedup: %f\n", SIZE[i], min_gpu, min_cpu, total_min_average_gpu/10, total_min_average/10, total_min_average/total_min_average_gpu);

		cudaFree(d_A_MAX);
		cudaFree(d_A_MIN);
		free(A);
	}
}




