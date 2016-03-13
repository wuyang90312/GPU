#include <stdio.h>
#include <time.h>
#include <stdlib.h>



const int SIZE = 1024 * 1024;
const int thread_per_block = 512;

#define MAX 0
#define MIN 1
#define ITERATION 10
#define CUDA_CLOCKS_PER_SEC     1000

#define record(a){      \
cudaEventCreate(&a);    \
cudaEventRecord(a,0);   }

#define calculate(a,b,time){    \
cudaEventCreate(&b);            \
cudaEventRecord(b,0);           \
cudaEventSynchronize(b);        \
cudaEventElapsedTime(&time, a,b);}

__global__ void GPU_MAX(int *A)
{
	__shared__ int sdata[thread_per_block];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    int tmp = A[i];
    if(A[i+blockDim.x] > tmp)
        tmp = A[i+blockDim.x];
    sdata[tid] = tmp;
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

    for(unsigned int s = 64; s >= 1; s /= 2)
	{
		if(tid < s)
		{
			if(sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
	}

	A[blockIdx.x] = sdata[0];
}
__global__ void GPU_MIN(int *A){

    __shared__ int sdata[thread_per_block];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    int tmp = A[i];
    if(A[i+blockDim.x] < tmp)
        tmp = A[i+blockDim.x];
    sdata[tid] = tmp;
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

    for(unsigned int s = 64; s >= 1; s /= 2)
	{
		if(tid < s)
		{
			if(sdata[tid] > sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
	}

	A[blockIdx.x] = sdata[0];
}

float CPU_extreme(int* input, int select, int size, int itr)
{
	clock_t  start, stop;
	float elapsedTime;// define time to calcualte the average duration

	int output, f_output = 0; // necessary variables
    //Assign the output with the first element value
    output = input[0];

    start = clock();
    if(select == MAX)
    {
    	for(int j = 0; j < itr; j++)
    	{
			for(int i = 1; i < size; i++)
			{

				if(input[i] > output)
					output = input[i];
			}

			f_output += output;
    	}
    }else{
    	for(int j = 0; j < itr; j++)
		{
			for(int i = 1; i < size; i++)
			{
				if(input[i] < output)
					output = input[i];
			}
			f_output += output;
		}
    }
    stop = clock();
    elapsedTime = (float) (stop - start)/ CLOCKS_PER_SEC;
    if(output !=0 && f_output%output!=0)
    	printf("[ERROR] Extreme values are not identical!!!\n");
    else
    	printf(" %d  ",output);

    return (elapsedTime/itr);
}

float GPU_run(int *input, int select, int size, int itr)
{
	cudaEvent_t start, stop;
	float elapsedTime, ave_t=0;// define time to calcualte the average duration

	int *cuda_A, *buff, output, f_output=0;

	dim3 dimGrid1(size / thread_per_block/2);
	dim3 dimBlock1(thread_per_block);

	dim3 dimGrid2(size / thread_per_block / thread_per_block/4);
	dim3 dimBlock2(512);

	dim3 dimGrid3(1);
	dim3 dimBlock3(size / thread_per_block / thread_per_block/8);
	for(int j = 0; j < itr; j++)
	{
        
		// Allocate memory
		cudaMalloc((void**)&cuda_A,size*sizeof(int));
		buff = (int*) malloc(sizeof(int)*size);
		cudaMemcpy(cuda_A, input, size*sizeof(int),cudaMemcpyHostToDevice);
		record(start);
		if(select == MAX)
		{
			GPU_MAX<<<dimGrid1, dimBlock1>>>(cuda_A);
			GPU_MAX<<<dimGrid2, dimBlock2>>>(cuda_A);
			GPU_MAX<<<dimGrid3, dimBlock3>>>(cuda_A);
		}else{
			GPU_MIN<<<dimGrid1, dimBlock1>>>(cuda_A);
			GPU_MIN<<<dimGrid2, dimBlock2>>>(cuda_A);
			GPU_MIN<<<dimGrid3, dimBlock3>>>(cuda_A);
		}
		calculate(start,stop, elapsedTime);
		ave_t += elapsedTime;
		cudaMemcpy(buff, cuda_A, size*sizeof(int), cudaMemcpyDeviceToHost);
		output = buff[0];
		f_output += output;
		// Release the memory
		cudaFree(cuda_A);
		free(buff);
	}

	if(output !=0 && f_output%output!=0)
		printf("[ERROR] Extreme values are not identical!!!\n");
	else
		printf(" %d  ",output); // The extreme number is stored in the first element

	return (ave_t/itr/CUDA_CLOCKS_PER_SEC);
}

int* random_generate(int size)
{
	time_t t; // seed of random
	// Allocate Heap with the give size
	int* random = (int*) malloc(size*sizeof(int));
	// Random fill integers into the array
	srand(time(&t));
	for(int i = 0; i < size; i++)
		random[i] = rand()%200000000;
	return random;
}

int main(){
	int size=0;
	int *random;
	float cpu_t, gpu_t;
//	check(random, size);
//	printf("size: %d\n", random[size-1]);
	for(int i =2; i <= 32; i*=4)
	{
		size = i*SIZE;
		random = random_generate(size);
		printf("N: %dM  GPUmax:",i);
		gpu_t = GPU_run(random, MAX, size, ITERATION);
		printf("CPUmax:");
		cpu_t = CPU_extreme(random, MAX, size, ITERATION);
		printf("GPUtime: %f  CPUtime: %f  SpeedUp: %f   ", gpu_t, cpu_t, (cpu_t/gpu_t));
		printf("GPUmin:");
		gpu_t = GPU_run(random, MIN, size, ITERATION);
		printf("CPUmin:");
		cpu_t = CPU_extreme(random, MIN, size, ITERATION);
		printf("GPUtime: %f  CPUtime: %f  SpeedUp: %f\n", gpu_t, cpu_t, (cpu_t/gpu_t));
	}

    return 0;
}
