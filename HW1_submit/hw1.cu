#include<stdio.h>
#include <stdlib.h>

#define record(a)   {cudaEventCreate(&a);cudaEventRecord(a,0);}
#define calculate(a,b,time)   {cudaEventCreate(&b);cudaEventRecord(b,0);cudaEventSynchronize(b);cudaEventElapsedTime(&time, a,b);}

//Define a constant variable for 1M
const int size_constant = 1000000;
const int multiple = 128;

__global__ void arradd_1(float*a, float X, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<size)
    	a[i] = a[i] + X;
}

__global__ void arradd_2(double* a, double X, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<size)
    	a[i] = a[i] + X;
}

__global__ void iarradd(int32_t* a, int32_t X, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<size)
	    a[i] = a[i] + X;
}

__global__ void xarradd(float* a, float X, int time, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<size)
    {
	    for(int j=0; j < time; j++)
	    	a[i] = a[i] + X;
    }
}

FILE*  file_process(FILE* fp,const char* target, int num)
{
    fp = fopen(target, "w+");
    if(num == 4)
        fprintf(fp, "Question 4\n------------------------------------------------------------\nXaddedTimes    \tElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)");
    else
        fprintf(fp, "Question %d\n------------------------------------------------------------\nElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)",num);
    return fp;
}

void time_measure_1(int multiple, FILE* fp){
	cudaEvent_t start, stop;
	float elapsedTime;
	int size = multiple*size_constant;

	float* A;
	float X = 0.1;
	float *cuda_A;

	A=(float*) malloc(size*sizeof(float));
	for(int i = 0;i < size; i++)
		A[i] = i / 3.0f;

	cudaMalloc((void**)&cuda_A,size*sizeof(float));
/*************************************************************************************************/
	record(start);
	cudaMemcpy(cuda_A, A, size*sizeof(float),cudaMemcpyHostToDevice);
	calculate(start,stop, elapsedTime);
    printf("\n%d\t\t %f" ,multiple, elapsedTime);
	fprintf(fp, "\n%d\t\t\t\t %f" ,multiple, elapsedTime);
/*************************************************************************************************/
	dim3 DimGrid(multiple*1000);
	dim3 DimBlock(1000);
/*************************************************************************************************/
	record(start);
	arradd_1<<<DimGrid, DimBlock>>>(cuda_A, X, size);
	cudaDeviceSynchronize();
    calculate(start,stop, elapsedTime);
    printf("\t %f" ,elapsedTime);
	fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	record(start);
	cudaMemcpy(A, cuda_A, size*sizeof(float), cudaMemcpyDeviceToHost);
	calculate(start,stop, elapsedTime);
    printf("\t %f" ,elapsedTime);
	fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	cudaFree(cuda_A);
    free(A);
}

void time_measure_2(int multiple, FILE* fp){
	cudaEvent_t start, stop;
	float elapsedTime;
	int size = multiple*size_constant;

	double* A;
	double X = 0.1;
	double *cuda_A;

	A=(double*) malloc(size*sizeof(double));
	for(int i = 0;i < size; i++)
		A[i] = i / 3.0f;

	cudaMalloc((void**)&cuda_A,size*sizeof(double));
/*************************************************************************************************/
    record(start);
	cudaMemcpy(cuda_A, A, size*sizeof(double),cudaMemcpyHostToDevice);
    calculate(start,stop, elapsedTime);
	printf("\n%d\t\t %f" ,multiple, elapsedTime);
    fprintf(fp, "\n%d\t\t\t\t %f" ,multiple, elapsedTime);
/*************************************************************************************************/
	dim3 DimGrid(multiple*1000);
	dim3 DimBlock(1000);
/*************************************************************************************************/
	record(start);
	arradd_2<<<DimGrid, DimBlock>>>(cuda_A, X, size);
	cudaDeviceSynchronize();
	calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	record(start);
	cudaMemcpy(A, cuda_A, size*sizeof(double), cudaMemcpyDeviceToHost);
	calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	cudaFree(cuda_A);
    free(A);
}

void time_measure_3(int multiple, FILE* fp){
	cudaEvent_t start, stop;
	float elapsedTime;
	int size = multiple*size_constant;

	int32_t* A;
	int32_t X = 0.1;
	int32_t *cuda_A;

	A=(int32_t*) malloc(size*sizeof(int32_t));
	for(int i = 0;i < size; i++)
		A[i] = i / 3.0f;

	cudaMalloc((void**)&cuda_A,size*sizeof(int32_t));
/*************************************************************************************************/
	record(start);
	cudaMemcpy(cuda_A, A, size*sizeof(int32_t),cudaMemcpyHostToDevice);
    calculate(start,stop, elapsedTime);
	printf("\n%d\t\t %f" ,multiple, elapsedTime);
    fprintf(fp, "\n%d\t\t\t\t %f" ,multiple, elapsedTime);
/*************************************************************************************************/
	dim3 DimGrid(multiple*1000);
	dim3 DimBlock(1000);
/*************************************************************************************************/
	record(start);
	iarradd<<<DimGrid, DimBlock>>>(cuda_A, X, size);
	cudaDeviceSynchronize();
    calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	record(start);
	cudaMemcpy(A, cuda_A, size*sizeof(int32_t), cudaMemcpyDeviceToHost);
    calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	cudaFree(cuda_A);
    free(A);
}

void time_measure_4(int times, FILE* fp){
	cudaEvent_t start, stop;
	float elapsedTime;
	int size = multiple*size_constant;

	float* A;
	float X = 0.1;
	float *cuda_A;

	A=(float*) malloc(size*sizeof(float));
	for(int i = 0;i < size; i++)
		A[i] = i / 3.0f;

	cudaMalloc((void**)&cuda_A,size*sizeof(float));
/*************************************************************************************************/
	record(start);
	cudaMemcpy(cuda_A, A, size*sizeof(float),cudaMemcpyHostToDevice);
    calculate(start,stop, elapsedTime);
	printf("\n%d \t\t %d\t\t %f" , times, multiple, elapsedTime);
    fprintf(fp, "\n%d\t\t\t\t%d\t\t\t\t %f" ,times, multiple, elapsedTime);
/*************************************************************************************************/
	dim3 DimGrid(multiple*1000);
	dim3 DimBlock(1000);
/*************************************************************************************************/
	record(start);
	xarradd<<<DimGrid, DimBlock>>>(cuda_A, X, times, size);
	cudaDeviceSynchronize();
	calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	record(start);
	cudaMemcpy(A, cuda_A, size*sizeof(float), cudaMemcpyDeviceToHost);
	calculate(start,stop, elapsedTime);
	printf("\t %f" ,elapsedTime);
    fprintf(fp,"\t\t %f" ,elapsedTime);
/*************************************************************************************************/
	cudaFree(cuda_A);
	free(A);
}

void trigger_1()
{
    FILE* fp;
    fp = file_process(fp, "outA.txt",1);
	printf("Question 1\n------------------------------------------------------------\nElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)");
	int multiple = 1;
	while(multiple <= 256)
	{
		time_measure_1(multiple, fp);
		multiple *=2;
	}
	printf("\n\n");
    fclose(fp);
}

void trigger_2()
{
    FILE* fp;
    fp = file_process(fp, "outB.txt",2);
	printf("Question 2\n------------------------------------------------------------\nElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)");
	int multiple = 1;
	while(multiple <= 256)
	{
		time_measure_2(multiple, fp);
		multiple *=2;
	}
	printf("\n\n");
    fclose(fp);
}

void trigger_3()
{
    FILE* fp;
    fp = file_process(fp, "outC.txt",3);
	printf("Question 3\n------------------------------------------------------------\nElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)");
	int multiple = 1;
	while(multiple <= 256)
	{
		time_measure_3(multiple, fp);
		multiple *=2;
	}
	printf("\n\n");
    fclose(fp);
}

void trigger_4()
{
    FILE* fp;
    fp = file_process(fp, "outD.txt",4);
	printf("Question 4\n------------------------------------------------------------\nXaddedTimes    \tElements(M)    \tCPUtoGPU(ms)  \tKernel(ms)   \tGPUtoCPU(ms)");
	int time = 1;
	while(time <= 256)
	{
		time_measure_4(time, fp);
		time *=2;
	}
    printf("\n\n");
    fclose(fp);
}

int main()
{
    trigger_1();
    trigger_2();
    trigger_3();
    trigger_4();
}
