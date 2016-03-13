#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

__global__ void imageFilterKernelPartA(char3* inputPixels, char3* outputPixels, uint width, uint height, int pixel_per_thread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int idx = index * pixel_per_thread; idx <  (index+1) * pixel_per_thread; idx++){
		int3 sum = {0, 0, 0};
		int count = 0;
	    
		for(int i = -4; i <= 4; i++){
			for(int j = -4; j <= 4; j++){
				int target = idx + j * width + i;
				if(target >= 0 && target < (width * height)){
					sum.x += (int)inputPixels[target].x;
					sum.y += (int)inputPixels[target].y;
					sum.z += (int)inputPixels[target].z;
					count++;
				}
			}
	    }	
		outputPixels[idx].x = sum.x/count;
		outputPixels[idx].y = sum.y/count;
		outputPixels[idx].z = sum.z/count;			
	}
}
__global__ void imageFilterKernelPartB(char3* inputPixels, char3* outputPixels, uint width, uint height, int pixel_per_thread, int total_thread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for(int idx = 0; idx < pixel_per_thread; idx++){
		int currIdx = idx * total_thread + index;
		int3 sum = {0, 0, 0};
		int count = 0;
		
		for(int i = -4; i <= 4; i++){
			for(int j = -4; j <= 4; j++){
				int target = currIdx + j * width + i;
				if(target >= 0 && target < (width * height)){
					sum.x += (int)inputPixels[target].x;
					sum.y += (int)inputPixels[target].y;
					sum.z += (int)inputPixels[target].z;
					count++;
				}
			}
	    }
		outputPixels[currIdx].x = sum.x/count;
		outputPixels[currIdx].y = sum.y/count;
		outputPixels[currIdx].z = sum.z/count;
	}
}

__global__ void imageFilterKernelPartC(char3* inputPixels, char3* outputPixels, uint width, uint height /*, other arguments */)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
}


#endif // _IMAGEFILTER_KERNEL_H_
