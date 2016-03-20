#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

#define BOUND   4
#define THREAD_PER_WARP   32
#define SQUARE_WIDTH   120

__global__ void imageFilterKernelPartA(char3* inputPixels, char3* outputPixels, uint width, uint height, int pxls_per_thrd)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int idx = 0; idx < pxls_per_thrd; idx ++){
		int currIdx = index * pxls_per_thrd + idx;

		int idx_x = currIdx % width;
		int idx_y = currIdx / width;

		int3 sum = {0, 0, 0};
		int count = 0;
	    
		for(int i = -BOUND; i <= BOUND; i++)
		{
			for(int j = -BOUND; j <= BOUND; j++)
			{
				if(((idx_x + i) >= 0) && ((idx_x + i) < width) && ((idx_y + j) >= 0) && ((idx_y + j) < height))
				{
					int target = currIdx + j * width + i;
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


__global__ void imageFilterKernelPartB(char3* inputPixels, char3* outputPixels, uint width, uint height, int pxls_per_thrd, int num_thread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int idx = 0; idx < pxls_per_thrd; idx++)
	{
		int currIdx = idx * num_thread + index;
		int3 sum = {0, 0, 0};
		int count = 0;

		int idx_x = currIdx % width;
		int idx_y = currIdx / width;
		
		for(int i = -BOUND; i <= BOUND; i++)
		{
			for(int j = -BOUND; j <= BOUND; j++)
			{
				if(((idx_x + i) >= 0) && ((idx_x + i) < width) && ((idx_y + j) >= 0) && ((idx_y + j) < height))
				{
					int target = currIdx + j * width + i;
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

__global__ void imageFilterKernelPartC(char3* inputPixels, char3* outputPixels, uint width, uint height, int blocks_row, int blocks_col, int num_loops)
{
	__shared__ char3 tile[128 * 128];
	
	int shared_x = threadIdx.x % THREAD_PER_WARP;
	int shared_y = threadIdx.x / THREAD_PER_WARP;

	for(int i = 0; i < num_loops; i++)
	{
		int global_x = (blockIdx.x + i * 12) % blocks_row;
		int global_y = (blockIdx.x + i * 12) / blocks_row;

		for(int k = 0; k < BOUND; k++)
		{
			for(int j = 0; j < BOUND; j++)
			{
				int idx = (global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) * width + global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP;
				int shared_idx = (shared_y + k * THREAD_PER_WARP) * 128 + shared_x + j * THREAD_PER_WARP;

				if(    ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) >=0) 
                    && ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) < height) 
					&& ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) >= 0) 
                    && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) < width))
				{
					tile[shared_idx] = inputPixels[idx];
				}
			}	
		}
		__syncthreads();

		for(int k = 0; k < BOUND; k++)
		{
			for(int j = 0; j < BOUND; j++)
			{
                if((shared_x + j * THREAD_PER_WARP >= BOUND) && (shared_x + j * THREAD_PER_WARP <= 123) && (shared_y + k * THREAD_PER_WARP >= BOUND) && (shared_y + k * THREAD_PER_WARP <= 123))   
                {
                    int3 sum = {0, 0, 0};
			        int count = 0;
				    for(int dx = -BOUND; dx <= BOUND; dx++)
				    {
					    for(int dy = -BOUND; dy <= BOUND; dy++)
					    {
							sum.x += (int)tile[(shared_y + k * THREAD_PER_WARP + dy) * 128 + (shared_x + dx) + j * THREAD_PER_WARP].x;
							sum.y += (int)tile[(shared_y + k * THREAD_PER_WARP + dy) * 128 + (shared_x + dx) + j * THREAD_PER_WARP].y;
							sum.z += (int)tile[(shared_y + k * THREAD_PER_WARP + dy) * 128 + (shared_x + dx) + j * THREAD_PER_WARP].z;
							count++;
					    }
				    }
	
				    int out_idx = (global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) * width + global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP;
				    if(    ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) >=0) 
                        && ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) < height) 
					    && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) >= 0) 
                        && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) < width))
				    {
					    outputPixels[out_idx].x = sum.x / count;
					    outputPixels[out_idx].y = sum.y / count;
					    outputPixels[out_idx].z = sum.z / count;
				    }
			    }
                /* Check whether the pixels are in the boundary with the width of 4 */
                if(    (global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) <= 3
                    || ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) >= height-BOUND && (global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) < height)
                    || (global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) <= 3
                    || ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) >= width-BOUND && (global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) < width))
                {
                    int3 sum = {0, 0, 0};
			        int count = 0;
				    for(int dx = -BOUND; dx <= BOUND; dx++)
				    {
					    for(int dy = -BOUND; dy <= BOUND; dy++)
					    {
                            if(    ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP + dy) >=0) 
                                && ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP + dy) < height) 
				                && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP + dx) >= 0) 
                                && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP + dx) < width))
						    {
							    sum.x += (int)inputPixels[(global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP + dy) * width + (global_x * SQUARE_WIDTH + shared_x + dx)  + j * THREAD_PER_WARP].x;
							    sum.y += (int)inputPixels[(global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP + dy) * width + (global_x * SQUARE_WIDTH + shared_x + dx)  + j * THREAD_PER_WARP].y;
							    sum.z += (int)inputPixels[(global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP + dy) * width + (global_x * SQUARE_WIDTH + shared_x + dx)  + j * THREAD_PER_WARP].z;
							    count++;
						    }
					    }
				    }
	
				    int out_boundary = (global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) * width + global_x * SQUARE_WIDTH + shared_x + j * 32;
				    if(    ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) >=0) 
                        && ((global_y * SQUARE_WIDTH + shared_y + k * THREAD_PER_WARP) < height) 
				        && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) >= 0) 
                        && ((global_x * SQUARE_WIDTH + shared_x + j * THREAD_PER_WARP) < width))
				    {
					    outputPixels[out_boundary].x = sum.x / count;
					    outputPixels[out_boundary].y = sum.y / count;
					    outputPixels[out_boundary].z = sum.z / count;
				    }
                }
			}
		}
        __syncthreads();
	}
}


#endif // _IMAGEFILTER_KERNEL_H_
