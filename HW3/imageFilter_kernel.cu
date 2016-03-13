#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

__global__ void imageFilterKernelPartA(char3* inputPixels, char3* outputPixels, uint width, uint height /*, other arguments */)
{

	//Sample code, that just copies input pixels to output pixels
	for(int i = 0; i < width * height; i ++)
	{
		outputPixels[i].x = inputPixels[i].x;
		outputPixels[i].y = inputPixels[i].y;
		outputPixels[i].z = inputPixels[i].z;
	}
	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}
__global__ void imageFilterKernelPartB(char3* inputPixels, char3* outputPixels, uint width, uint height /*, other arguments */)
{

	//Sample code, that just copies input pixels to output pixels
	for(int i = 0; i < width * height; i ++)
	{
		outputPixels[i].x = inputPixels[i].x;
		outputPixels[i].y = inputPixels[i].y;
		outputPixels[i].z = inputPixels[i].z;
	}
	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}
__global__ void imageFilterKernelPartC(char3* inputPixels, char3* outputPixels, uint width, uint height /*, other arguments */)
{

	//Sample code, that just copies input pixels to output pixels
	for(int i = 0; i < width * height; i ++)
	{
		outputPixels[i].x = inputPixels[i].x;
		outputPixels[i].y = inputPixels[i].y;
		outputPixels[i].z = inputPixels[i].z;
	}
	//Assign IDs to threads
	//distribute work between threads
	//do the computation and store the output pixels in outputPixels

}

#endif // _IMAGEFILTER_KERNEL_H_
