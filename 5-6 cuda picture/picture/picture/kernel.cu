#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_image.h"

#include <iostream> 
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <windows.h>
#include <ctime>
#include <cmath>
#include <cstdlib>


using namespace std;

#define THREAD_X 32
#define THREAD_Y 32

void cudaCheckStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("CUDA return error code: %s - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		exit(-1);
	}
}

unsigned char* addFrameToImage(const unsigned char* img, unsigned int width, unsigned int height) {
	
	unsigned char* result = new unsigned char[width * height];

	for (int i = 0; i < width; i++) {
		result[i] = 0;
		result[(height - 1) * width + i] = 0;
	}

	for (int i = 1; i < height - 1; i++) {
		result[i * width] = 0;
		result[(i + 1) * width - 1] = 0;
	}

	int index = 0;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			result[i * width + j] = img[index++];
		}
	}

	return result;
}

void deleteFrameFromoImage(const unsigned char* imgIn, unsigned char* imgOut, unsigned int width, unsigned int height) {

	int index = 0;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			imgOut[index++] = imgIn[i * width + j];
		}
	}
}

//<----------------------------------CPU------------------------------------->

unsigned char stampingEffect(unsigned char* arrayAround)
{
	return arrayAround[1] * 1 + arrayAround[3] * 1 + arrayAround[5] * (-1) + arrayAround[7] * (-1);
}

void cpu_work(const unsigned char* imgIn, unsigned char* imgOut, int width, int height)
{
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&start);
	unsigned char temp[9];

	for (int i = 1; i < width - 1; ++i)
	{
		for (int j = 1; j < height - 1; ++j)
		{
			temp[0] = imgIn[(j - 1)*width + (i - 1)];
			temp[1] = imgIn[(j - 1)*width + (i + 0)];
			temp[2] = imgIn[(j - 1)*width + (i + 1)];
			
			temp[3] = imgIn[(j + 0)*width + (i - 1)];
			temp[4] = imgIn[(j + 0)*width + (i + 0)];
			temp[5] = imgIn[(j + 0)*width + (i + 1)];
			
			temp[6] = imgIn[(j + 1)*width + (i - 1)];
			temp[7] = imgIn[(j + 1)*width + (i + 0)];
			temp[8] = imgIn[(j + 1)*width + (i + 1)];

			imgOut[j*width + i] = stampingEffect(temp);
		}
	}

	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	printf("The time for cpu spend: %.3f ms\n", delay);
}

//<----------------------------------GPU------------------------------------->

__global__ void stampingEffectCuda(unsigned char* src, unsigned char* dst, int width, int height)
{
	//const int offsetX = BLOCK_X*blockIdx.x*THREAD_ELEMENT_X + threadIdx.x*THREAD_ELEMENT_X;
	//const int offsetY = BLOCK_Y*blockIdx.y*THREAD_ELEMENT_Y + threadIdx.y*THREAD_ELEMENT_Y;

	//int a = src[(offsetY + 0)*rowLength + offsetX];
	//int b = src[(offsetY + 1)*rowLength + offsetX];
	//int c = src[(offsetY + 2)*rowLength + offsetX];
	//int d = src[(offsetY + 3)*rowLength + offsetX];

	///*int startWriteIndex = offsetY*rowLength + offsetX*THREAD_ELEMENT_Y;

	//dst[startWriteIndex+0] = a;
	//dst[startWriteIndex+1] = b;
	//dst[startWriteIndex+2] = c;
	//dst[startWriteIndex+3] = d;*/


	//const int offsetX_out = BLOCK_Y*blockIdx.x*THREAD_ELEMENT_Y + threadIdx.x*THREAD_ELEMENT_Y;
	//const int offsetY_out = BLOCK_X*blockIdx.y*THREAD_ELEMENT_X + threadIdx.y*THREAD_ELEMENT_X;

	//dst[offsetY_out*rowLength * 4 + offsetX_out + 0] = a;
	//dst[offsetY_out*rowLength * 4 + offsetX_out + 1] = b;
	//dst[offsetY_out*rowLength * 4 + offsetX_out + 2] = c;
	//dst[offsetY_out*rowLength * 4 + offsetX_out + 3] = d;

	int offsetX = blockIdx.x*blockDim.x + threadIdx.x;
	int offsetY = blockIdx.y*blockDim.y + threadIdx.y;

	if (offsetX == 0 || offsetY == 0 || offsetX == width - 1 || offsetY == height - 1)
	{
		return;
	}

	unsigned char top = src[(offsetY - 1)*width + (offsetX + 0)];
	unsigned char left = src[(offsetY + 0)*width + (offsetX - 1)];
	unsigned char right = src[(offsetY + 0)*width + (offsetX + 1)];
	unsigned char botom = src[(offsetY + 1)*width + (offsetX + 0)];

	dst[offsetY*width + offsetX] = top + left - right - botom;
}

void cuda_work(unsigned char* imgIn, unsigned char* imgOut, int width, int height)
{
	unsigned char* deviceInMatrix;
	unsigned char* deviceOutMatrix;

	cudaEvent_t start, stop;
	cudaCheckStatus(cudaEventCreate(&start));
	cudaCheckStatus(cudaEventCreate(&stop));

	int widthGpu = width, heightGpu = height;
	if (widthGpu%THREAD_X != 0)
	{
		widthGpu = (width / THREAD_X + 1)*THREAD_X;
	}
	if (heightGpu%THREAD_Y != 0)
	{
		heightGpu = (height / THREAD_Y + 1)*THREAD_Y;
	}

	//int pitch= (width / THREAD_X + 1)*THREAD_X;

	cudaCheckStatus(cudaMalloc(&deviceInMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	cudaCheckStatus(cudaMalloc(&deviceOutMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	for (int i = 0; i < height; ++i)
	{
		cudaCheckStatus(cudaMemcpy(&deviceInMatrix[i*widthGpu], &imgIn[i*width], width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}

	dim3 dimGrid(widthGpu/ THREAD_X, heightGpu/ THREAD_Y);
	dim3 dimBlock(THREAD_X, THREAD_Y);
	printf("gridX = %d gridY = %d\n", dimGrid.x, dimGrid.y);
	printf("blockX = %d blockY = %d\n", dimBlock.x, dimBlock.y);

	cudaCheckStatus(cudaEventRecord(start, 0));

	stampingEffectCuda << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix, widthGpu, heightGpu);

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaDeviceSynchronize());
	cudaCheckStatus(cudaEventRecord(stop, 0));
	cudaCheckStatus(cudaEventSynchronize(stop));

	float elapsedTime;
	cudaCheckStatus(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("The time for cuda with global memory spend: %.3f ms\n", elapsedTime);

	cudaCheckStatus(cudaEventDestroy(start));
	cudaCheckStatus(cudaEventDestroy(stop));

	printf("problem1");
	unsigned char* temp = new unsigned char[widthGpu*heightGpu];
	cudaCheckStatus(cudaMemcpy2D(imgOut, width, deviceOutMatrix, widthGpu, width, height, cudaMemcpyDeviceToHost));
	printf("problem2\n");

	cudaCheckStatus(cudaFree(deviceInMatrix));
	cudaCheckStatus(cudaFree(deviceOutMatrix));
}


int main()
{
	printf("start\n");

	char* imagePathInput = "mountain.pgm";
	char* imagePathResCpu = "mountain_outCPU.pgm";
	char* imagePathResGpu = "mountain_outGPU.pgm";

	unsigned char* inputImg = nullptr;
	unsigned int widthImage = 0, heightImage = 0, channels = 0;

	__loadPPM(imagePathInput, &inputImg, &widthImage, &heightImage, &channels);

	unsigned char* imgCPUOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char));
	unsigned char* imgGPUOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char));
	unsigned char* imgGPUWithSharedOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char));

	printf("Input image is \"%s\" - size: %dX%d - channels: %d\n", imagePathInput, widthImage, heightImage, channels);

	//ADD FRAME
	unsigned int width = widthImage + 2;
	unsigned int height = heightImage + 2;
	unsigned char* imgWithFrame = addFrameToImage(inputImg, width, height);

	//CPU
	unsigned char* temp_CPU = new unsigned char[width*height];
	cpu_work(imgWithFrame, temp_CPU, width, height);
	deleteFrameFromoImage(temp_CPU, imgCPUOut, width, height);

	//CUDA
	unsigned char* temp_GPU = new unsigned char[width*height];
	cuda_work(imgWithFrame, temp_GPU, width, height);
	deleteFrameFromoImage(temp_GPU, imgGPUOut, width, height);

	__savePPM(imagePathResCpu, imgCPUOut, widthImage, heightImage, channels);
	__savePPM(imagePathResGpu, imgGPUOut, widthImage, heightImage, channels);

	printf("end\n");
}

