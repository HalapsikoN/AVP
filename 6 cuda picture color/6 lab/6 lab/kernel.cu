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

#define BLOCK_X 32
#define BLOCK_Y 32

void cudaCheckStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("CUDA return error code: %s - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		exit(-1);
	}
}

unsigned char* addFrameToImage(const unsigned char* img, unsigned int width, unsigned int height, unsigned int channels) {

	int newWidth = width + 2;
	int newHeight = height + 2;
	unsigned char* result = (unsigned char*)calloc(newWidth * newHeight * channels, sizeof(unsigned char));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width*channels; j++) {
			result[(i + 1) * newWidth * channels + (j + channels)] = img[i*width * channels + j];
		}
	}

	return result;
}

void deleteFrameFromoImage(const unsigned char* imgIn, unsigned char* imgOut, unsigned int width, unsigned int height, unsigned int channels) {

	int index = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width*channels; j++) {
			imgOut[i*width * channels + j] = imgIn[(i+1) * (width+2)*channels + (j+ channels)];
		}
	}
}

//<----------------------------------CPU------------------------------------->

unsigned char stampingEffect(unsigned char* arrayAround)
{
	return arrayAround[1] * 1 + arrayAround[3] * 1 + arrayAround[5] * (-1) + arrayAround[7] * (-1);
}

void cpu_work(const unsigned char* imgIn, unsigned char* imgOut, int width, int height, int channels)
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
			for (int channel = 0; channel < channels; ++channel)
			{
				//temp[0] = imgIn[(j - 1)*width + (i - 1)];
				temp[1] = imgIn[(j - 1)*width * channels + (i + 0) * channels + channel];
				//temp[2] = imgIn[(j - 1)*width + (i + 1)];

				temp[3] = imgIn[(j + 0)*width * channels + (i - 1) * channels + channel];
				//temp[4] = imgIn[(j + 0)*width + (i + 0)];
				temp[5] = imgIn[(j + 0)*width * channels + (i + 1) * channels + channel];

				//temp[6] = imgIn[(j + 1)*width + (i - 1)];
				temp[7] = imgIn[(j + 1)*width * channels + (i + 0) * channels + channel];
				//temp[8] = imgIn[(j + 1)*width + (i + 1)];

				imgOut[j*width * channels + i * channels + channel] = stampingEffect(temp);
			}
		}
	}

	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	printf("The time for cpu spend: %.3f ms\n\n", delay);
}

//<----------------------------------GPU------------------------------------->

__global__ void stampingEffectCuda(unsigned char* src, unsigned char* dst, int width, int height, int channels)
{

	int offsetX = blockIdx.x*blockDim.x + threadIdx.x;
	int offsetY = blockIdx.y*blockDim.y + threadIdx.y;

	if (offsetX == 0 || offsetY == 0 || offsetX == width - 1 || offsetY == height - 1)
	{
		return;
	}

	unsigned char top = src[(offsetY - 1)*width + (offsetX + 0)];
	unsigned char left = src[(offsetY + 0)*width + (offsetX - channels)];
	unsigned char right = src[(offsetY + 0)*width + (offsetX + channels)];
	unsigned char botom = src[(offsetY + 1)*width + (offsetX + 0)];

	dst[offsetY*width + offsetX] = top + left - right - botom;
}

void cuda_work(unsigned char* imgIn, unsigned char* imgOut, int width, int height, int channels)
{
	unsigned char* deviceInMatrix;
	unsigned char* deviceOutMatrix;

	cudaEvent_t start, stop;
	cudaCheckStatus(cudaEventCreate(&start));
	cudaCheckStatus(cudaEventCreate(&stop));

	int widthGpu = width, heightGpu = height;
	if (widthGpu%BLOCK_X != 0)
	{
		widthGpu = (width / BLOCK_X + 1)*BLOCK_X;
	}
	if (heightGpu%BLOCK_Y != 0)
	{
		heightGpu = (height / BLOCK_Y + 1)*BLOCK_Y;
	}

	//int pitch= (width / THREAD_X + 1)*THREAD_X;

	cudaCheckStatus(cudaMalloc(&deviceInMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	cudaCheckStatus(cudaMalloc(&deviceOutMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	for (int i = 0; i < height; ++i)
	{
		cudaCheckStatus(cudaMemcpy(&deviceInMatrix[i*widthGpu], &imgIn[i*width], width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}

	dim3 dimGrid(widthGpu / BLOCK_X, heightGpu / BLOCK_Y);
	dim3 dimBlock(BLOCK_X, BLOCK_Y);
	printf("gridX = %d gridY = %d\n", dimGrid.x, dimGrid.y);
	printf("blockX = %d blockY = %d\n", dimBlock.x, dimBlock.y);

	cudaCheckStatus(cudaEventRecord(start, 0));

	stampingEffectCuda << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix, widthGpu, heightGpu, channels);

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaDeviceSynchronize());
	cudaCheckStatus(cudaEventRecord(stop, 0));
	cudaCheckStatus(cudaEventSynchronize(stop));

	float elapsedTime;
	cudaCheckStatus(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("The time for cuda with global memory spend: %.3f ms\n\n", elapsedTime);

	cudaCheckStatus(cudaEventDestroy(start));
	cudaCheckStatus(cudaEventDestroy(stop));

	unsigned char* temp = new unsigned char[widthGpu*heightGpu];
	cudaCheckStatus(cudaMemcpy2D(imgOut, width, deviceOutMatrix, widthGpu, width, height, cudaMemcpyDeviceToHost));

	cudaCheckStatus(cudaFree(deviceInMatrix));
	cudaCheckStatus(cudaFree(deviceOutMatrix));
}


//<----------------------------------GPU WITH SHARED------------------------------------->

__global__ void stampingEffectCudaShared(unsigned char* src, unsigned char* dst, int width, int height, int channels)
{
	int offsetX = blockIdx.x*blockDim.x + threadIdx.x;
	int offsetY = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ int smemIn[(BLOCK_X + 6)*(BLOCK_Y + 2)];
	__shared__ int smemOut[(BLOCK_X + 6)*(BLOCK_Y + 2)];
	int sharedBlockSizeX = BLOCK_X + 6;
	int sharedBlockSizeY = BLOCK_Y + 6;

	if (offsetX == 0 || offsetY == 0 || offsetX == width - 3 || offsetY == height - 1)
	{
		return;
	}

	//
	smemIn[(threadIdx.y + 1)*sharedBlockSizeX + (threadIdx.x + 3)] = src[(offsetY + 0)*width + (offsetX + 0)];

	//
	if (threadIdx.y == 0)
	{
		smemIn[threadIdx.x + 3] = src[(offsetY - 1)*width + (offsetX + 0)];
	}
	if (threadIdx.x == 0)
	{
		smemIn[(threadIdx.y + 1)*sharedBlockSizeX] = src[(offsetY + 0)*width + (offsetX - 1)];
	}
	if (threadIdx.x == blockDim.x - 1)
	{
		smemIn[(threadIdx.y + 1)*sharedBlockSizeX + (sharedBlockSizeX - 3)] = src[(offsetY + 0)*width + (offsetX + 1)];
	}
	if (threadIdx.y == blockDim.y - 1)
	{
		smemIn[(sharedBlockSizeY - 1)*sharedBlockSizeX + (threadIdx.x + 3)] = src[(offsetY + 1)*width + (offsetX + 0)];
	}

	__syncthreads();

	int offX = threadIdx.x + 3;
	int offY = threadIdx.y + 1;

	unsigned char top = smemIn[(offY - 1)*sharedBlockSizeX + (offX + 0)];
	unsigned char left = smemIn[(offY + 0)*sharedBlockSizeX + (offX - channels)];
	unsigned char right = smemIn[(offY + 0)*sharedBlockSizeX + (offX + channels)];
	unsigned char botom = smemIn[(offY + 1)*sharedBlockSizeX + (offX + 0)];

	smemOut[offY*sharedBlockSizeX + offX] = top + left - right - botom;

	__syncthreads();

	dst[(offsetY + 0)*width + (offsetX + 0)] = smemOut[(threadIdx.y + 1)*sharedBlockSizeX + (threadIdx.x + 3)];
}


void cuda_work_shared(unsigned char* imgIn, unsigned char* imgOut, int width, int height, int channels)
{
	unsigned char* deviceInMatrix;
	unsigned char* deviceOutMatrix;

	cudaEvent_t start, stop;
	cudaCheckStatus(cudaEventCreate(&start));
	cudaCheckStatus(cudaEventCreate(&stop));

	int widthGpu = width, heightGpu = height;
	if (widthGpu%BLOCK_X != 0)
	{
		widthGpu = (width / BLOCK_X + 1)*BLOCK_X;
	}
	if (heightGpu%BLOCK_Y != 0)
	{
		heightGpu = (height / BLOCK_Y + 1)*BLOCK_Y;
	}

	cudaCheckStatus(cudaMalloc(&deviceInMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	cudaCheckStatus(cudaMalloc(&deviceOutMatrix, (widthGpu*heightGpu * sizeof(unsigned char))));
	for (int i = 0; i < height; ++i)
	{
		cudaCheckStatus(cudaMemcpy(&deviceInMatrix[i*widthGpu], &imgIn[i*width], width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}

	dim3 dimGrid(widthGpu / BLOCK_X, heightGpu / BLOCK_Y);
	dim3 dimBlock(BLOCK_X, BLOCK_Y);
	printf("gridX = %d gridY = %d\n", dimGrid.x, dimGrid.y);
	printf("blockX = %d blockY = %d\n", dimBlock.x, dimBlock.y);

	cudaCheckStatus(cudaEventRecord(start, 0));

	stampingEffectCudaShared << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix, widthGpu, heightGpu, channels);

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaDeviceSynchronize());
	cudaCheckStatus(cudaEventRecord(stop, 0));
	cudaCheckStatus(cudaEventSynchronize(stop));

	float elapsedTime;
	cudaCheckStatus(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("The time for cuda with shared memory spend: %.3f ms\n\n", elapsedTime);

	cudaCheckStatus(cudaEventDestroy(start));
	cudaCheckStatus(cudaEventDestroy(stop));

	unsigned char* temp = new unsigned char[widthGpu*heightGpu];
	cudaCheckStatus(cudaMemcpy2D(imgOut, width, deviceOutMatrix, widthGpu, width, height, cudaMemcpyDeviceToHost));

	cudaCheckStatus(cudaFree(deviceInMatrix));
	cudaCheckStatus(cudaFree(deviceOutMatrix));
}

//<----------------------------------CHECKING------------------------------------->

bool compareTwoPictures(unsigned char* first, unsigned char* second, int width, int height)
{
	for (int i = 0; i < width*height; ++i)
	{
		if (first[i] != second[i])
		{
			printf("The mistake is at index: %d", i);
			return false;
		}
	}
	return true;
}

int main()
{
	char* imagePathInput = "autumn.ppm";
	char* imagePathResCpu = "autumn_outCPU.ppm";
	char* imagePathResGpu = "autumn_outGPU.ppm";
	char* imagePathResGpuShared = "autumn_outGPUShared.ppm";

	unsigned char* inputImg = nullptr;
	unsigned int widthImage = 0, heightImage = 0, channels = 0;

	__loadPPM(imagePathInput, &inputImg, &widthImage, &heightImage, &channels);

	unsigned char* imgCPUOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char)*channels);
	unsigned char* imgGPUOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char)*channels);
	unsigned char* imgGPUWithSharedOut = (unsigned char*)malloc(widthImage * heightImage * sizeof(unsigned char)*channels);

	printf("Input image is \"%s\" - size: %dX%d - channels: %d\n\n", imagePathInput, widthImage, heightImage, channels);

	//ADD FRAME
	unsigned int width = widthImage + 2;
	unsigned int height = heightImage + 2;
	//unsigned char* imgWithFrame = addFrameToImage(inputImg, width, height);
	unsigned char* imgWithFrame = addFrameToImage(inputImg, widthImage, heightImage, channels);

	//CPU
	unsigned char* temp_CPU = new unsigned char[width*height*channels];
	cpu_work(imgWithFrame, temp_CPU, width, height, channels);
	deleteFrameFromoImage(temp_CPU, imgCPUOut, widthImage, heightImage, channels);
	

	//CUDA
	unsigned char* temp_GPU = new unsigned char[width*height*channels];
	cuda_work(imgWithFrame, temp_GPU, width*channels, height, channels);
	deleteFrameFromoImage(temp_GPU, imgGPUOut, widthImage, heightImage, channels);

	//CUDA WITH SHARED
	unsigned char* temp_GPU_shared = new unsigned char[width*height*channels];
	//cuda_work_shared(imgWithFrame, temp_GPU_shared, width*channels, height, channels);
	deleteFrameFromoImage(temp_GPU_shared, imgGPUWithSharedOut, widthImage, heightImage, channels);

	__savePPM(imagePathResCpu, imgCPUOut, widthImage, heightImage, channels);
	printf("SAVED CPU Picture\n");
	__savePPM(imagePathResGpu, imgGPUOut, widthImage, heightImage, channels);
	printf("SAVED GPU Picture\n");
	//__savePPM(imagePathResGpuShared, imgGPUWithSharedOut, widthImage, heightImage, channels);
	//printf("SAVED GPU SHARED Picture\n");

	printf("\nRESULT OF COMPARE CUDA AND CPU: %s\n", (compareTwoPictures(imgCPUOut, imgGPUOut, widthImage, heightImage) ? "+" : "-"));
	//printf("\nRESULT OF COMPARE CUDA SHARED AND CPU: %s\n", (compareTwoPictures(imgCPUOut, imgGPUWithSharedOut, widthImage, heightImage) ? "+" : "-"));

	free(inputImg);
	free(imgCPUOut);
	free(imgGPUOut);
	free(imgGPUWithSharedOut);
	free(temp_CPU);
	free(temp_GPU);
	free(temp_GPU_shared);
}

