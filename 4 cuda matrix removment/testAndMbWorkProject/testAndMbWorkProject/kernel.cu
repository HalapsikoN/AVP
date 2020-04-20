#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <Windows.h>
#include <cuda_runtime.h> 
#include <intrin.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define SIZE_M 64
#define SIZE_N 1024
//почему у гридов какой-то реверс x и y?
#define GRID_X 2
#define GRID_Y 8
#define BLOCK_X 32
#define BLOCK_Y 32
#define THREAD_ELEMENT_X 1
#define THREAD_ELEMENT_Y 4

using namespace std;

//__global__ void vectorAdd(int *a, int *b, int *c, int n)
//{
//	int i = threadIdx.x;
//
//	if(i<n)
//	{
//		c[i] = a[i] + b[i];
//	}
//}

void fillMatrix(int* matrix, int sizeM, int sizeN)
{
	int counter = 0;
	for (int i = 0; i < sizeN; ++i)
	{
		for (int j = 0; j < sizeM; ++j)
		{
			matrix[sizeM*i + j] = counter++;
		}
	}
}

void printfFirstNForInit(int* matrix, int N)
{
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			printf("%4d ", matrix[i*SIZE_M + j]);
		}
		printf("\n");
	}
	printf("<------------------------%d------------------------------->\n", N);
}

void printfFirstNForOut(int* matrix, int N)
{
	for (int i = 0; i < 16/4; ++i)
	{
		for (int j = 0; j < 16*4; ++j)
		{
			printf("%4d ", matrix[i*SIZE_M*4 + j]);
		}
		printf("\n");
	}
	printf("<------------------------%d------------------------------->\n", N);
}

void cudaCheckStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("CUDA return error code: %s - %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		exit(-1);
	}
}

__global__ void matrixRebuild(int* src, int* dst, int rowLength)
{
	const int offsetX = BLOCK_X*blockIdx.x*THREAD_ELEMENT_X + threadIdx.x*THREAD_ELEMENT_X;
	const int offsetY = BLOCK_Y*blockIdx.y*THREAD_ELEMENT_Y + threadIdx.y*THREAD_ELEMENT_Y;

	int a = src[(offsetY + 0)*rowLength+ offsetX];
	int b = src[(offsetY + 1)*rowLength + offsetX];
	int c = src[(offsetY + 2)*rowLength + offsetX];
	int d = src[(offsetY + 3)*rowLength + offsetX];

	/*int startWriteIndex = offsetY*rowLength + offsetX*THREAD_ELEMENT_Y;

	dst[startWriteIndex+0] = a;
	dst[startWriteIndex+1] = b;
	dst[startWriteIndex+2] = c;
	dst[startWriteIndex+3] = d;*/


	const int offsetX_out = BLOCK_X*blockIdx.x*THREAD_ELEMENT_Y + threadIdx.x*THREAD_ELEMENT_Y;
	const int offsetY_out = BLOCK_Y*blockIdx.y*THREAD_ELEMENT_X + threadIdx.y*THREAD_ELEMENT_X;

	dst[offsetY_out*rowLength * 4 + offsetX_out + 0] = a;
	dst[offsetY_out*rowLength * 4 + offsetX_out + 1] = b;
	dst[offsetY_out*rowLength * 4 + offsetX_out + 2] = c;
	dst[offsetY_out*rowLength * 4 + offsetX_out + 3] = d;
}

__global__ void matrixRebuildShared(int* src, int* dst, int rowLength)
{
	const int offsetX = BLOCK_X*blockIdx.x*THREAD_ELEMENT_X + threadIdx.x;
	const int offsetY = BLOCK_Y*blockIdx.y*THREAD_ELEMENT_Y + threadIdx.y;

	__shared__ int smemIn[BLOCK_X*BLOCK_Y*THREAD_ELEMENT_X*THREAD_ELEMENT_Y];
	__shared__ int smemOut[BLOCK_X*BLOCK_Y*THREAD_ELEMENT_X*THREAD_ELEMENT_Y];

	//int row = BLOCK_X*THREAD_ELEMENT_X;
	int row = 32;

	smemIn[(threadIdx.y + 0)*row + threadIdx.x] = src[(offsetY + 0)*rowLength + offsetX];
	smemIn[(threadIdx.y + 32)*row + threadIdx.x] = src[(offsetY + 32)*rowLength + offsetX];
	smemIn[(threadIdx.y + 64)*row + threadIdx.x] = src[(offsetY + 64)*rowLength + offsetX];
	smemIn[(threadIdx.y + 96)*row + threadIdx.x] = src[(offsetY + 96)*rowLength + offsetX];

	__syncthreads();

	int a = smemIn[(threadIdx.y*4 + 0)*row + threadIdx.x];
	int b = smemIn[(threadIdx.y*4 + 1)*row + threadIdx.x];
	int c = smemIn[(threadIdx.y*4 + 2)*row + threadIdx.x];
	int d = smemIn[(threadIdx.y*4 + 3)*row + threadIdx.x];

	smemOut[threadIdx.y*4*row + (threadIdx.x * 4 + 0)] = a;
	smemOut[threadIdx.y*4*row + (threadIdx.x * 4 + 1)] = b;
	smemOut[threadIdx.y*4*row + (threadIdx.x * 4 + 2)] =c;
	smemOut[threadIdx.y*4*row + (threadIdx.x * 4 + 3)] = d;

	dst[(offsetY + 0)*rowLength + offsetX] = smemOut[(threadIdx.y)*row + threadIdx.x+0];
	dst[(offsetY + 32)*rowLength + offsetX] = smemOut[(threadIdx.y)*row + threadIdx.x+32];
	dst[(offsetY + 64)*rowLength + offsetX] = smemOut[(threadIdx.y)*row + threadIdx.x+64];
	dst[(offsetY + 96)*rowLength + offsetX] = smemOut[(threadIdx.y)*row + threadIdx.x+96];

	/*
	dst[(offsetY + 0)*rowLength + offsetX] = smemIn[(threadIdx.y+0)*row+threadIdx.x];
	dst[(offsetY + 1)*rowLength + offsetX] = smemIn[(threadIdx.y + 1)*row + threadIdx.x];
	dst[(offsetY + 2)*rowLength + offsetX] = smemIn[(threadIdx.y + 0)*row + threadIdx.x+1];
	dst[(offsetY + 3)*rowLength + offsetX] = smemIn[(threadIdx.y + 1)*row + threadIdx.x+1];*/
}

void cpuWork(int* initMatrix, int* outMatrix)
{

	int rowLength = GRID_X*BLOCK_X*THREAD_ELEMENT_X;

	int addresToWrite = 0;
	
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&start);
	
	for (int i = 0; i < SIZE_N; i+=4)
	{
		for (int j = 0; j < SIZE_M; j++)
		{
			outMatrix[i * rowLength + j * 4 + 0] = initMatrix[(i + 0) * rowLength + j];
			outMatrix[i * rowLength + j * 4 + 1] = initMatrix[(i + 1) * rowLength + j];
			outMatrix[i * rowLength + j * 4 + 2] = initMatrix[(i + 2) * rowLength + j];
			outMatrix[i * rowLength + j * 4 + 3] = initMatrix[(i + 3) * rowLength + j];
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	printf("The time for cpu spend: %.3f ms\n", delay);

}

void cudaWork(int* initMatrix, int* cuda_outMatrix)
{
	int* deviceInMatrix;
	int* deviceOutMatrix;

	cudaEvent_t start, stop;
	cudaCheckStatus(cudaEventCreate(&start));
	cudaCheckStatus(cudaEventCreate(&stop));

	cudaCheckStatus(cudaMalloc(&deviceInMatrix, (SIZE_M*SIZE_N * sizeof(int))));
	cudaCheckStatus(cudaMalloc(&deviceOutMatrix, (SIZE_N*SIZE_M * sizeof(int))));
	cudaCheckStatus(cudaMemcpy(deviceInMatrix, initMatrix, SIZE_M*SIZE_N * sizeof(int), cudaMemcpyHostToDevice));

	//dim3 dimGrid(((SIZE_M + BLOCK_X * 4 - 1) / BLOCK_X * 4), ((SIZE_N + BLOCK_Y * 1 - 1) / BLOCK_Y * 1));
	dim3 dimGrid(GRID_X, GRID_Y);
	dim3 dimBlock(BLOCK_X, BLOCK_Y);

	printf("gridX = %d gridY = %d\n", dimGrid.x, dimGrid.y);
	printf("blockX = %d blockY = %d\n", dimBlock.x, dimBlock.y);
	printf("threads in block = %d elements in block = %d\n", BLOCK_X*BLOCK_Y, BLOCK_X*BLOCK_Y*THREAD_ELEMENT_X*THREAD_ELEMENT_Y);

	int rowLengthInit = GRID_X*BLOCK_X*THREAD_ELEMENT_X;
	int rowLengthOut = GRID_Y*BLOCK_Y*THREAD_ELEMENT_Y;

	printf("row lenght init = %d row lenght out = %d\n", rowLengthInit, rowLengthOut);

	cudaCheckStatus(cudaEventRecord(start, 0));
	matrixRebuild << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix, rowLengthInit);

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaDeviceSynchronize());
	cudaCheckStatus(cudaEventRecord(stop, 0));
	cudaCheckStatus(cudaEventSynchronize(stop));

	float elapsedTime;
	cudaCheckStatus(cudaEventElapsedTime(&elapsedTime, start, stop));

	cudaCheckStatus(cudaEventDestroy(start));
	cudaCheckStatus(cudaEventDestroy(stop));

	printf("The time for cuda with global memory spend: %.3f ms\n", elapsedTime);

	cudaCheckStatus(cudaMemcpy(initMatrix, deviceInMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheckStatus(cudaMemcpy(cuda_outMatrix, deviceOutMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost));

	cudaCheckStatus(cudaFree(deviceInMatrix));
	cudaCheckStatus(cudaFree(deviceOutMatrix));
}

void cudaShredWork(int* initMatrix, int* cuda_outMatrix) {

	int* deviceInMatrix;
	int* deviceOutMatrix;

	cudaEvent_t start, stop;
	cudaCheckStatus(cudaEventCreate(&start));
	cudaCheckStatus(cudaEventCreate(&stop));

	cudaCheckStatus(cudaMalloc(&deviceInMatrix, (SIZE_M*SIZE_N * sizeof(int))));
	cudaCheckStatus(cudaMalloc(&deviceOutMatrix, (SIZE_N*SIZE_M * sizeof(int))));
	cudaCheckStatus(cudaMemcpy(deviceInMatrix, initMatrix, SIZE_M*SIZE_N * sizeof(int), cudaMemcpyHostToDevice));

	dim3 dimGrid(GRID_X, GRID_Y);
	dim3 dimBlock(BLOCK_X, BLOCK_Y);

	printf("gridX = %d gridY = %d\n", dimGrid.x, dimGrid.y);
	printf("blockX = %d blockY = %d\n", dimBlock.x, dimBlock.y);
	printf("threads in block = %d elements in block = %d\n", BLOCK_X*BLOCK_Y, BLOCK_X*BLOCK_Y*THREAD_ELEMENT_X*THREAD_ELEMENT_Y);

	int rowLengthInit = GRID_X*BLOCK_X*THREAD_ELEMENT_X;
	int rowLengthOut = GRID_Y*BLOCK_Y*THREAD_ELEMENT_Y;

	printf("row lenght init = %d row lenght out = %d\n", rowLengthInit, rowLengthOut);

	cudaCheckStatus(cudaEventRecord(start, 0));
	matrixRebuildShared << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix, rowLengthInit);

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaDeviceSynchronize());
	cudaCheckStatus(cudaEventRecord(stop, 0));
	cudaCheckStatus(cudaEventSynchronize(stop));

	float elapsedTime;
	cudaCheckStatus(cudaEventElapsedTime(&elapsedTime, start, stop));

	cudaCheckStatus(cudaEventDestroy(start));
	cudaCheckStatus(cudaEventDestroy(stop));

	printf("The time for cuda with global memory spend: %.3f ms\n", elapsedTime);

	cudaCheckStatus(cudaMemcpy(initMatrix, deviceInMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheckStatus(cudaMemcpy(cuda_outMatrix, deviceOutMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost));

	cudaCheckStatus(cudaFree(deviceInMatrix));
	cudaCheckStatus(cudaFree(deviceOutMatrix));
}

bool compareOutMatrix(int* first, int* second)
{

	for (int i = 0; i < SIZE_N*SIZE_M; ++i)
	{
		if (first[i] != second[i])
		{
			printf("position = %d, GPU =  %d, CPU = %d ", i, first[i], second[i]);
			return false;
		}
	}

	return true;
}

int main()
{
	/*int *a, *b, *c;

	cudaMallocManaged(&a,SIZE * sizeof(int));
	cudaMallocManaged(&b,SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = i;
	}

	vectorAdd <<<1,SIZE>>> (a, b, c, SIZE);

	cudaDeviceSynchronize();

	for (int i = 0; i < 10; ++i)
	{
		printf("c[%d] = %d\n", i, c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);*/

	int* initMatrix = (int*)calloc(SIZE_M*SIZE_N,sizeof(int));
	int* cpu_outMatrix = (int*)calloc(SIZE_N*SIZE_M,sizeof(int));
	int* cuda_outMatrix = (int*)calloc(SIZE_N*SIZE_M,sizeof(int));
	int* cuda_outMatrixSharedMemory = (int*)calloc(SIZE_N*SIZE_M,sizeof(int));

	fillMatrix(initMatrix, SIZE_M, SIZE_N);

	//printfFirstNForInit(initMatrix, 10);
	//printfFirstNForOut(cuda_outMatrix, 10);

	cpuWork(initMatrix, cpu_outMatrix);

	//printfFirstNForInit(initMatrix, 10);
	//printfFirstNForOut(cpu_outMatrix, 10);

	cudaWork(initMatrix, cuda_outMatrix);

	//printfFirstNForInit(initMatrix, 10);
	printfFirstNForOut(cuda_outMatrix, 10);

	cudaShredWork(initMatrix, cuda_outMatrixSharedMemory);

	printfFirstNForInit(initMatrix, 10);
	printfFirstNForOut(cuda_outMatrixSharedMemory, 10);

	printf("\n\n RESULT OF COMPARE CUDA AND CPU: %s\n", (compareOutMatrix(cuda_outMatrix, cpu_outMatrix) ? "+" : "-"));
	printf("\n\n RESULT OF COMPARE CUDA SHARED AND CPU: %s\n", (compareOutMatrix(cuda_outMatrixSharedMemory, cpu_outMatrix) ? "+" : "-"));

	free(initMatrix);
	free(cpu_outMatrix);
	free(cuda_outMatrix);
	free(cuda_outMatrixSharedMemory);

	return 0;
}

