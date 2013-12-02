// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"


const int N = 1024;
int grids = 64;
int blocksize = N / grids; 


__global__ 
void matrix(float *a, float *b, float *c) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x * N + y;
	c[index] = a[index] + b[index];
}

void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int size = N*N*sizeof(float);
	
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	
	ResetMilli();
	add_matrix(a, b, c, N);	
	int time = GetMilliseconds();
	/*for (int i = 0; i < N; i++)	{
		for (int j = 0; j < N; j++)	{
			printf("%0.2f ", c[i+j*N]);
			c[i+j*N] = -1;
		}
		printf("\n");
	}*/
	printf("cpu took: %d ms", time);
	printf("\n-----------------\n");

	float *cd, *a_g, *b_g;
	cudaMalloc( (void**)&cd, size );
	cudaMalloc( (void**)&a_g, size );
	cudaMalloc( (void**)&b_g, size );
	cudaMemcpy( a_g, a, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( b_g, b, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( cd, c, size, cudaMemcpyHostToDevice ); 
	dim3 dimBlock( blocksize, blocksize);
	dim3 dimGrid( grids, grids );

  cudaEvent_t myEvent, myEvent2;
  cudaEventCreate(&myEvent);
  cudaEventCreate(&myEvent2);

  cudaEventRecord(myEvent, 0);
  cudaEventSynchronize(myEvent);

	matrix<<<dimGrid, dimBlock>>>(a_g, b_g, cd);
	cudaThreadSynchronize();


  cudaEventRecord(myEvent2, 0);
  cudaEventSynchronize(myEvent2);

	float theTime;

  cudaEventElapsedTime(&theTime, myEvent, myEvent2);

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	cudaFree( a_g );
	cudaFree( b_g );
	
	/*for (int i = 0; i < N; i++)	{
		for (int j = 0; j < N; j++)	{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}*/

	printf("The gpu calculation took: %0.2f ms\n", theTime);
}
