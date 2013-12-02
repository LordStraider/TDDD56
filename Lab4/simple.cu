// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 


__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = threadIdx.x;
}

__global__ 
void cuda_sqrt(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	const int size = N*sizeof(float);

	float *c = new float[N];	
	float *input;
	cudaMalloc((void**)&input, size*sizeof(float));

	float *cd;
	cudaMalloc( (void**)&cd, size );
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	cudaMemcpy( input, c, size, cudaMemcpyHostToDevice ); 
	cuda_sqrt<<<dimGrid, dimBlock>>>(input);
	cudaThreadSynchronize();
	cudaMemcpy( c, input, size, cudaMemcpyDeviceToHost ); 
	cudaFree( input );

	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");
	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
