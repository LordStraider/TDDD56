
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

#define BLOCKDIM 16

__global__ void filter(unsigned char *image, unsigned char *out, int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ii = threadIdx.y;
	int jj = threadIdx.x;
	int nn = blockDim.x;
	int sumx, sumy, sumz, k, l;

// printf is OK under --device-emulation
//	printf("%d %d %d %d\n", i, j, n, m);

	__shared__ unsigned char gpu_image [BLOCKDIM * BLOCKDIM * 3];

	if (j < n && i < m)
	{
		gpu_image[(ii*nn+jj)*3 + 0] = image[(i*n+j)*3+0];
		gpu_image[(ii*nn+jj)*3 + 1] = image[(i*n+j)*3+1];
		gpu_image[(ii*nn+jj)*3 + 2] = image[(i*n+j)*3+2];
	}
//(i*n+j) = (threadIdx.x+threadIdx.y*blockDim.x)
	__syncthreads();
	
	if (ii > 1 && ii < nn-2 && jj > 1 && jj < nn-2)
		{
			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-2;k<3;k++)
				for(l=-2;l<3;l++)
				{
					sumx += gpu_image[((ii + k) * nn + (jj + l)) * 3 + 0];
					sumy += gpu_image[((ii + k) * nn + (jj + l)) * 3 + 1];
					sumz += gpu_image[((ii + k) * nn + (jj + l)) * 3 + 2];
				}
			out[(i*n+j)*3+0] = sumx/25;
			out[(i*n+j)*3+1] = sumy/25;
			out[(i*n+j)*3+2] = sumz/25;
		}

	__syncthreads();
/*
		if (j < n && i < m)
	{
		out[(i*n+j)*3+0] = gpu_image[(ii*nn+jj)*3 + 0];
		out[(i*n+j)*3+1] = gpu_image[(ii*nn+jj)*3 + 1];
		out[(i*n+j)*3+2] = gpu_image[(ii*nn+jj)*3 + 2];
	} */
}


// Compute CUDA kernel and display image
void Draw()
{
	unsigned char *image, *out;
	int n, m;
	unsigned char *dev_image, *dev_out;
	
	image = readppm("maskros512.ppm", &n, &m);
	out = (unsigned char*) malloc(n*m*3);
	
	cudaMalloc( (void**)&dev_image, n*m*3);
	cudaMalloc( (void**)&dev_out, n*m*3);
	cudaMemcpy( dev_image, image, n*m*3, cudaMemcpyHostToDevice);
	
	dim3 dimBlock( BLOCKDIM, BLOCKDIM );
	dim3 dimGrid( 32, 32 );
	
	filter<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	
	cudaMemcpy( out, dev_out, n*m*3, cudaMemcpyDeviceToHost );
	cudaFree(dev_image);
	cudaFree(dev_out);
	
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glRasterPos2f(-1, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
	glRasterPos2i(0, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( 1024, 512 );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);
	
	glutMainLoop();
}
