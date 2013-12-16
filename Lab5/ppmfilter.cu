
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

#define BLOCKDIM 32

__device__ void setPixel(unsigned char* out, unsigned char* gpu_img, unsigned char* source, int local_index, int global_index){
  gpu_img[local_index + 0] = source[global_index + 0];
  gpu_img[local_index + 1] = source[global_index + 1];
  gpu_img[local_index + 2] = source[global_index + 2];
 /* out[global_index+0] = 255;
	out[global_index+1] = 0;
	out[global_index+2] = 0;*/
}

__global__ void filter(unsigned char *image, unsigned char *out, int m, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //x
	int y = blockIdx.y * blockDim.y + threadIdx.y; //y
	int yy = threadIdx.y+2; //yy
	int xx = threadIdx.x+2; //xx
	int nn = blockDim.x+4;
	int sumx, sumy, sumz, k, l;

// printf is OK under -arch=sm_20 
// printf("%d %d %d %d\n", y, x, n, m);

	__shared__ unsigned char gpu_image [(BLOCKDIM+4) * (BLOCKDIM+4) * 3];


		if (yy <= 3 && y > 1){ // upper border
		  if(xx <= 3 && x > 1){ // left border
		      setPixel(out, gpu_image, image, ((yy-2)*nn+(xx-2))*3, ((y-2)*n+(x-2))*3);
		  } else if (xx >= nn-4 && x < n-2){ // right border
		      setPixel(out, gpu_image, image, ((yy-2)*nn+(xx+2))*3, ((y-2)*n+(x+2))*3);
		  }
		  setPixel(out, gpu_image, image, ((yy-2)*nn+xx)*3, ((y-2)*n+x)*3);
		  
		} else if(yy >= nn-4 && y < n-2){ // lower border
		  if(xx <= 3 && x > 1){ // left border
		      setPixel(out, gpu_image, image, ((yy+2)*nn+(xx-2))*3, ((y+2)*n+(x-2))*3);
		  } else if (xx >= nn-4 && x < n-2){ // right border
		      setPixel(out, gpu_image, image, ((yy+2)*nn+(xx+2))*3, ((y+2)*n+(x+2))*3);
		  }
		  setPixel(out, gpu_image, image, ((yy+2)*nn+xx)*3, ((y+2)*n+x)*3);

		}

		if(xx <= 3 && x > 1){ // left border
		  setPixel(out, gpu_image, image, (yy*nn+xx-2)*3, (y*n+x-2)*3);
		} else if(xx >= nn-4 && x < n-2){ // right border
		  setPixel(out, gpu_image, image, (yy*nn+xx+2)*3, (y*n+x+2)*3);
		}
		
		
	  setPixel(out, gpu_image, image, (yy*nn+xx)*3, (y*n+x)*3);

	__syncthreads();
	
		// Filter kernel
		sumx=0;sumy=0;sumz=0;
		for(k=-2;k<3;k++)
			for(l=-2;l<3;l++)
			{
				sumx += gpu_image[((yy + k) * nn + (xx + l)) * 3 + 0];
				sumy += gpu_image[((yy + k) * nn + (xx + l)) * 3 + 1];
				sumz += gpu_image[((yy + k) * nn + (xx + l)) * 3 + 2];
			}
		out[(y*n+x)*3+0] = sumx/25;
		out[(y*n+x)*3+1] = sumy/25;
		out[(y*n+x)*3+2] = sumz/25;
 		
	//__syncthreads();

}

__global__ void filterNaive(unsigned char *image, unsigned char *out, int n, int m)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int sumx, sumy, sumz, k, l;

// printf is OK under --device-emulation
//        printf("%d %d %d %d\n", i, j, n, m);

        if (j < n && i < m)
        {
                out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
                out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
                out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
        }
        
        if (i > 1 && i < m-2 && j > 1 && j < n-2)
                {
                        // Filter kernel
                        sumx=0;sumy=0;sumz=0;
                        for(k=-2;k<3;k++)
                                for(l=-2;l<3;l++)
                                {
                                        sumx += image[((i+k)*n+(j+l))*3+0];
                                        sumy += image[((i+k)*n+(j+l))*3+1];
                                        sumz += image[((i+k)*n+(j+l))*3+2];
                                }
                        out[(i*n+j)*3+0] = sumx/25;
                        out[(i*n+j)*3+1] = sumy/25;
                        out[(i*n+j)*3+2] = sumz/25;
                }
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
	dim3 dimGrid( 512/BLOCKDIM, 512/BLOCKDIM );
	//dim3 dimBlock( 512/8 , 512/8);
	//dim3 dimGrid( 8, 8 );
	

  cudaEvent_t myEvent, myEvent2;
  cudaEventCreate(&myEvent);
  cudaEventCreate(&myEvent2);
  cudaEventRecord(myEvent, 0);
  cudaEventSynchronize(myEvent);


	filter<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();

  cudaEventRecord(myEvent2, 0);
  cudaEventSynchronize(myEvent2);
	float theTime;
  cudaEventElapsedTime(&theTime, myEvent, myEvent2);
	printf("The gpu calculation (optimized) took: %0.2f ms\n", theTime);


  cudaEventCreate(&myEvent);
  cudaEventCreate(&myEvent2);
  cudaEventRecord(myEvent, 0);
  cudaEventSynchronize(myEvent);
	filterNaive<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();

  cudaEventRecord(myEvent2, 0);
  cudaEventSynchronize(myEvent2);
  cudaEventElapsedTime(&theTime, myEvent, myEvent2);
	printf("The gpu calculation (naive) took: %0.2f ms\n", theTime);


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
