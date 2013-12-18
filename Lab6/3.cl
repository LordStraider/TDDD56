/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

#define SIZE 512

#define dataWidth 512

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int length)
{
	//__local unsigned int myBuffer[SIZE];
	
	int id = get_local_id(0);
	int size = get_local_size(0);
	
	int gid = get_global_id(0);
	
	int pixel_id = gid / 3;
	
	int y = pixel_id / dataWidth;
	int x = pixel_id % dataWidth;
	int pixel = 0;
	int j;
	int channel = gid % 3;
	
	if (y >= (dataWidth) / 2 && x >= (dataWidth) / 2){ // top right corner = out4
	
	  j = 3*(pixel_id*2) + channel - length + 3*dataWidth;
	  pixel = (image[j] - image[j+3] - image[j+dataWidth*3] + image[j+3+dataWidth*3])/4+128;
	  
	} else if (y >= dataWidth / 2 && x < dataWidth / 2){ // top left corner = out3
	
	  j = 3*(pixel_id*2) + channel - length;
	  pixel = (image[j] - image[j+3] + image[j+dataWidth*3] - image[j+3+dataWidth*3])/4+128;
	  
	} else if (y < dataWidth / 2 && x >= dataWidth / 2){ // lower right corner = out2
	
	  j = 3*(pixel_id*2) + channel + 3*dataWidth;
	  pixel = (image[j] + image[j+3] - image[j+dataWidth*3] - image[j+3+dataWidth*3])/4+128;
	  
	} else if (y < dataWidth / 2 && x < dataWidth / 2){ // lower left corner = out1
	  
	  j = 3*(pixel_id*2) + channel;
	  pixel = (image[j] + image[j+3] + image[j+dataWidth*3] + image[j+3+dataWidth*3])/4;
	  
  } else {
  
    pixel = 0;
  
  }
  
  data[get_global_id(0)] = pixel;
}
