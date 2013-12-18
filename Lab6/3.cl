/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

#define SIZE 512

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int length)
{
	//__local unsigned int myBuffer[SIZE];
	
	int id = get_local_id(0);
	int size = get_local_size(0);
	
  data[get_global_id(0)] = 255 - image[get_global_id(0)];
}
