/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

#define SIZE 512

__kernel void sort(__global unsigned int *data, __global unsigned int *out, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int i, j;
  unsigned int val;
  /*
	__local unsigned int myBuffer[SIZE];

	int id = get_local_id(0);
	int size = get_local_size(0);
	
	val = data[get_global_id(0)];

	for (j = 0; j < length; j+=get_local_size(0)){
		myBuffer[id] = data[id + j];
		barrier(CLK_GLOBAL_MEM_FENCE);
	
		//find out how many values are smaller
		for (i = 0; i < get_local_size(0); i++)
		  if (val > myBuffer[i])
		    pos++;

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	out[pos]=val;
*/
	for (i = 0; i < get_global_size(0); i++)
		  if (data[get_global_id(0)] > data[i])
		    pos++;
	
		val = data[get_global_id(0)];
		out[pos]=val;
}
