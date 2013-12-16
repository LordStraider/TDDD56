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

	__local float myBuffer[SIZE];

	int id = get_global_id(0);
	
	val = myBuffer[id];

	for (j = 0; j < length; j+=SIZE){
		myBuffer[id] = data[id*j];
		barrier(CLK_LOCAL_MEM_FENCE);
	
		//find out how many values are smaller
		for (i = 0; i < get_global_size(0); i++)
		  if (myBuffer[id] > myBuffer[id])
		    pos++;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	out[pos]=val;
/*
	for (i = 0; i < get_global_size(0); i++)
		  if (data[get_global_id(0)] > data[i])
		    pos++;
	
		val = data[get_global_id(0)];
		out[pos]=val;*/
}
