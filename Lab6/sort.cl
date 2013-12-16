/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

__kernel void sort(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int i;
  unsigned int val;

	//__local float myBuffer[get_global_size(0)];
  //find out how many values are smaller
  for (i = 0; i < get_global_size(0); i++)
    if (data[get_global_id(0)] > data[i])
      pos++;
	
  val = data[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
  data[pos]=val;
}
