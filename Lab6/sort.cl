/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

__kernel void sort(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int i;
  unsigned int val;

	__local float myBuffer[4096];

	const int n = get_global_size(0);
	int x = get_group_id(0);
	int y = get_group_id(1);
	int idX = get_local_id(0);
	int idY = get_local_id(1);
	int xx = x*n + idX;
	int yy = y*n + idY;
	int index = get_global_id(0); //xx*n+yy;

	myBuffer[index] = data[index];
	barrier(CLK_LOCAL_MEM_FENCE);
	
  //find out how many values are smaller
  for (i = 0; i < n; i++)
    if (myBuffer[index] > data[i])
      pos++;
	
  val = myBuffer[index];
	barrier(CLK_LOCAL_MEM_FENCE);
  data[pos]=val;
}
