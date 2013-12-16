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

	myBuffer[xx*n+yy] = data[xx*n+yy];
	barrier(CLK_GLOBAL_MEM_FENCE);
	
  //find out how many values are smaller
  for (i = 0; i < n; i++)
    if (myBuffer[xx*n+yy] > myBuffer[i])
      pos++;
	
  val = myBuffer[xx*n+yy];
	barrier(CLK_LOCAL_MEM_FENCE);
  data[pos]=val;
}
