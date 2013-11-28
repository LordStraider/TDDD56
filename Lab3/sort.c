/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// Do not touch or move these lines
#include <stdio.h>
#include <pthread.h>
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

struct sort_args
{
  int id;
  int length;
  int start;
  int stop;
  value* data;
};
typedef struct sort_args sort_args_t;
#define NB_THREADS 4
void par_merge_sort(void* arg){
  sort_args_t *args = (sort_args_t*) arg;
  value* data = args->data;
  
	//printf("thread id: %d, start: %d, stop: %d\n, length: %d\n", args->id, args->start, args->stop, args->length);
	
  value * result = (value*) malloc(args->length * sizeof(value)) ;
	if(result == NULL){
	//	printf("MALLOC FAILED\n");
	}

  split(data, 0, args->length, result);

	free(result);
}

void split(value* data, int begin, int end, value* result){
  if(end - begin < 2)
    return;

  float length = end-begin;
  int middle = ceil(begin + length / 2);
  split(data, begin, middle, result);
  split(data, middle, end, result);  
	
	//printf("s: merging %d to %d with %d to %d\n", begin, middle, middle, end);     
	merge(data, begin, middle, end, result);
  
  memcpy(data + begin, result + begin, (end - begin)*sizeof(value));
}

void recursive_merging(value* data, int begin, int end, value* result, int chunk_size){

  if(end - begin <= chunk_size)
    return;

  float length = end - begin;

  int middle = ceil(begin + length / 2);

  recursive_merging(data, begin, middle, result, chunk_size);
  recursive_merging(data, middle, end, result, chunk_size);

  merge(data, begin, middle, end, result);
  memcpy(data + begin, result + begin, (end - begin)*sizeof(value));
    
}

void merge(value* data, int begin, int middle, int end, value* result){
  int i0 = begin;
  int i1 = middle;  
  int j;

  // While there are elements in the left or right runs
  for (j = begin; j < end; j++) {
    // If left run head exists and is <= existing right run head.
    if (i0 < middle && (i1 >= end || data[i0] <= data[i1])) {
      result[j] = data[i0++];
    } else
      result[j] = data[i1++];
  }
}

void insSort(value* data, int length) {
  int i, valueToInsert, holePos;
  for (i = 0; i < length; i++) {
    // at the start of the iteration, A[0..i-1] are in sorted order
    // this iteration will insert A[i] into that sorted order
    // save A[i], the value that will be inserted into the array on this iteration
    valueToInsert = data[i];
    // now mark position i as the hole; A[i]=A[holePos] is now empty
    holePos = i;
    // keep moving the hole down until the valueToInsert is larger than 
    // what's just below the hole or the hole has reached the beginning of the array
    while (holePos > 0 && valueToInsert < data[holePos - 1]) { 
      //value to insert doesn't belong where the hole currently is, so shift 
      data[holePos] = data[holePos - 1]; //shift the larger value up
      holePos--;       //move the hole position down
    }
    // hole is in the right position, so put valueToInsert into the hole
    data[holePos] = valueToInsert;
    // A[0..i] are now in sorted order
  }
}


int
sort(struct array * array)
{
    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    sort_args_t args[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    pthread_mutex_t lock;

    size_t counter;

    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);
    
    /*printf("---------\n");  
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }
    printf("---------\n");*/

    float chunkSize = array->length;
    chunkSize = chunkSize/NB_THREADS;
    for (i = 0; i < NB_THREADS; i++)
    {
        args[i].id = i;

        args[i].length = ceil(chunkSize);
        args[i].start = ceil(chunkSize * i);
        args[i].stop = ceil(chunkSize * (i+1));
				args[i].data = array->data+args[i].start;        
				pthread_create(&thread[i], &attr, &par_merge_sort, (void*) &args[i]);
    }

    for (i = 0; i < NB_THREADS;i++)
    {
        pthread_join(thread[i], NULL);
    }

    value * result = (value*) malloc(array->length * sizeof(value));
		if(result == NULL){
		//	printf("final malloc failed!\n");
		}

/*
    for (i = 0; i < NB_THREADS-1; i+=2){
      int begin = args[i].start;
      int middle = args[i].stop;
      int end = args[i+1].stop;
      printf("b: %d, m: %d, e: %d\n", begin, middle, end);
      merge(array->data, begin, middle, end, result);
      memcpy(array->data + begin, result + begin, (end - begin)*sizeof(value));
      
    }*/


    //simple_quicksort_ascending(array);
/*    printf("\n-------------\n");
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }
  */  

   // printf("\n-------------\n");
   // for (i = 0; i < array->length; i++) {
   //   printf("data[%d] = %d\n", i, array->data[i]);
   // }

  recursive_merging(array->data, 0, array->length, result, ceil(chunkSize));

	free(result);

    //insSort(array->data, array->length);
  //printf("Sorted %d elements.\n", array->length);
    //simple_quicksort_ascending(array);

  //  printf("\n-------------\n");
   // for (i = 0; i < array->length; i++) {
   //   printf("data[%d] = %d\n", i, array->data[i]);
   // }

    return 0;
}

