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
  struct array * array;
};
typedef struct sort_args sort_args_t;
#define MY_NB_THREADS 4
void par_merge_sort(void* arg){
  sort_args_t *args = (sort_args_t*) arg;
  value* data = args->array->data;
  
  //printf("thread id: %d, start: %d, stop: %d\n", args->id, args->start, args->stop);

  value result[args->length];

  split(data, args->start, args->stop, result);
}

void split(value* data, int begin, int end, value* result){
  if(end - begin < 2)
    return;

  float length = end-begin;
  int middle = ceil(begin + length / 2);
  split(data, begin, middle, result);
  split(data, middle, end, result);  
  merge(data, begin, middle, end, result);
  
  memcpy(data + begin, result + begin, (end - begin)*sizeof(value));
}

void recursive_merging(value* data, int begin, int end, value* result){

  if(end - begin < 2)
    return;

  float length = end-begin;
  

  int middle = ceil(begin + length / 2);

  recursive_merging(data, begin, middle, result);
  recursive_merging(data, middle, end, result);
  
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
    if (i0 < middle && (i1 >= end || data[i0] <= data[i1])) { // 
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
    pthread_t thread[MY_NB_THREADS];
    sort_args_t args[MY_NB_THREADS];
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
    chunkSize = ceil(chunkSize/MY_NB_THREADS);
    for (i = 0; i < MY_NB_THREADS; i++)
    {
        args[i].id = i;
        args[i].length = chunkSize;
        args[i].start = chunkSize * i);
        args[i].stop = chunkSize * (i+1));
        args[i].array = array_alloc(chunkSize);
        memcpy(array->data + args[i].start, args[i].array, args[i].length*sizeof(value));
        pthread_create(&thread[i], &attr, &par_merge_sort, (void*) &args[i]);
    }

    for (i = 0; i < MY_NB_THREADS;i++)
    {
        pthread_join(thread[i], NULL);
    }

    for (i = 0; i < MY_NB_THREADS;i++)
    {
        memcpy(args[i].array, array->data + args[i].start, args[i].length*sizeof(value));
        array_free(args[i].array);
    }

    value result[array->length];
/*
    for (i = 0; i < MY_NB_THREADS-1; i+=2){
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
  recursive_merging(array->data, 0, array->length, result);
    //insSort(array->data, array->length);
  printf("Sorted %d elements.\n", array->length);
    //simple_quicksort_ascending(array);
    /*printf("\n-------------\n");
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }*/

    return 0;
}

