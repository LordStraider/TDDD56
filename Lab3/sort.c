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
#define NUMMER_AV_THREADS 2
void par_merge_sort(void* arg){
  sort_args_t *args = (sort_args_t*) arg;
  value* data = args->array->data;
  
  printf("thread id: %d, start: %d, stop: %d\n", args->id, args->start, args->stop);

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

void swap(int * a, int * b){
  int * tmp = a;
  *a = *b;
  *b = *tmp;
}

void par_merge(void* arg){
  sort_args_t *args = (sort_args_t*) arg;
  
}

void seq_merge(void* arg){
  sort_args_t *args = (sort_args_t*) arg;

}

int
sort(struct array * array)
{
    pthread_attr_t attr;
    pthread_t thread[NUMMER_AV_THREADS];
    sort_args_t args[NUMMER_AV_THREADS];
    pthread_mutexattr_t mutex_attr;
    pthread_mutex_t lock;

    size_t counter;

    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);
    
    printf("---------\n");  
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }
    printf("---------\n");

    float chunkSize = array->length;
    for (i = 0; i < NUMMER_AV_THREADS; i++)
    {
        args[i].array = array;
        args[i].id = i;
        args[i].length = array->length; // / NUMMER_AV_THREADS;
        args[i].start = ceil((chunkSize / NUMMER_AV_THREADS) * i);
        args[i].stop = ceil((chunkSize / NUMMER_AV_THREADS) * (i+1));
        pthread_create(&thread[i], &attr, &par_merge_sort, (void*) &args[i]);
    }

    for (i = 0; i < NUMMER_AV_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    simple_quicksort_ascending(array);
    printf("\n-------------\n");
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }

    return 0;
}

