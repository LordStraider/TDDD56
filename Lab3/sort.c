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
    int n;
    value* data;
		value* result;
};
typedef struct sort_args sort_args_t;

pthread_attr_t attr;
pthread_t thread[NB_THREADS];
sort_args_t args[NB_THREADS];

/*void par_merge_sort(void* arg){
    sort_args_t *args = (sort_args_t*) arg;
    value* data = args->data;

    //printf("thread id: %d, start: %d, stop: %d\n, length: %d\n", args->id, args->start, args->stop, args->length);

    value * result = (value*) malloc(args->length * sizeof(value)) ;
    if(result == NULL){
        printf("MALLOC FAILED\n");
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
        } else {
           result[j] = data[i1++];
        }
    }
}*/

int numb_threads_created; //låst variabel

    pthread_mutex_t lock;

//int MY_NB_THREADS = 3;
void SParMergesort (void* arg){
    sort_args_t *args = (sort_args_t*) arg;
    value* data = args->data;
		value* result = args->result;
    float n = args->n;
    if (n < 2) {
        return; // nothing to sort
    } 
    
		pthread_mutex_lock(&lock);
		printf("thread %d has %d elements to sort\n", args->id, args->n);
    if (numb_threads_created >= NB_THREADS - 1 || n < 20) {
				printf("thread %d working seq\n", args->id);				
				pthread_mutex_unlock(&lock);
		
 // switch to sequential
				int halfSize = (int)ceil(n / 2);
				args->n = halfSize;
        SParMergesort(&args);
				args->n = n - halfSize;
        args->data = data + halfSize;
        args->result = result + halfSize;
				SParMergesort(&args);
        SeqMerge(data, halfSize, (int)n, result);
        memcpy(data, result, n * sizeof(value));
    } else {
      //  parallel divide and conquer:
        numb_threads_created++;
				int new_thread = numb_threads_created;
				printf("thread %d spawns %d\n", args->id, new_thread);				

				pthread_mutex_unlock(&lock);
				int halfSize = (int)ceil(n / 2);
				args[new_thread].id = new_thread;
        args[new_thread].n = n - halfSize;
        args[new_thread].data = data + halfSize;
        args[new_thread].result = result + halfSize;

        pthread_create(&thread[new_thread], &attr, &SParMergesort, (void*) &args[new_thread]);

        args[args->id].n = halfSize;
        SParMergesort(&args[0]);

        pthread_join(thread[new_thread], NULL);
				
				pthread_mutex_lock(&lock);
        numb_threads_created--;
				pthread_mutex_unlock(&lock);

        int middle = halfSize;
        //printf("n: %d, middle: %d \n", n, middle);
        SeqMerge(data, middle, (int)n, result);
        memcpy(data, result, n * sizeof(value));

    }
}

void SeqMergesort(value* data, value* result, int n) {
    if(n < 2)
        return;
    
    float length = n;
    int middle = ceil(length / 2);
    //printf("midd: %d, n: %d, data: %X\n", middle, n, data);
    SeqMergesort(data, result, middle);
    SeqMergesort(data+middle, result+middle, n-middle); 
    
    //printf("s: merging %d to %d with %d to %d\n",, middle, middle, n);  

    if(result == NULL){
        printf("MALLOC FAILED\n");
    }

    SeqMerge(data, middle, n, result);

    memcpy(data, result, n * sizeof(value));
}

void SeqMerge(value* data, int middle, int end, value* result){
    int i0 = 0;
    int i1 = middle;
    int j;
//  if (data[0] == 0)
//printf("midd: %d, end: %d, data[0] = %d\n", middle, end, data[0]);
    // While there are elements in the left or right runs
    for (j = 0; j < end; j++) {
        // If left run head exists and is <= existing right run head.
        if (i0 < middle && (i1 >= end || data[i0] <= data[i1])) {
            result[j] = data[i0++];
        } else {
           result[j] = data[i1++];
        }
    }
}

int
sort(struct array * array)
{
    numb_threads_created = 0; 
    pthread_mutexattr_t mutex_attr;


    int i;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);


    value * result = (value*) malloc(array->length * sizeof(value));

    args[numb_threads_created].id = numb_threads_created;
    args[numb_threads_created].n = array->length;
    args[numb_threads_created].data = array->data;
		args[numb_threads_created].result = result;
    SParMergesort(&args[0]);
		free(result);
    /*printf("---------\n");  
    for (i = 0; i < array->length; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }
    printf("---------\n");*/
/*
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
*/
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

  //recursive_merging(array->data, 0, array->length, result, ceil(chunkSize));

	//free(result);

    //insSort(array->data, array->length);
  //printf("Sorted %d elements.\n", array->length);
    //simple_quicksort_ascending(array);

   /*printf("\n-------------\n");
    for (i = 0; i < 10; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }*/

    return 0;
}

