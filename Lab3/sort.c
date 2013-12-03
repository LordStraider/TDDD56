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
};
typedef struct sort_args sort_args_t;

void SeqMergesort(value* data, int n);
void SeqMerge(value* data, int middle, int end, value* result);
pthread_attr_t attr;
pthread_t thread[NB_THREADS];
sort_args_t args[NB_THREADS];
int numb_threads_created; 
pthread_mutex_t lock;

//int MY_NB_THREADS = 3;
void SParMergesort (void* arg){
    sort_args_t *args = (sort_args_t*) arg;
    value* data = args->data;
    float n = args->n;
    if (n==1) {
        return; // nothing to sort
    } 
    
    //pthread_mutex_lock(&lock);

    if (numb_threads_created >= NB_THREADS - 1) {
				//pthread_mutex_unlock(&lock);

        SeqMergesort(data, (int)n); // switch to sequential
    } else {
        // parallel divide and conquer:
        pthread_mutex_lock(&lock);
        numb_threads_created++; //mutex lås på denna
				
				pthread_mutex_unlock(&lock);
				int halfSize = (int)ceil(n / 2);
				args[new_thread].id = new_thread;
        args[new_thread].n = n - halfSize;
        args[new_thread].data = data + halfSize;
        pthread_create(&thread[new_thread], &attr, &SParMergesort, (void*) &args[new_thread]);

        args[args->id].n = halfSize;
        SParMergesort(&args[0]);

        pthread_join(thread[new_thread], NULL);

        value * result = (value*) malloc(n * sizeof(value));
        int middle = halfSize;
        //printf("n: %d, middle: %d \n", n, middle);
        SeqMerge(data, middle, (int)n, result);
        memcpy(data, result, n * sizeof(value));
        free(result);
    }
}

void SeqMergesort(value* data, int n) {
    if(n < 2)
        return;
    
    float length = n;
    int middle = ceil(length / 2);
    //printf("midd: %d, n: %d, data: %X\n", middle, n, data);
    SeqMergesort(data, middle);
    SeqMergesort(data+middle, n-middle); 
    
    //printf("s: merging %d to %d with %d to %d\n",, middle, middle, n);  
    value * result = (value*) malloc(n * sizeof(value));

    if(result == NULL){
        printf("MALLOC FAILED\n");
    }

    SeqMerge(data, middle, n, result);

    memcpy(data, result, n * sizeof(value));
    free(result);
}

void SeqMerge(value* data, int middle, int end, value* result) {
    int i0 = 0;
    int i1 = middle;
    int j;
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


    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);

    args[numb_threads_created].id = numb_threads_created;
    args[numb_threads_created].n = array->length;
    args[numb_threads_created].data = array->data;

    SParMergesort(&args[0]);

   /*printf("\n-------------\n");
    for (i = 0; i < 10; i++) {
      printf("data[%d] = %d\n", i, array->data[i]);
    }*/

    return 0;
}

