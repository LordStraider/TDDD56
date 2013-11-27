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
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

void par_merge_sort(void* arg){
  sort_args_t *args = (sort_args_t*) arg;

}

void par_merge(void* arg){
  sort_args_t *args = (sort_args_t*) arg;

}

void seq_merge(void* arg){
  sort_args_t *args = (sort_args_t*) arg;

}

struct sort_args
{
  struct array * array;
};
typedef struct sort_args sort_args_t;

int
sort(struct array * array)
{
    
    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    sort_args args[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    pthread_mutex_t lock;

    size_t counter;

    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);

    for (i = 0; i < NB_THREADS; i++)
    {
        args[i].array = array;
        pthread_create(&thread[i], &attr, &par_merge_sort, (void*) &args[i]);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    simple_quicksort_ascending(array);

    return 0;
}

