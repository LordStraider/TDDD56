/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
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

#ifndef DEBUG
#define NDEBUG
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;

void stack_dump() {
  printf("\n\nhead :: ");
  stack_t* current = stack->next;
  while (current != NULL) {
    printf("%c->", *(char*)current->data);
    current = current->next;
  }
  printf("\n\n");
}

void
test_init()
{
  // Initialize your test batch
  //stack_init(stack, 1);
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;
  stack = stack_alloc();
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks as now
  free(stack);
  stack->next = NULL;
}

void
test_finalize()
{
  // Destroy properly your test batch
  
}

void* push_safe(void* arg) {
	char buffer = 'A';
  int i;
	for (i = 0; i < MAX_PUSH_POP; i++) {
		    
		stack_push(stack, &buffer);
	}
  return NULL;
}

void* pop_safe(void* arg) {
  char buffer;
  int i;
  for (i = 0; i < MAX_PUSH_POP; i++) {
    stack_pop(stack, &buffer);
  }
  return NULL;
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++) {
   pthread_create(&thread[i], &attr, &push_safe, NULL);
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }
	
  char buffer;
  while (stack->next != NULL) {
    stack_pop(stack, &buffer);
    counter ++;
  }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success) {
    printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
  } 

  assert(success);

  return success;
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  int counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  char buffer = 'A';
  for (i = 0; i < NB_THREADS * MAX_PUSH_POP; i++) {
    stack_push(stack, &buffer);
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_create(&thread[i], &attr, &pop_safe, NULL);
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }

  success = stack->next == NULL;

  if (!success) {
    printf("Got %i, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
  }

  assert(success);

  return success;
}



// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3
void* thread_test_aba (void* args) {
  int id = *(int*)args;
  
  //printf("before pop: Thread %d buffer %p\n", id, buffer);
  stack_t* elem;
  int i,j;
  switch(id){
    case 0:
      aba_test_stack_pop(stack, &elem, id);
      printf("Thread %d popping %c\n", id, *(char*)elem->data);
      break;
    case 1:
      for(i = 0; i < 5000; i++){j++;}
      aba_test_stack_pop(stack, &elem, id);
      printf("Thread %d popping %c at address %X\n", id, *(char*) elem->data, elem);
      for(i = 0; i < 80000; i++){j++;}
      aba_test_stack_push(stack, elem);
      printf("Thread %d pushing %c at address %X\n", id, *(char*) elem->data, elem);
      break;
    case 2:
      for(i = 0; i < 50000; i++){j++;}
      aba_test_stack_pop(stack, &elem, id);
      printf("Thread %d popping %c\n", id, *(char*)elem->data);
      break;
  }
  //printf("after  pop: Thread %d buffer %X char: %c\n", id, buffer, *buffer);
  //free(buffer);
  return 0;
}

int
test_aba()
{
  int success, aba_detected = 0;
  // Write here a test for the ABA problem


  pthread_attr_t attr;
  pthread_t thread[ABA_NB_THREADS];
  int args[ABA_NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  int i;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  char buffer[3];
  buffer[0] = 'C';
  stack_push(stack, &buffer[0]);
  buffer[1] = 'B';
  stack_push(stack, &buffer[1]);
  buffer[2] = 'A';
  stack_push(stack, &buffer[2]);

  stack_dump();

  for (i = 0; i < ABA_NB_THREADS; i++)
    {
      args[i] = i;
      pthread_create(&thread[i], &attr, &thread_test_aba, (void*) &args[i]);
    }

  for (i = 0; i < ABA_NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  
  stack_dump();
  aba_detected = (*(char*)stack->next->data == 'B');
  success = aba_detected;
  return success;
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
      } while (cas(args->counter, old, local) != old);
    }

  return NULL;
}

int
test_cas()
{
#if 1
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
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
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);

  return success;
#else
  int a, b, c, *a_p, res;
  a = 1;
  b = 2;
  c = 3;

  a_p = &a;

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %d\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int) res);

  res = cas((void**)&a_p, (void*)&c, (void*)&b);

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %X\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int)res);

  return 0;
#endif
}

// Stack performance test
#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;
#endif

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  // Run performance tests
  stack_measure_arg_t arg[NB_THREADS];  

  test_init();
  test_setup();

  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  int i, counter, buffer;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

#if MEASURE == 2
  for (i = 0; i < NB_THREADS * MAX_PUSH_POP; i++) {
      stack_push(stack, &data);
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++) {
      arg[i].id = i;
      (void)arg[i].id; // Makes the compiler to shut up about unused variable arg
      // Run push-based performance test based on MEASURE token
#if MEASURE == 1
    // Push MAX_PUSH_POP times in parallel
    clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
    pthread_create(&thread[i], &attr, &push_safe, (void*) &args[i]);
    clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#else
    // Run pop-based performance test based on MEASURE token
    clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
    pthread_create(&thread[i], &attr, &pop_safe, (void*) &args[i]);
    clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#endif
    }

    for (i = 0; i < NB_THREADS; i++) {
        pthread_join(thread[i], NULL);
    }
  // Wait for all threads to finish
  clock_gettime(CLOCK_MONOTONIC, &stop);


#if MEASURE == 1
/*  while (stack->next->next != NULL) {
      stack_pop(stack->next, &buffer);      
      counter ++;
      //stack = stack->next;
 printf("measure 1 %d, %d\n", counter, MAX_PUSH_POP);
  }*/
 #endif


  test_finalize();

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
