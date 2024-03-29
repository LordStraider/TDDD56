/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H


struct stack
{
  // This is a fake structure; change it to your needs
	struct stack* next;
	void* data;
};

typedef struct stack stack_t;

// Pushes an element in a thread-safe manner
int stack_push_safe(stack_t *, void*);
// Pops an element in a thread-safe manner
int stack_pop_safe(stack_t *, void*);
int aba_test_stack_pop(stack_t *, stack_t **, int id);
int aba_test_stack_push(stack_t *, stack_t *);
int stack_pop(stack_t *stack, void* buffer);
int stack_push(stack_t *stack, void* buffer);
stack_t*
stack_alloc();

#endif /* STACK_H */
