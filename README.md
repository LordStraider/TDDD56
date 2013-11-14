TDDC56
======

labboration 1
Q: Write a detailed explanation why computation load can be imbalanced and how it affects the global performance.

A: Since a lot of iterations are required in the mandelbrot set to know that the pixel should be black, it takes more calculations than it the pixel isn't in the mandelbrot. This makes the number of calculations very imbalanced throught the picture.

Q: Describe a load-balancing method that would help reducing the performance loss due to load-imbalance.

A: Make some processors take a bigger part of the picture then others.

labboration 1

Q: Write an explaination on how CAS can be used to implement protection for concur- rent use of data structures

A: We have a CAS that we use to see if someone has changed the data since last time. If the data haven't been changed we can change it.

Q: Sketch a scenario featuring several threads raising the ABA problem

A: We have a stack where threads can push and pop tasks.

Thread 1 :
	old -> Null
	new -> Null
	pool -> Null
Thread 2 :
	old -> Null
	new -> Null
	pool -> Null
Thread 3 :
	old -> Null
	new -> Null
	pool -> Null
Stack -> A -> B -> C -> Null

Thread 1 and 2 concurrently pops from the thread.
(due to thread 1 ran out of time in the middle of popping, thus both threads succeed.)
Thread 1 :
	old -> A
	new -> B
	pool -> Null
Thread 2 :
	old -> A
	new -> B
	pool -> A -> Null
Thread 3 :
	old -> Null
	new -> Null
	pool -> Null
Stack -> B -> C -> Null


Thread 1 :
	old -> A
	new -> B
	pool -> Null
Thread 2 :
	old -> A
	new -> B
	pool -> A -> Null
Thread 3 :
	old -> B
	new -> C
	pool -> B -> Null
Stack -> C -> Null


Thread 1 :
	old -> A
	new -> B
	pool -> Null
Thread 2 :
	old -> C
	new -> A
	pool -> Null
Thread 3 :
	old -> B
	new -> C
	pool -> B -> Null
Stack -> A -> C -> Null


Thread 1 :
	old -> A
	new -> B
	pool -> A -> Null
Thread 2 :
	old -> C
	new -> A
	pool -> Null
Thread 3 :
	old -> B
	new -> C
	pool -> B -> Null
Stack -> B -> Null

The shared stack should be empty, but it
points to B in Thread 3’s recycling bin

