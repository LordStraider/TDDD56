TDDC56
======

labboration 1
Q: Write a detailed explanation why computation load can be imbalanced and how it affects the global performance.

A: Since a lot of iterations are required in the mandelbrot set to know that the pixel should be black, it takes more calculations than it the pixel isn't in the mandelbrot. This makes the number of calculations very imbalanced throught the picture.

Q: Describe a load-balancing method that would help reducing the performance loss due to load-imbalance.

A: Make some processors take a bigger part of the picture then others.

