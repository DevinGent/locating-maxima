# Approximating Maxima of Lipschitz Continuous Functions

## Introduction

### Lipschitz Continuity
Suppose $a$ and $b$ are real numbers and $f:[a,b]\to\mathbb{R}$ is a real-valued function defined on the interval $[a,b]$. We say that $f$ is *Lipschitz continuous* if there exists a real number $M$ such that, for all $x_1$ and $x_2$ in the interval $[a,b]$,
$$|f(x_{1})-f(x_{2})|\leq K|x_{1}-x_{2}|.$$
In this case we call $M$ a Lipschitz constraint of the function $f$. 

## `approximating_maxima.py`

#### Sample Function
When setting the sample function (either at the time of instantiation as follows) please ensure that:

1. Your input is a string.
2. Your equation has only one variable and it is denoted by `x`.
3. Your equation is formatted Pythonically. i.e. Use <em>`'x*x'`</em> to represent the function $x^2$ rather than using `'x^2'`.


