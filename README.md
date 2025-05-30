# Approximating Maxima of Lipschitz Continuous Functions

This project is intended as an extension of an article I wrote during my undergraduate education. Given an unknown function having a particular property (Lipschitz continuity) on an interval $[a,b]$, the script `approximating_maxima.py` provides the optimal choice of which $x$-values the function should be evaluated at to approximate its maximum value on the interval.

This README is focussed on explaining the use of the script `approximating_maxima.py` and in particular its class `Approximator`. For more information on the other content in this repository please see the final section of this document.

# Table of Contents
- [Introduction] (#introduction)
- [Instance Variables] (#instance-variables)
- 
- [Other Content] (#other-content)

## Introduction

### Lipschitz Continuity
Suppose $a$ and $b$ are real numbers and $f:[a,b]\to\mathbb{R}$ is a real-valued function defined on the interval $[a,b]$. We say that $f$ is *Lipschitz continuous* if there exists a real number $M$ such that, for all $x_1$ and $x_2$ in the interval $[a,b]$,
$$|f(x_{1})-f(x_{2})|\leq M|x_{1}-x_{2}|.$$
In this case we call $M$ a Lipschitz constraint of the function $f$. 

## Instance Variables

## Methods

## Other Content 


#### Sample Function
When setting the sample function (either at the time of initiation or later) please ensure that:

1. Your input is a string.
2. Your equation has only one variable and it is denoted by `x`.
3. Your equation is formatted Pythonically. i.e. Use <em>`'x*x'`</em> to represent the function $x^2$ rather than using `'x^2'`.


