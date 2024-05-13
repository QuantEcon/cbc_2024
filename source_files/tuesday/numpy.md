---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Vectorization and Array Operations with NumPy

### Written for the CBC Workshop (May 2024)

#### John Stachurski

+++

[NumPy](https://numpy.org/) is the standard library for numerical array operations in Python.

+++

This notebook contains a very quick introduction to NumPy.

(Although the syntax and some reference concepts differ, the basic framework is similar to Matlab.)

We use the following imports

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## NumPy arrays

Let's review the basics of NumPy arrays.

### Creating arrays

Here are a few ways to create arrays:

```{code-cell} ipython3
a = (10.0, 20.0)
print(type(a))
a = np.array(a)   # Create array from Python tuple
type(a)
```

```{code-cell} ipython3
a = np.array((10, 20), dtype='float64')  # Specify data type -- must be homogeneous
a
```

```{code-cell} ipython3
a = np.linspace(0, 10, 5)
a
```

```{code-cell} ipython3
a = np.ones(3)
a
```

```{code-cell} ipython3
a = np.zeros(3)
a
```

```{code-cell} ipython3
a = np.random.randn(4)
a
```

```{code-cell} ipython3
a = np.random.randn(2, 2)
a
```

```{code-cell} ipython3
b = np.zeros_like(a)
b
```

### Reshaping

```{code-cell} ipython3
a = np.random.randn(2, 2)
a
```

```{code-cell} ipython3
a.shape
```

```{code-cell} ipython3
np.reshape(a, (1, 4))
```

```{code-cell} ipython3
np.reshape(a, (4, 1))
```

### Array operations

+++

Standard arithmetic operators are pointwise:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
b
```

```{code-cell} ipython3
a + b
```

```{code-cell} ipython3
a * b  # pointwise multiplication
```

### Matrix multiplication

```{code-cell} ipython3
a @ b  
```

```{code-cell} ipython3
np.ones(3) @ np.zeros(3)  # inner product
```

### Reductions

+++

There are various functions for acting on arrays, such as

```{code-cell} ipython3
np.mean(a)
```

```{code-cell} ipython3
np.sum(a)
```

These operations have an equivalent OOP syntax, as in

```{code-cell} ipython3
a.mean()
```

```{code-cell} ipython3
a.sum()
```

These operations also work on higher-dimensional arrays:

```{code-cell} ipython3
a = np.linspace(0, 3, 4).reshape(2, 2)
a
```

```{code-cell} ipython3
a.sum(axis=0)  # sum columns
```

```{code-cell} ipython3
a.sum(axis=1)  # sum rows
```

### Broadcasting

+++

When possible, arrays are "streched" across missing dimensions to perform array operations.

For example,

```{code-cell} ipython3
a = np.zeros((3, 3))
a
```

```{code-cell} ipython3
b = np.array((1.0, 2.0, 3.0))
b = np.reshape(b, (1, 3))
b
```

```{code-cell} ipython3
a + b
```

```{code-cell} ipython3
b = np.reshape(b, (3, 1))
b
```

```{code-cell} ipython3
a + b
```

For more on broadcasting see [this tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).

+++

### Ufuncs

+++

Many NumPy functions can act on either scalars or arrays.

When they act on arrays, they act pointwise (element-by-element).

These kinds of functions are called **universal functions** or **ufuncs**.

```{code-cell} ipython3
np.cos(np.pi)
```

```{code-cell} ipython3
a = np.random.choice((0, np.pi), 6).reshape(2, 3)
a
```

```{code-cell} ipython3
np.cos(a)
```

Some user-defined functions will be ufuncs, such as

```{code-cell} ipython3
def f(x):
    return np.cos(np.sin(x))
```

```{code-cell} ipython3
f(a)
```

But some are not:

```{code-cell} ipython3
def f(x):
    if x < 0:
        return np.cos(x)
    else:
        return np.sin(x)
```

```{code-cell} ipython3
f(a)
```

If we want to turn this into a vectorized function we can use `np.vectorize`

```{code-cell} ipython3
f_vec = np.vectorize(f)
```

Let's test it, and also time it.

```{code-cell} ipython3
a = np.linspace(0, 1, 10_000_000)
%time f_vec(a)
```

This is pretty slow.

Here's a version of `f` that uses NumPy functions to create a more efficient ufunc.

```{code-cell} ipython3
def f(x):
    return np.where(x < 0, np.cos(x), np.sin(x))
```

```{code-cell} ipython3
%time f(a)
```

Moral of the story: Don't use `np.vectorize` unless you have to.

(There are good alternatives, which we will discuss soon.)

+++

### Mutability

+++

NumPy arrays are mutable (can be altered in memory).

```{code-cell} ipython3
a = np.array((10.0, 20.0))
a
```

```{code-cell} ipython3
a[0] = 1
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3
a[:] = 42
```

```{code-cell} ipython3
a
```

**All names** bound to an array have equal rights.

```{code-cell} ipython3
a
```

```{code-cell} ipython3
b = a  # bind the name b to the same array object
```

```{code-cell} ipython3
id(a)
```

```{code-cell} ipython3
id(b)
```

```{code-cell} ipython3
b[0] = 1_000
```

```{code-cell} ipython3
b
```

```{code-cell} ipython3
a
```

## Vectorizing loops

+++

### Accelerating slow loops

In scripting languages, native loops are slow:

```{code-cell} ipython3
n = 10_000_000
x_vec = np.linspace(0.1, 1.1, n)
```

Let's say we want to compute the sum of of $\cos(2\pi / x)$ over $x$ in

```{code-cell} ipython3
%%time
current_sum = 0.0
for x in x_vec:
    current_sum += np.cos(2 * np.pi / x)
```

The reason is that Python, like most high level languages is dynamically typed.

This means that the type of a variable can freely change.

Moreover, the interpreter doesn't compile the whole program at once, so it doesn't know when types will change.

So the interpreter has to check the type of variables before any operation like addition, comparison, etc.

Hence there's a lot of fixed cost for each such operation

+++

The code runs much faster if we use **vectorized** expressions to avoid explicit loops.

```{code-cell} ipython3
%%time
np.sum(np.cos(2 * np.pi / x_vec))
```

Now high level overheads are paid *per array rather than per float*.

+++

### Implict Multithreading


Recent versions of Anaconda are compiled with Intel MKL support, which accelerates NumPy operations.

Watch system resources when you run this code.  

(For example, install `htop` (Linux / Mac), `perfmon` (Windows) or another system load monitor and set it running in another window.)

```{code-cell} ipython3
n = 20
m = 1000
for i in range(n):
    X = np.random.randn(m, m)
    λ = np.linalg.eigvals(X)
```

You should see all your cores light up.  With MKL, many matrix operations are automatically parallelized.
