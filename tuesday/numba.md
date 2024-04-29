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

# Accelerating Python Code with Numba

### Written for the CBC Quantitative Economics Workshop (September 2022)

+++

#### John Stachurski

+++

This notebook contains a very quick introduction to Numba, as well as comparing Numba to NumPy.

We use the following imports

```{code-cell} ipython3
---
nbpresent:
  id: cacd76f0-600a-4ac9-ba39-ae23747177c8
slideshow:
  slide_type: '-'
---
import numpy as np
from numba import vectorize, jit, float64   
from quantecon.util import tic, toc
import matplotlib.pyplot as plt
```

## Vectorization

+++

In scripting languages, native loops are slow:

```{code-cell} ipython3
n = 10_000_000
x_vec = np.linspace(0.1, 1.1, n)
```

Let's say we want to compute the sum of of $\cos(2\pi / x)$ over $x$ in 

```{code-cell} ipython3
tic()
current_sum = 0.0
for x in x_vec:
    current_sum += np.cos(2 * np.pi / x)
toc()
```

The reason is that Python, like most high level languages is dynamically typed.

This means that the type of a variable can freely change.

Moreover, the interpreter doesn't compile the whole program at once, so it doesn't know when types will change.

So the interpreter has to check the type of variables before any operation like addition, comparison, etc.

Hence there's a lot of fixed cost for each such operation

+++

The code runs much faster if we use **vectorized** expressions to avoid explicit loops.

```{code-cell} ipython3
tic()
np.sum(np.cos(2 * np.pi / x_vec))
toc()
```

Now high level overheads are paid per *array* rather than per float or integer.

+++

#### Implict Multithreading

+++

Recent versions of Anaconda are compiled with Intel MKL support, which accelerates NumPy operations.

+++

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

+++

### Problems with Vectorization

+++

Vectorization is neat but somewhat inflexible.

Some problems cannot be vectorized and need to be written in loops.

In other cases, loops are clearer than vectorized code.

+++

## Numba

+++

Numba circumvents some problems with vectorization by providing fast, efficient just-in-time compilation within Python.  

+++

Consider the time series model

$$ x_{t+1} = \alpha x_t (1 - x_t) $$

Let's set $\alpha = 4$

```{code-cell} ipython3
α = 4.0
```

Here's a typical time series:

```{code-cell} ipython3
n = 200
x =  np.empty(n)
x[0] = 0.2
for t in range(n-1):
    x[t+1] = α * x[t] * (1 - x[t])
    
plt.plot(x)
plt.show()
```

Here's a function that simulates for `n` periods, starting from `x0`, and returns **only the final** value:

```{code-cell} ipython3
def quad(x0, n):
    x = x0
    for i in range(1, n):
        x = α * x * (1 - x)
    return x
```

Let's see how fast this runs:

```{code-cell} ipython3
n = 10_000_000
```

```{code-cell} ipython3
tic()
x = quad(0.2, n)
toc()
```

Now let's try this in FORTRAN.  In the following code, all parameters are the same as for the Python code above.

```{code-cell} ipython3
%%file fastquad.f90

PURE FUNCTION QUAD(X0, N)
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                           
 REAL(dp), INTENT(IN) :: X0
 REAL(dp) :: QUAD
 INTEGER :: I
 INTEGER, INTENT(IN) :: N
 QUAD = X0
 DO I = 1, N - 1                                                
  QUAD = 4.0_dp * QUAD * real(1.0_dp - QUAD, dp)
 END DO
 RETURN
END FUNCTION QUAD

PROGRAM MAIN
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                          
 REAL(dp) :: START, FINISH, X, QUAD
 INTEGER :: N
 N = 10000000
 X = QUAD(0.2_dp, 10)
 CALL CPU_TIME(START)
 X = QUAD(0.2_dp, N)
 CALL CPU_TIME(FINISH)
 PRINT *,'last val = ', X
 PRINT *,'elapsed time = ', FINISH-START
END PROGRAM MAIN
```

**Note** The next step will only execute if you have a FORTRAN compiler installed and modify the compilation code below appropriately.

If you don't then skip execution of the next two cells --- you can just look at my numbers.

```{code-cell} ipython3
!gfortran -O3 fastquad.f90
```

```{code-cell} ipython3
!./a.out
```

Now let's do the same thing in Python using Numba's JIT compilation:

```{code-cell} ipython3
quad_jitted = jit(quad)
```

```{code-cell} ipython3
tic()
x = quad_jitted(0.2, n)
toc()
```

After JIT compilation, function execution speed is about the same as Fortran (and Julia)

+++

#### How does it work?

+++

The secret sauce is type inference inside the function body.

When we call `quad_jitted` with particular arguments, Numba's compiler works through the function body and infers the types of the variables inside the function.

It then produces compiled code *specialized to that type signature*

For example, we called `quad_jitted` with a `float, int` pair in the cell above and the compiler produced code specialized to those types.

That code runs fast because the compiler can fully specialize all operations inside the function based on that information and hence write very efficient machine code.

+++

#### Limitations of Numba

```{code-cell} ipython3
from scipy.integrate import quad


def compute_integral(n):
    return quad(lambda x: x**(1/n), 0, 1)
        
```

```{code-cell} ipython3
compute_integral(4)
```

```{code-cell} ipython3
@jit
def compute_integral(n):
    return quad(lambda x: x**(1/n), 0, 1)
        
```

If we try to run `compute_integral` we will get an error.

```{code-cell} ipython3
import traceback

try:
    compute_integral(4)
except Exception:
    print(traceback.format_exc())
```

The reason is that the `quad` function is from SciPy and Numba doesn't know how to handle it.

This is the biggest problem with Numba.  A large suit of scientific tools aren't currently compatible.

On the other hand, it's a problem that people are well aware of, and various people are racing to overcome.

Second, even though it might not be possible to JIT-compile your whole program, you might well be able to compile the hot loops that are eating up 99% of your computation time.

If you can do this, you open up large speed gains, as the following sections make clear.

+++

### Vectorization vs Numba

+++

The problem is to maximize the function 

$$ f(x, y) = \frac{\cos \left(x^2 + y^2 \right)}{1 + x^2 + y^2} + 1$$

using brute force --- searching over a grid of $(x, y)$ pairs.

```{code-cell} ipython3
def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

```{code-cell} ipython3
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

gridsize = 50
gmin, gmax = -3, 3
xgrid = np.linspace(gmin, gmax, gridsize)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)

# === plot value function === #
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.4,
                linewidth=0.05)


ax.scatter(x, y, c='k', s=0.6)

ax.scatter(x, y, f(x, y), c='k', s=0.6)

ax.view_init(25, -57)
ax.set_zlim(-0, 2.0)
ax.set_xlim(gmin, gmax)
ax.set_ylim(gmin, gmax)

plt.show()
```

#### Vectorized code

```{code-cell} ipython3
grid = np.linspace(-3, 3, 10_000)
```

```{code-cell} ipython3
x, y = np.meshgrid(grid, grid)
```

```{code-cell} ipython3
---
nbpresent:
  id: 1ba9f9f9-f737-4ee1-86e6-0a33c4752188
---
tic()
np.max(f(x, y))
toc()
```

#### JITTed code

+++

A jitted version

```{code-cell} ipython3
@jit
def compute_max():
    m = -np.inf
    for x in grid:
        for y in grid:
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
            if z > m:
                m = z
    return m
```

```{code-cell} ipython3
tic()
compute_max()
toc()
```

```{code-cell} ipython3
tic()
compute_max()
toc()
```

#### JITTed, parallelized code: @vectorize

+++ {"nbpresent": {"id": "3de27362-4528-4669-8e3d-2e7db1dd0721"}}

Numba for vectorization with automatic parallization - even faster:

```{code-cell} ipython3
---
nbpresent:
  id: e443f7ad-f26e-4148-983e-83a5f6a2214e
---
signature = 'float64(float64, float64)'  # A function sending (float, float) into float

@vectorize(signature, target='parallel')
def f_par(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

```{code-cell} ipython3
x, y = np.meshgrid(grid, grid)

np.max(f_par(x, y))
```

```{code-cell} ipython3
---
nbpresent:
  id: 08ff9d10-b80e-489f-8e7e-5e1869903393
---
tic()
np.max(f_par(x, y))
toc()
```

```{code-cell} ipython3
---
nbpresent:
  id: 08ff9d10-b80e-489f-8e7e-5e1869903393
---
tic()
np.max(f_par(x, y))
toc()
```

### Summary

+++

We just scratched the surface of Numba goodness.  For further reading see:
    
* [the docs](http://numba.pydata.org/)
* [this overview](http://matthewrocklin.com/blog/work/2018/01/30/the-case-for-numba)

```{code-cell} ipython3

```
