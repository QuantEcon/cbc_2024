---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Accelerating Python Code with Numba

----

### Written for the CBC Workshop (May 2024)

#### John Stachurski


-----

+++

In the lecture on NumPy we saw that vectorization via NumPy can help accelerate our code.

But

* Vectorization can be very memory intensive.
* NumPy-style vectorization cannot fully exploit parallel hardware (precompiled NumPy binaries cannot optimize on array size or over hardware accelerators).
* Some problems cannot be vectorized --- they need to be written in loops.

This notebook contains a very quick introduction to an alternative method for accelerating code, via 
[Numba](https://numba.pydata.org/).

We use the following imports:

```{code-cell} ipython3
import numpy as np
from numba import vectorize, jit, float64   
import matplotlib.pyplot as plt
```

## Example: Solow-Swan dynamics


Here we look at one example of the last case and discuss how to accelerate it.

### A loop in Python

Let's suppose we are interested in the long-run behavior of capital stock in the stochasic
Solow-Swan model

$$
    k_{t+1} = a_{t+1} s k_t^\alpha + (1-\delta)k_t 
$$

where $(a_t)$ is IID and uniform.

Here's some code to generate a short time series, plus a plot.

```{code-cell} ipython3
α, s, δ, a = 0.4, 0.3, 0.1, 1.0
n = 120
k =  np.empty(n)
k[0] = 0.2
for t in range(n-1):
    k[t+1] = np.random.rand() * s * k[t]**α + (1 - δ) * k[t]
    
fig, ax = plt.subplots()
ax.plot(k, 'o-', ms=2, lw=1)
ax.set_xlabel('time')
ax.set_ylabel('capital')
plt.show()
```

Let's say that we want to compute long-run average capital stock.

```{code-cell} ipython3
def solow(n=10_000_000, α=0.4, s=0.3, δ=0.1, k0=0.2):
    k = k0
    k_sum = k0
    for t in range(n):
        a = np.random.rand()
        k = a * s * k**α + (1 - δ) * k
        k_sum += k
    return k_sum / n

%time k = solow()
print(f"Steady-state capital = {k}.")
```

Steady-state capital:

+++

Notice that the run-time is pretty slow.

Also, we can't use NumPy to accelerate it because there's no way to vectorize the loop.

Let's look at some alternatives.

+++

### A Fortran version

We can make it fast if we rewrite it in Fortran.

To execute the following Fortran code on your machine, you need 

* a Fortran compiler (such as the open source `gfortran` compiler) and also
* the `fortranmagic` Jupyter extension, which can be installed by uncommenting

```{code-cell} ipython3
#!pip install fortran-magic
```

Now we load the extension (skip executing this and the rest of the section if you don't have a Fortran compiler).

```{code-cell} ipython3
%load_ext fortranmagic
```

In the following code, all parameters are the same as for the Python code above.

+++

Now we add the cell magic ``%%fortran`` to a cell that contains a Fortran subroutine for the Solow-Swan computation:

```{code-cell} ipython3
%%fortran

subroutine solow_fortran(k0, s, delta, alpha, n, kt)
 implicit none
 integer, parameter :: dp=kind(0.d0) 
 integer, intent(in) :: n
 real(dp), intent(in) :: k0, s, delta, alpha
 real(dp), intent(out) :: kt
 real(dp) :: k, k_sum, a
 integer :: i
 k = k0
 k_sum = k0
 call random_seed
 do i = 1, n - 1      
  call random_number(a)
  k = a * s * k**alpha + (1 - delta) * k
  k_sum = k_sum + k
 end do
 kt = k_sum / real(n)
end subroutine solow_fortran 
```

Now we can call the function `solow_fortran` from Python.

(`fortranmagic` uses a program called `F2Py` to create a Python "wrapper" for the Fortran subroutine so we can access it from within Python.)

Let's make sure it gives a reasonable answer.

```{code-cell} ipython3
n = 10_000_000
solow_fortran(0.2, s, δ, α, n)
```

Now let's time it:

```{code-cell} ipython3
%time solow_fortran(0.2, s, δ, α, n)
```

Let's time it more carefully, over multiple runs:

```{code-cell} ipython3
%timeit solow_fortran(0.2, s, δ, α, n)
```

The speed gain is about 1 order of magnitude.

+++

## Numba

Now let's try the same thing in Python using Numba's JIT compilation.

We recall the Python function from above.

```{code-cell} ipython3
def solow(n=10_000_000, α=0.4, s=0.3, δ=0.1, k0=0.2):
    k = k_sum = k0
    for t in range(n-1):
        a = np.random.rand()
        k = a * s * k**α + (1 - δ) * k
        k_sum += k
    return k_sum / n
```

Now let's flag it for JIT-compilation:

```{code-cell} ipython3
solow_jitted = jit(solow)
```

And then run it:

```{code-cell} ipython3
%time k = solow_jitted()
```

```{code-cell} ipython3
%time k = solow_jitted()
```

```{code-cell} ipython3
%timeit k = solow_jitted()
```

Hopefully we get a similar value (about 1.95):

```{code-cell} ipython3
k
```

Here's the same thing using decorator notation.

```{code-cell} ipython3
@jit
def solow(n=10_000_000, α=0.4, s=0.3, δ=0.1, k0=0.2):
    k = k_sum = k0
    for t in range(n-1):
        a = np.random.rand()
        k = a * s * k**α + (1 - δ) * k
        k_sum += k
    return k_sum / n
```

```{code-cell} ipython3
%time k = solow()
```

```{code-cell} ipython3
%time k = solow()
```

After JIT compilation, function execution speed is about the same as Fortran.

```{code-cell} ipython3
k
```

#### How does it work?

+++

The secret sauce is type inference inside the function body.

When we call `solow_jitted` with particular arguments, Numba's compiler works
through the function body and infers the types of the variables inside the
function.

It then produces compiled code *specialized to that type signature*

For example, we called `solow_jitted` with a `float, int` pair in the cell above and the compiler produced code specialized to those types.

That code runs fast because the compiler can fully specialize all operations inside the function based on that information and hence write very efficient machine code.

+++

#### Limitations of Numba

Numba is great when it works but it can't compile functions that aren't
themselves JIT compiled.

+++

In practice, this means we can't use most third party libraries:

```{code-cell} ipython3
from scipy.integrate import quad

def compute_integral(n):
    return quad(lambda x: x**(1/n), 0, 1)
```

This works fine if we don't jit the function.

```{code-cell} ipython3
compute_integral(4)
```

But if we do...

```{code-cell} ipython3
@jit
def compute_integral(n):
    return quad(lambda x: x**(1/n), 0, 1)
        
```

```{code-cell} ipython3
compute_integral(4)
```

The reason is that the `quad` function is from SciPy and Numba doesn't know how
to handle it.

Key message: even though it might not be possible to JIT-compile your whole
program, you might well be able to compile the hot loops that are eating up 99%
of your computation time.

If you can do this, you open up large speed gains, as the following sections
make clear.

+++

### Vectorization vs Numba

+++

We made the point above that some problems are hard or impossible to vectorize and, in these situations, that we can use Numba instead of NumPy to accelerate our code.

However, there are also many situations where we *can* vectorize our code but Numba is still the better option.

Let's look at an example.

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
n = 10_000
grid = np.linspace(-3, 3, n)
```

```{code-cell} ipython3
x, y = np.meshgrid(grid, grid)
```

```{code-cell} ipython3
---
nbpresent:
  id: 1ba9f9f9-f737-4ee1-86e6-0a33c4752188
---
%%time
np.max(f(x, y))
```

#### JITTed code

+++

Let's try a jitted version with loops.

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
%%time
compute_max()
```

```{code-cell} ipython3
%%time
compute_max()
```

Why the speed gain?

+++

#### JITTed, parallelized code

We can parallelize on the CPU through Numba via `@jit(parallel=True)`

Our strategy is

- Compute the max value along each row, parallelizing this task across rows
- Take the max of these row maxes

The memory footprint is still relatively light, because the size of the rows is only `n x 1`.

```{code-cell} ipython3
@jit
def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

```{code-cell} ipython3
from numba import prange
```

```{code-cell} ipython3
@jit(parallel=True)
def compute_max():
    row_maxes = np.empty(n)
    y_grid = grid
    for i in prange(n):
        x = grid[i]
        row_maxes[i] = np.max(f(x, y_grid))
    return np.max(row_maxes)
```

```{code-cell} ipython3
%%time
compute_max()
```

```{code-cell} ipython3
%%time
compute_max()
```

```{code-cell} ipython3

```
