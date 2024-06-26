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

# An Introduction to JAX

----

#### John Stachurski

#### Prepared for the CBC Workshop (May 2024)

----


This lecture provides a short introduction to [Google JAX](https://github.com/google/jax).

What GPUs do we have access to?

```{code-cell} ipython3
!nvidia-smi
```

## JAX as a NumPy Replacement


One way to use JAX is as a plug-in NumPy replacement. Let's look at the
similarities and differences.

### Similarities


The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

Now we can use `jnp` in place of `np` for the usual array operations:

```{code-cell} ipython3
a = jnp.array((1.0, 3.2, -1.5))
```

```{code-cell} ipython3
print(a)
```

```{code-cell} ipython3
print(jnp.sum(a))
```

```{code-cell} ipython3
print(jnp.mean(a))
```

```{code-cell} ipython3
print(jnp.dot(a, a))
```

```{code-cell} ipython3
print(a @ a)  # Equivalent
```

However, the array object `a` is not a NumPy array:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
type(a)
```

Even scalar-valued maps on arrays return JAX arrays.

```{code-cell} ipython3
jnp.sum(a)
```

Operations on higher dimensional arrays are also similar to NumPy:

```{code-cell} ipython3
A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```{code-cell} ipython3
jnp.linalg.inv(B)   # Inverse of identity is identity
```

### Differences

Let's now look at the differences between JAX and NumPy

#### 32 bit floats

One difference between NumPy and JAX is that JAX currently uses 32 bit floats by default.

```{code-cell} ipython3
jnp.ones(3)
```

+++ {"jp-MarkdownHeadingCollapsed": true}

This is standard for GPU computing and can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.  In these cases 64 bit floats can be enforced via the command

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

Let's check this works:

```{code-cell} ipython3
jnp.ones(3)
```

#### Mutability

+++

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.  

For example, with NumPy we can write

```{code-cell} ipython3
import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell} ipython3
a[0] = 42
a
```

In JAX this fails:

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
a
```

```{code-cell} ipython3
:tags: [raises-exception]

a[0] = 42
```

The designers of JAX chose to make arrays immutable because JAX uses a
functional programming style.  More on this below.  

+++

## Random Numbers

Random numbers are also a bit different in JAX.


### Controlling the state

Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

(This is also related to JAX's functional programming paradigm, discussed below.)

+++

First we produce a key, which seeds the random number generator.

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
```

```{code-cell} ipython3
type(key)
```

```{code-cell} ipython3
key
```

Now we can use the key to generate some random numbers:

```{code-cell} ipython3
x = jax.random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell} ipython3
jax.random.normal(key, (3, 3))
```

### Generating fresh draws

+++

To produce a (quasi-) independent draw, we can use `split`

```{code-cell} ipython3
new_keys = jax.random.split(key, 5)   # Generate 5 new keys
```

```{code-cell} ipython3
len(new_keys)
```

```{code-cell} ipython3
shape = (3, )
for i, key in enumerate(new_keys):
    print(f'Draw from key {i} = {jax.random.normal(key, shape)}')
```

Another function we can use to update the key is `fold_in`.

```{code-cell} ipython3
seed = 1234  
new_key = jax.random.fold_in(key, seed)

jax.random.normal(new_key, (3, 1))
```

This is often used in loops -- here's an example that produces `k` (quasi-) independent random `n x n` matrices using this procedure and prints their determinants.

```{code-cell} ipython3
def gen_random_matrices(seed=1234, n=10, k=5):
    key = jax.random.PRNGKey(seed)
    for i in range(k):
        key = jax.random.fold_in(key, i)
        d = jnp.linalg.det(jax.random.uniform(key, (n, n)))
        print(f"Determinant = {d:.4}")

gen_random_matrices()
```

## JIT compilation

The JAX just-in-time (JIT) compiler generates efficient, parallelized machine code at runtime.

Code is optimized for either the CPU or the GPU/TPU.

### A first example

To see the JIT compiler in action, consider the following function.

```{code-cell} ipython3
def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2)
    return jnp.sum(a)
```

Let's build an array to call the function on.

```{code-cell} ipython3
n = 50_000_000
x = jnp.ones(n)
```

How long does the function take to execute?

```{code-cell} ipython3
%time f(x).block_until_ready()
```

In order to measure actual speed, we use `block_until_ready()` method 
to hold the interpreter until the array is returned from
the device. 

This is necessary because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of GPU computations.

+++

The code doesn't run as fast as we might hope, given that it's running on a GPU.

But if we run it a second time it becomes much faster:

```{code-cell} ipython3
%time f(x).block_until_ready()
```

This is because the built in functions like `jnp.cos` are JIT compiled and the
first run includes compile time.

+++

### When does JAX recompile?

You might remember that Numba recompiles if we change the *types* of variables in a function call.

JAX recompiles more often --- in particular, it recompiles every time we change types or array sizes.

For example, let's try

```{code-cell} ipython3
m = n + 1
y = jnp.ones(m)
```

```{code-cell} ipython3
%time f(y).block_until_ready()
```

Notice that the execution time increases, because now new versions of 
the built-ins like `jnp.cos` are being compiled, specialized to the new array
size.

If we run again, the code is dispatched to the correct compiled version and we
get faster execution.

```{code-cell} ipython3
%time f(y).block_until_ready()
```

Why does JAX generate fresh machine code every time we change the array size???

+++

The compiled versions for the previous array size are still available in memory
too, and the following call is dispatched to the correct compiled code.

```{code-cell} ipython3
%time f(x).block_until_ready()
```

### Compiling user-built functions

We can instruct JAX to compile entire functions that we build.

For example, consider

```{code-cell} ipython3
def g(x):
    y = jnp.zeros_like(x)
    for i in range(10):
        y += x**i
    return y
```

```{code-cell} ipython3
n = 1_000_000
x = jnp.ones(n)
```

Let's time it.

```{code-cell} ipython3
%time g(x).block_until_ready()
```

```{code-cell} ipython3
%time g(x).block_until_ready()
```

Now let's compile the whole function.

```{code-cell} ipython3
g_jit = jax.jit(g)   # target for JIT compilation
```

Let's run once to compile it:

```{code-cell} ipython3
g_jit(x)
```

And now let's time it.

```{code-cell} ipython3
%time g_jit(x).block_until_ready()
```

Note the speed gain.

This is because 

1. the loop is compiled and
2. the array operations are fused and no intermediate arrays are created.


Incidentally, a more common syntax when targetting a function for the JIT
compiler is

```{code-cell} ipython3
@jax.jit
def g_jit_2(x):
    y = jnp.zeros_like(x)
    for i in range(10):
        y += x**i
    return y
```

```{code-cell} ipython3
%time g_jit_2(x).block_until_ready()
```

```{code-cell} ipython3
%time g_jit_2(x).block_until_ready()
```

#### Static arguments

+++

Because the compiler specializes on array sizes, it needs to recompile code when array sizes change.

As a result, any argument that determines sizes of arrays should be flagged by `static_argnums` -- a signal that JAX can treat the argument as fixed and compile efficient code that specializes on this value.

(JAX will recompile the function when it changes).

Here's a example.

```{code-cell} ipython3
def f(n, seed=1234):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (n, ))
    return x.std()
```

```{code-cell} ipython3
f(5)
```

```{code-cell} ipython3
f_jitted = jax.jit(f)
```

```{code-cell} ipython3
f_jitted(5)
```

Let's fix this:

```{code-cell} ipython3
f_jitted = jax.jit(f, static_argnums=(0, ))   # First argument is static
f_jitted(5)
```

## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has “una anima di pura programmazione funzionale”.*


In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.
    
A pure function will always return the same result if invoked with the same inputs.

In particular, a pure function has

* no dependence on global variables and
* no side effects

+++

### Examples: Python/NumPy/Numba style code is not generally pure

+++

#### Example 1

+++

Here's an example to show that NumPy functions are not pure:

```{code-cell} ipython3
np.random.randn(3)
```

```{code-cell} ipython3
np.random.randn(3)
```

This function returns the different results when called on the same inputs!

The issue is that the function maintains state between function calls --- the state of the random number generator.

```{code-cell} ipython3
np.random.get_state()[2]
```

```{code-cell} ipython3
np.random.randn(3)
```

```{code-cell} ipython3
np.random.get_state()[2]
```

#### Example 2

Here's a function that's also not pure because it depends on a global value.

```{code-cell} ipython3
a = 10
def f(x): 
    return a * x

f(1)
```

```{code-cell} ipython3
a = 20
f(1)
```

As with Example 1, the output of the function cannot be fully predicted from the inputs.

+++

#### Example 3

+++

Here's a function that fails to be pure because it modifies external state.

```{code-cell} ipython3
def change_input(x):   # Not pure -- side effects
    x[:] = 42
```

Let's test this with

```{code-cell} ipython3
x = np.ones(5)
x
```

```{code-cell} ipython3
change_input(x)
```

```{code-cell} ipython3
x
```

### Compiling impure functions

JAX does *not* insist on pure functions, even when we use the JIT compiler.

For example, JAX will not usually throw errors when compiling impure functions 

However, execution becomes unpredictable!

Here's an illustration of this fact, using global variables:

```{code-cell} ipython3
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell} ipython3
x = jnp.ones(2)
```

```{code-cell} ipython3
x
```

```{code-cell} ipython3
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected --- as long as the same compiled version is called.

```{code-cell} ipython3
a = 42
```

```{code-cell} ipython3
f(x)
```

Notice that the change in the value of `a` takes effect in the code below:

```{code-cell} ipython3
x = jnp.ones(3)
```

```{code-cell} ipython3
f(x)
```

Can you explain why?

+++

#### Moral

+++

Moral of the story: write pure functions when using JAX!

+++

## Gradients

JAX can use automatic differentiation to compute gradients.

This can be extremely useful for optimization and solving nonlinear systems.

We will see significant applications later in this lecture series.

For now, here's a very simple illustration involving the function

```{code-cell} ipython3
def f(x):
    return (x**2) / 2
```

Let's take the derivative:

```{code-cell} ipython3
f_prime = jax.grad(f)
```

```{code-cell} ipython3
f_prime(10.0)
```

Let's plot the function and derivative, noting that $f'(x) = x$.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

## Writing vectorized code

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

In some ways, vectorization is the same in JAX as it is in NumPy.

But there are also differences, which we highlight here.

As a running example, consider the function

```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)
```


Suppose that we want to evaluate this function on a square grid of $x$ and $y$ points.


### A slow version with loops

To clarify, here is the slow `for` loop version, which we run in a setting where `len(x) = len(y)` is very small.

```{code-cell} ipython3
n = 80
x = jnp.linspace(-2, 2, n)
y = x
z_loops = np.empty((n, n))
```

```{code-cell} ipython3
%%time
for i in range(n):
    for j in range(n):
        z_loops[i, j] = f(x[i], y[j])
```

Even for this very small grid, the run time is slow.

(Notice that we used a NumPy array for `z_loops` because we wanted to write to it.)

+++

OK, so how can we do the same operation in vectorized form?

If you are new to vectorization, you might guess that we can simply write

```{code-cell} ipython3
z_bad = f(x, y)
```

But this gives us the wrong result because JAX doesn't understand the nested for loop.

```{code-cell} ipython3
z_bad.shape
```

Here is what we actually wanted:

```{code-cell} ipython3
z_loops.shape
```

### Vectorization attempt 1

+++

To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation that originated in MATLAB and was replicated in NumPy and then JAX:

```{code-cell} ipython3
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is fast.

```{code-cell} ipython3
%time z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

```{code-cell} ipython3
%time z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's confirm that we got the right answer.

```{code-cell} ipython3
jnp.allclose(z_mesh, z_loops)
```

### Larger grids

+++

Now we can set up a serious grid and run the same calculation (on the larger grid) in a short amount of time.

```{code-cell} ipython3
n = 6000
x = jnp.linspace(-2, 2, n)
y = x
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

```{code-cell} ipython3
x_mesh.shape
```

```{code-cell} ipython3
y_mesh.shape
```

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh) 
```

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh) 
```

But there is one problem here: the inputs use a lot of memory.

```{code-cell} ipython3
(x_mesh.nbytes + y_mesh.nbytes) / 1_000_000  # MB of memory
```

By comparison, the flat array `x` is just

```{code-cell} ipython3
x.nbytes / 1_000_000   # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

```{code-cell} ipython3
del x_mesh  # Free memory
del y_mesh  # Free memory
```

### Vectorization attempt 2

+++

We can achieve a similar effect through NumPy style broadcasting rules.

```{code-cell} ipython3
x_reshaped = jnp.reshape(x, (n, 1))   # Give x another dimension (column)
y_reshaped = jnp.reshape(y, (1, n))   # Give y another dimension (row)
```

```{code-cell} ipython3
x_reshaped.shape
```

```{code-cell} ipython3
y_reshaped.shape
```

When we evaluate $f$ on these reshaped arrays, we replicate the nested for loops in the original version.

```{code-cell} ipython3
%time z_reshaped = f(x_reshaped, y_reshaped).block_until_ready()
```

```{code-cell} ipython3
%time z_reshaped = f(x_reshaped, y_reshaped).block_until_ready()
```

Let's check that we got the same result

```{code-cell} ipython3
jnp.allclose(z_reshaped, z_mesh)
```

The memory usage for the inputs is much more moderate.

```{code-cell} ipython3
(x_reshaped.nbytes + y_reshaped.nbytes) / 1_000_000
```

### Vectorization attempt 3


There's another approach to vectorization we can pursue, using [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)

+++

It runs out that, when we are working with complex functions and operations, this `vmap` approach can be the easiest to implement.

+++

Here's our function again:

```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)
```

Here are the steps

```{code-cell} ipython3
f = jax.vmap(f, in_axes=(None, 0))    # vectorize in y
f = jax.vmap(f, in_axes=(0, None))    # vectorize in x
```

With this construction, we can now call the function $f$ on flat (low memory) arrays.

```{code-cell} ipython3
%%time
z_vmap = f(x, y).block_until_ready()
```

```{code-cell} ipython3
%%time
z_vmap = f(x, y).block_until_ready()
```

Let's check we produce the correct answer:

```{code-cell} ipython3
jnp.allclose(z_vmap, z_mesh)
```

Let's finish by cleaning up.

```{code-cell} ipython3
del z_mesh
del z_vmap
del z_reshaped
```

### Exercises

+++

#### Exercise 1

+++

Compute an approximation to $\pi$ by simulation:

1. draw $n$ observations of a bivariate uniform on the unit square $[0, 1] \times [0, 1]$
2. count the fraction that fall in the unit circle (radius 0.5) centered on (0.5, 0.5)
3. multiply the result by 4

Use JAX

```{code-cell} ipython3
for i in range(18):
    print("Solution below 🐠")
```

```{code-cell} ipython3
@jax.jit
def approx_pi(n=1_000_000, seed=1234):
    key = jax.random.PRNGKey(seed)
    u = jax.random.uniform(key, (2, n))
    distances = jnp.sqrt((u[0, :] - 0.5)**2 + (u[1, :] - 0.5)**2)
    fraction_in_circle = jnp.mean(distances < 0.5)
    return fraction_in_circle * 4  # dividing by radius**2
```

```{code-cell} ipython3
%time approx_pi()
```

```{code-cell} ipython3
%time approx_pi()
```

#### Exercise 2

The following code uses Monte Carlo to price a European call option under certain assumptions on stock price dynamics.

The code is accelerated by Numba.

```{code-cell} ipython3
import numba
from numpy.random import randn
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@numba.jit(parallel=True)
def compute_call_price_parallel(β=β,
                                μ=μ,
                                S0=S0,
                                h0=h0,
                                K=K,
                                n=n,
                                ρ=ρ,
                                ν=ν,
                                M=M):
    current_sum = 0.0
    # For each sample path
    for m in numba.prange(M):
        s = np.log(S0)
        h = h0
        # Simulate forward in time
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # And add the value max{S_n - K, 0} to current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)
        
    return β**n * current_sum / M
```

Let's run it once to compile it:

```{code-cell} ipython3
compute_call_price_parallel()
```

And now let's time it:

```{code-cell} ipython3
%%time 
compute_call_price_parallel()
```

Try writing a version of this operation for JAX, using all the same
parameters.

If you are running your code on a GPU, you should be able to achieve faster execution.

```{code-cell} ipython3
for i in range(18):
    print("Solution below 🐠")
```

**Solution**

Here is one solution:

```{code-cell} ipython3
@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           seed=1234):

    key = jax.random.PRNGKey(seed)
    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        Z = jax.random.normal(key, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
        key = jax.random.fold_in(key, t)

    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))
        
    return β**n * expectation
```

Let's run it once to compile it:

```{code-cell} ipython3
compute_call_price_jax()
```

And now let's time it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{code-cell} ipython3

```
