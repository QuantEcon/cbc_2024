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

# Inventory Dynamics

+++

----

#### John Stachurski
#### Prepared for the CBC Computational Workshop (May 2024)

----

+++

## Overview

This lecture explores s-S inventory dynamics.

We also studied this model in an earlier notebook using NumPy.

Here we study the same problem using JAX.

One issue we will consider is whether or not we can improve execution speed by using JAX's `fori_loop` function (as a replacement for looping in Python).

We will use the following imports:

```{code-cell} ipython3
:hide-output: false

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from collections import namedtuple
```

Here’s a description of our GPU:

```{code-cell} ipython3
:hide-output: false

!nvidia-smi
```

## Model

We briefly recall the dynamics.

Let $a \vee b = \max\{a, b\}$

Consider a firm with inventory $ X_t $.

The firm waits until $ X_t \leq s $ and then restocks up to $ S $ units.

It faces stochastic demand $ \{ D_t \} $, which we assume is IID across time and
firms.

$$
X_{t+1} =
    \begin{cases}
      (S - D_{t+1}) \vee  0 & \quad \text{if } X_t \leq s \\
      (X_t - D_{t+1}) \vee 0 &  \quad \text{if } X_t > s
    \end{cases}
$$

In what follows, we will assume that each $ D_t $ is lognormal, so that

$$
D_t = \exp(\mu + \sigma Z_t)
$$

where $ \mu $ and $ \sigma $ are parameters and $ \{Z_t\} $ is IID
and standard normal.

Here’s a `namedtuple` that stores parameters.

```{code-cell} ipython3
:hide-output: false

Parameters = namedtuple('Parameters', ('s', 'S', 'μ', 'σ'))

# Create a default instance
params = Parameters(s=10, S=100, μ=1.0, σ=0.5)
```

## Cross-sectional distributions

Let's simulate the inventories of a cross-section of firms.

We will use the following code to update the cross-section by one period.

```{code-cell} ipython3
:hide-output: false

@jax.jit
def update_cross_section(params, X_vec, D):
    """
    Update by one period a cross-section of firms with inventory levels 
    X_vec, given the vector of demand shocks in D.

       * X_vec[i] is the inventory of firm i
       * D[i] is the demand shock for firm i 

    """
    # Unpack
    s, S = params.s, params.S
    # Restock if the inventory is below the threshold
    X_new = jnp.where(X_vec <= s, 
                      jnp.maximum(S - D, 0), jnp.maximum(X_vec - D, 0))
    return X_new
```

### For loop version

Now we provide code to compute the cross-sectional distribution $ \psi_T $ given some
initial distribution $ \psi_0 $ and a positive integer $ T $.

In this code we use an ordinary Python `for` loop to step forward through time

While Python loops are slow, this approach is reasonable here because
efficiency of outer loops has far less influence on runtime than efficiency of inner loops.

(Below we will squeeze out more speed by compiling the outer loop as well as the
update rule.)

In the code below, the initial distribution $ \psi_0 $ takes all firms to have
initial inventory `x_init`.

```{code-cell} ipython3
:hide-output: false

def compute_cross_section(params, x_init, T, key, num_firms=50_000):
    # Unpack
    μ, σ = params.μ, params.σ
    # Set up initial distribution
    X = jnp.full((num_firms, ), x_init)
    # Loop
    for i in range(T):
        Z = random.normal(key, shape=(num_firms, ))
        D = jnp.exp(params.μ + params.σ * Z)
        X = update_cross_section(params, X, D)
        key = random.fold_in(key, i)

    return X
```

We’ll use the following specification

```{code-cell} ipython3
:hide-output: false

x_init = 50
T = 500
# Initialize random number generator
key = random.PRNGKey(10)
```

Let’s look at the timing.

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section(params, x_init, T, key).block_until_ready()
```

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section(params, x_init, T, key).block_until_ready()
```

Here’s a histogram of inventory levels at time $ T $.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.hist(X_vec, bins=50, 
        density=True, 
        histtype='step', 
        label=f'cross-section when $T = {T}$')
ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

### Compiling the outer loop

Now let’s see if we can gain some speed by compiling the outer loop, which steps
through the time dimension.

We will do this using `jax.jit` and a `fori_loop`, which is a compiler-ready version of a `for` loop provided by JAX.

```{code-cell} ipython3
:hide-output: false

def compute_cross_section_fori(params, x_init, T, key, num_firms=50_000):

    s, S, μ, σ = params.s, params.S, params.μ, params.σ

    # Define the function for each update
    def fori_update(t, state):
        # Unpack
        X, key = state
        # Draw shocks using key
        Z = random.normal(key, shape=(num_firms,))
        D = jnp.exp(μ + σ * Z)
        # Update X
        X = jnp.where(X <= s,
                  jnp.maximum(S - D, 0),
                  jnp.maximum(X - D, 0))
        # Refresh the key
        key = random.fold_in(key, t)
        new_state = X, key
        return new_state

    # Loop t from 0 to T, applying fori_update each time.
    X = jnp.full((num_firms, ), x_init)
    initial_state = X, key
    X, key = lax.fori_loop(0, T, fori_update, initial_state)

    return X

# Compile taking T and num_firms as static (changes trigger recompile)
compute_cross_section_fori = jax.jit(
    compute_cross_section_fori, static_argnums=(2, 4))
```

Let’s see how fast this runs with compile time.

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section_fori(params, x_init, T, key).block_until_ready()
```

And let’s see how fast it runs without compile time.

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section_fori(params, x_init, T, key).block_until_ready()
```

Compared to the original version with a pure Python outer loop, we have
produced a nontrivial speed gain.

This is due to the fact that we have compiled the whole operation.

Let's check that we get a similar cross-section.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.hist(X_vec, bins=50, 
        density=True, 
        histtype='step', 
        label=f'cross-section when $T = {T}$')
ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

### Further vectorization

For relatively small problems, we can make this code run even faster by generating
all random variables at once.

This improves efficiency because we are taking more operations out of the loop.

```{code-cell} ipython3
:hide-output: false

def compute_cross_section_fori(params, x_init, T, key, num_firms=50_000):

    s, S, μ, σ = params.s, params.S, params.μ, params.σ
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(T, num_firms))
    D = jnp.exp(μ + σ * Z)

    def update_cross_section(i, X):
        X = jnp.where(X <= s,
                  jnp.maximum(S - D[i, :], 0),
                  jnp.maximum(X - D[i, :], 0))
        return X

    X = lax.fori_loop(0, T, update_cross_section, X)

    return X

# Compile taking T and num_firms as static (changes trigger recompile)
compute_cross_section_fori = jax.jit(
    compute_cross_section_fori, static_argnums=(2, 4))
```

Let’s test it with compile time included.

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section_fori(params, x_init, T, key).block_until_ready()
```

Let’s run again to eliminate compile time.

```{code-cell} ipython3
:hide-output: false

%time X_vec = compute_cross_section_fori(params, x_init, T, key).block_until_ready()
```

On one hand, this version is faster than the previous one, where random variables were
generated inside the loop.

On the other hand, this implementation consumes far more memory, as we need to
store large arrays of random draws.

The high memory consumption becomes problematic for large problems.

+++

## Restock frequency

As an exercise, let’s study the probability that firms need to restock over a given time period.

In the exercise, we will

- set the starting stock level to $ X_0 = 70 $ and  
- calculate the proportion of firms that need to order twice or more in the first 50 periods.  


This proportion approximates the probability of the event when the sample size
is large.

+++

### For loop version

We start with an easier `for` loop implementation

```{code-cell} ipython3
:hide-output: false

# Define a jitted function for each update
@jax.jit
def update_stock(params, counter, X, D):
    s, S = params.s, params.S
    X = jnp.where(X <= s,
                  jnp.maximum(S - D, 0),
                  jnp.maximum(X - D, 0))
    counter = jnp.where(X <= s, counter + 1, counter)
    return counter, X

def compute_freq(params, key,
                 x_init=70,
                 sim_length=50,
                 num_firms=1_000_000):

    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)
    counter = jnp.zeros((num_firms, ))

    # Use a for loop to perform the calculations on all states
    for i in range(sim_length):
        Z = random.normal(key, (num_firms, ))
        D = jnp.exp(params.μ + params.σ * Z)
        counter, X = update_stock(params, counter, X, D)
        key = random.fold_in(key, i)

    return jnp.mean(counter > 1, axis=0)
```

```{code-cell} ipython3
:hide-output: false

key = random.PRNGKey(42)
```

```{code-cell} ipython3
:hide-output: false

%time freq = compute_freq(params, key).block_until_ready()
print(f"Frequency of at least two stock outs = {freq}")
```

```{code-cell} ipython3
:hide-output: false

%time freq = compute_freq(params, key).block_until_ready()
print(f"Frequency of at least two stock outs = {freq}")
```

### Exercise 4.1

Write a `fori_loop` version of the last function.  See if you can increase the
speed while generating a similar answer.

+++

### Solution to[ Exercise 4.1](https://jax.quantecon.org/#inventory_dynamics_ex1)

Here is a `fori_loop` version that JIT compiles the whole function

```{code-cell} ipython3
:hide-output: false

def compute_freq_fori_loop(params, 
                 key,
                 x_init=70,
                 sim_length=50,
                 num_firms=1_000_000):

    s, S, μ, σ = params
    Z = random.normal(key, shape=(sim_length, num_firms))
    D = jnp.exp(μ + σ * Z)

    # Define the function for each update
    def update(t, state):
        # Separate the inventory and restock counter
        X, counter = state
        X = jnp.where(X <= s,
                      jnp.maximum(S - D[t, :], 0),
                      jnp.maximum(X - D[t, :], 0))
        counter = jnp.where(X <= s, counter + 1, counter)
        return X, counter

    X = jnp.full((num_firms, ), x_init)
    counter = jnp.zeros(num_firms)
    initial_state = X, counter
    X, counter = lax.fori_loop(0, sim_length, update, initial_state)

    return jnp.mean(counter > 1)

compute_freq_fori_loop = jax.jit(compute_freq_fori_loop, static_argnums=((3, 4)))
```

Note the time the routine takes to run, as well as the output

```{code-cell} ipython3
:hide-output: false

%time freq = compute_freq_fori_loop(params, key).block_until_ready()
```

```{code-cell} ipython3
:hide-output: false

%time freq = compute_freq_fori_loop(params, key).block_until_ready()
```

```{code-cell} ipython3
:hide-output: false

print(f"Frequency of at least two stock outs = {freq}")
```

```{code-cell} ipython3

```
