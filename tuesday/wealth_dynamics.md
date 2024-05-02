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

# Wealth Dynamics

------

#### John Stachurski

#### Prepared for the CBC Workshop (May 2024)

In this lecture we examine wealth dynamics in large cross-section of agents who
are subject to both 

* idiosyncratic shocks, which affect labor income and returns, and 
* an aggregate shock, which also impacts on labor income and returns


In most macroeconomic models savings and consumption are determined by optimization.

Here savings and consumption behavior is taken as given -- you can plug in your
favorite model to obtain savings behavior and then analyze distribution dynamics
using the techniques described below.

On of our interests will be how different aspects of wealth dynamics -- such
as labor income and the rate of return on investments -- feed into measures of
inequality, such as the Gini coefficent.

Please uncomment the following if necessary

```{code-cell} ipython3
#!pip install quantecon 
```

We use the following imports.

```{code-cell} ipython3
import numba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from time import time
```

## Wealth dynamics


Wealth evolves as follows:

```{math}
    w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

Here

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is labor income and
- $s(w_t)$ is savings (current wealth minus current consumption)


There is an aggregate state process 

$$
    z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1}
$$

that affects the interest rate and labor income.

In particular, the gross interest rates obeys

$$
    R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t)
$$

while

$$
    y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t)
$$

The tuple $\{ (\epsilon_t, \xi_t, \zeta_t) \}$ is IID and standard normal in $\mathbb R^3$.

(Each household receives their own idiosyncratic shocks.)

Regarding the savings function $s$, our default model will be

```{math}
:label: sav_ah

s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\}
```

where $s_0$ is a positive constant.

Thus, 

* for $w < \hat w$, the household saves nothing, while
* for $w \geq \bar w$, the household saves a fraction $s_0$ of their wealth.



## Implementation (Numba version)


Here's a function that collects parameters and useful constants

```{code-cell} ipython3
def create_wealth_model(w_hat=1.0,   # Savings parameter
                        s_0=0.75,    # Savings parameter
                        c_y=1.0,     # Labor income parameter
                        μ_y=1.0,     # Labor income parameter
                        σ_y=0.2,     # Labor income parameter
                        c_r=0.05,    # Rate of return parameter
                        μ_r=0.1,     # Rate of return parameter
                        σ_r=0.5,     # Rate of return parameter
                        a=0.5,       # Aggregate shock parameter
                        b=0.0,       # Aggregate shock parameter
                        σ_z=0.1):    # Aggregate shock parameter
    """
    Create a wealth model with given parameters. 

    Return a tuple model = (household_params, aggregate_params), where
    household_params collects household information and aggregate_params
    collects information relevant to the aggregate shock process.
    
    """
    # Mean and variance of z process
    z_mean = b / (1 - a)
    z_var = σ_z**2 / (1 - a**2)
    exp_z_mean = np.exp(z_mean + z_var / 2)
    # Mean of R and y processes
    R_mean = c_r * exp_z_mean + np.exp(μ_r + σ_r**2 / 2)
    y_mean = c_y * exp_z_mean + np.exp(μ_y + σ_y**2 / 2)
    # Test stability condition ensuring wealth does not diverge
    # to infinity.
    α = R_mean * s_0
    if α >= 1:
        raise ValueError("Stability condition failed.")
    # Pack values into tuples and return them
    household_params = (w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean)
    aggregate_params = (a, b, σ_z, z_mean, z_var)
    model = household_params, aggregate_params
    return model
```

Here's a function that generates the aggregate state process

```{code-cell} ipython3
@numba.jit
def generate_aggregate_state_sequence(aggregate_params, length=100):
    a, b, σ_z, z_mean, z_var = aggregate_params 
    z = np.empty(length+1)
    z[0] = z_mean   # Initialize at z_mean
    for t in range(length):
        z[t+1] = a * z[t] + b + σ_z * np.random.randn()
    return z
```

Here's a function that updates household wealth by one period, taking the
current value of the aggregate shock.

```{code-cell} ipython3
@numba.jit
def update_wealth(household_params, w, z):
    """
    Generate w_{t+1} given w_t and z_{t+1}.
    """
    # Unpack
    w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean = household_params
    # Update wealth
    y = c_y * np.exp(z) + np.exp(μ_y + σ_y * np.random.randn())
    wp = y
    if w >= w_hat:
        R = c_r * np.exp(z) + np.exp(μ_r + σ_r * np.random.randn())
        wp += R * s_0 * w
    return wp
```

Here's function to simulate the time series of wealth for an individual household.

```{code-cell} ipython3
@numba.jit
def wealth_time_series(model, w_0, sim_length):
    """
    Generate a single time series of length sim_length for wealth given initial
    value w_0.  The function generates its own aggregate shock sequence.

    """
    # Unpack
    household_params, aggregate_params = model
    a, b, σ_z, z_mean, z_var = aggregate_params 
    # Initialize and update
    z = generate_aggregate_state_sequence(aggregate_params, 
                                          length=sim_length)
    w = np.empty(sim_length)
    w[0] = w_0
    for t in range(sim_length-1):
        w[t+1] = update_wealth(household_params, w[t], z[t+1])
    return w
```

Let's look at the wealth dynamics of an individual household.

```{code-cell} ipython3
model = create_wealth_model()
household_params, aggregate_params = model
w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean = household_params
a, b, σ_z, z_mean, z_var = aggregate_params 
ts_length = 200
w = wealth_time_series(model, y_mean, ts_length)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w)
plt.show()
```

Notice the large spikes in wealth over time.

Such spikes are related to heavy tails in the wealth distribution, which we
discuss below.

+++

Here's function to simulate a cross section of households forward in time.

Note the use of parallelization to speed up computation.

```{code-cell} ipython3
@numba.jit(parallel=True)
def update_cross_section(model, w_distribution, z_sequence):
    """
    Shifts a cross-section of household forward in time

    Takes 

        * a current distribution of wealth values as w_distribution and
        * an aggregate shock sequence z_sequence

    and updates each w_t in w_distribution to w_{t+j}, where
    j = len(z_sequence).

    Returns the new distribution.

    """
    # Unpack
    household_params, aggregate_params = model

    num_households = len(w_distribution)
    new_distribution = np.empty_like(w_distribution)
    z = z_sequence

    # Update each household
    for i in numba.prange(num_households):
        w = w_distribution[i]
        for t in range(sim_length):
            w = update_wealth(household_params, w, z[t])
        new_distribution[i] = w
    return new_distribution
```

Parallelization works in the function above because the time path of each
household can be calculated independently once the path for the aggregate state
is known.

+++

Let's see how long it takes to shift a large cross-section of households forward
200 periods

```{code-cell} ipython3
sim_length = 200
num_households = 10_000_000
ψ_0 = np.full(num_households, y_mean)  # Initial distribution
z_sequence = generate_aggregate_state_sequence(aggregate_params,
                                               length=sim_length)
print("Generating cross-section using Numba")
start_time = time()
ψ_star = update_cross_section(model, ψ_0, z_sequence)
numba_time = time() - start_time
print(f"Generated cross-section in {numba_time} seconds.\n")
```

```{code-cell} ipython3

```


### Pareto tails

In most countries, the cross-sectional distribution of wealth exhibits a Pareto
tail (power law).


Let's see if our model can replicate this stylized fact by running a simulation
that generates a cross-section of wealth and generating a suitable rank-size plot.

We will use the function `rank_size` from `quantecon` library.

In the limit, data that obeys a power law generates a straight line.

```{code-cell} ipython3
model = create_wealth_model()
ψ_star = update_cross_section(model, ψ_0, num_households, z_sequence)
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(ψ_star, c=0.001)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```


### Lorenz curves and Gini coefficents

To study the impact of parameters on inequality, we examine Lorenz curves
and the Gini coefficients at different parameters.

QuantEcon provides functions to compute Lorenz curves and gini coefficients that are accelerated using Numba.

Here we provide JAX-based functions that do the same job and are faster for large data sets on parallel hardware.


#### Lorenz curve

Recall that, for sorted data $w_1, \ldots, w_n$, the Lorenz curve 
generates data points $(x_i, y_i)_{i=0}^n$  according to

\begin{equation*}
    x_0 = y_0 = 0
    \qquad \text{and, for $i \geq 1$,} \quad
    x_i = \frac{i}{n},
    \qquad
    y_i =
       \frac{\sum_{j \leq i} w_j}{\sum_{j \leq n} w_j}  
\end{equation*}


Exercise: write code for this

Solution:

```{code-cell} ipython3
def _lorenz_curve_jax(w, w_size):
    n = w.shape[0]
    w = np.sort(w)
    x = np.arange(n + 1) / n
    s = np.concatenate((np.zeros(1), np.cumsum(w)))
    y = s / s[n]
    return x, y
```

Let's test

```{code-cell} ipython3
sim_length = 200
num_households = 1_000_000
ψ_0 = np.full(num_households, y_mean)  # Initial distribution
z_sequence = generate_aggregate_state_sequence(aggregate_params,
                                               length=sim_length)
z_sequence = np.array(z_sequence)
```

```{code-cell} ipython3
ψ_star = update_cross_section_jax_compiled(
        model, ψ_0, num_households, z_sequence
)
```


```{code-cell} ipython3
x, y = lorenz_curve(ψ_star, num_households)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x, y, label="Lorenz curve at defaults")
ax.plot(x, x, 'k-', lw=1)
ax.legend()
plt.show()
```


#### Gini Coefficient

+++

Recall that, for sorted data $w_1, \ldots, w_n$, the Gini coefficient takes the form

\begin{equation}
    \label{eq:gini}
    G :=
    \frac
        {\sum_{i=1}^n \sum_{j = 1}^n |w_j - w_i|}
        {2n\sum_{i=1}^n w_i}.
\end{equation}

Exercise: Write a function that computes the Gini coefficient using vectorization.

```{code-cell} ipython3
def gini(w, w_size):
    w_1 = np.reshape(w, (w_size, 1))
    w_2 = np.reshape(w, (1, w_size))
    g_sum = np.sum(np.abs(w_1 - w_2))
    return g_sum / (2 * w_size * np.sum(w))
```

```{code-cell} ipython3
gini = gini(ψ_star, num_households)
gini
```
