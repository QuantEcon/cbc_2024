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

------

#### John Stachurski
#### Prepared for the CBC Computational Workshop May 2024

-----

## Overview

This lecture explores the inventory dynamics of a firm using so-called s-S inventory control.

Loosely speaking, this means that the firm

- waits until inventory falls below some value $ s $  
- and then restocks with a bulk order of $ S $ units (or, in some models, restocks up to level $ S $).  

This lecture will help use become familiar with NumPy.

(Later we will try similar operations with JAX)

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
```

## Sample paths

Consider a firm with inventory $ X_t $.

The firm waits until $ X_t \leq s $ and then restocks up to $ S $ units.

It faces stochastic demand $ \{ D_t \} $, which we assume is IID across time and
firms.

With notation $ a^+ := \max\{a, 0\} $, inventory dynamics can be written
as

$$
X_{t+1} =
    \begin{cases}
      ( S - D_{t+1})^+ & \quad \text{if } X_t \leq s \\
      ( X_t - D_{t+1} )^+ &  \quad \text{if } X_t > s
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
Parameters = namedtuple('Parameters', ['s', 'S', 'μ', 'σ'])
```

Here's a function that updates from $X_t = x$ to $X_{t+1}$

```{code-cell} ipython3
def update(params, x):
    """
    Update the state from t to t+1 given current state x.
    
    """
    s, S, μ, σ = params
    Z = np.random.randn()
    D = np.exp(μ + σ * Z)
    return max(S - D, 0) if x <= s else max(x - D, 0)
```

Here's a function that generates a time series.

```{code-cell} ipython3
def sim_inventory_path(x_init, params, sim_length):
    """
    Simulate a time series (X_t) with X_0 = x_init.

    """
    X = np.empty(sim_length)
    X[0] = x_init
    for t in range(sim_length-1):
        X[t+1] = update(params, X[t])
    return X
```

Let's test it.

```{code-cell} ipython3
params = Parameters(s=10, S=100, μ=1.0, σ=0.5)
s, S = params.s, params.S
sim_length = 100
x_init = 50

X = sim_inventory_path(x_init, params, sim_length)

fig, ax = plt.subplots()
bbox = (0., 1.02, 1., .102)
legend_args = {'ncol': 3,
               'bbox_to_anchor': bbox,
               'loc': 3,
               'mode': 'expand'}

ax.plot(X, label="inventory")
ax.plot(np.full(sim_length, s), 'k--', label="$s$")
ax.plot(np.full(sim_length, S), 'k-', label="$S$")
ax.set_ylim(0, S+10)
ax.set_xlabel("time")
ax.legend(**legend_args)

plt.show()
```

## Cross-sectional distributions

Now let’s look at the marginal distribution $ \psi_T $ of $ X_T $ for some fixed $ T $.

The probability distribution $ \psi_T $ is the time $ T $ distribution of firm
inventory levels implied by the model.

We will approximate this distribution by

1. fixing $ n $ to be some large number, indicating the number of firms in the
  simulation,  
1. fixing $ T $, the time period we are interested in,  
1. generating $ n $ independent draws from some fixed distribution $ \psi_0 $ that gives the
  initial cross-section of inventories for the $ n $ firms, and  
1. shifting this distribution forward in time $ T $ periods, updating each firm
  $ T $ times via the dynamics described above (independent of other firms).  


We will then visualize $ \psi_T $ by histogramming the cross-section.

We will use the following code to update the cross-section of firms by one period.

```{code-cell} ipython3
def update_cross_section(params, X):
    """
    Update by one period a cross-section of firms with inventory levels 
    given by array X.  (Thus, X[i] is the inventory of the i-th firm.)
    
    """
    s, S, μ, σ = params
    num_firms = len(X)
    Z = np.random.randn(num_firms)
    D = np.exp(μ + σ * Z)
    X_new = np.where(X <= s, np.maximum(S - D, 0), np.maximum(X - D, 0))
    return X_new
```

### Shifting the cross-section

Now we provide code to compute the cross-sectional distribution $ \psi_T $ given some
initial distribution $ \psi_0 $ and a positive integer $ T $.

In the code below, the initial distribution $ \psi_0 $ takes all firms to have
initial inventory `x_init`.

```{code-cell} ipython3
def shift_cross_section(params, X, T):
    """
    Shift the cross-sectional distribution X = X[i] forward by T periods.
    
    """
    for i in range(T):
        X = update_cross_section(params, X)
    return X
```

We’ll use the following specification

```{code-cell} ipython3
x_init = 50  #  All firms start at this value
num_firms = 100_000
X = np.full(num_firms, x_init)
T = 500
```

```{code-cell} ipython3
X = shift_cross_section(params, X, T)
```

Here’s a histogram of inventory levels at time $ T $.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(X, bins=50, 
        density=True, 
        histtype='step', 
        label=f'cross-section when $t = {T}$')
ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

## Distribution dynamics

Next let’s take a look at how the distribution sequence evolves over time.

Here is code that repeatedly shifts the cross-section forward while
recording the cross-section at the dates in `sample_dates`.

All firms start at the same level `x_init`.

```{code-cell} ipython3
def shift_forward_and_sample(x_init, params, sample_dates,
                             num_firms=50_000):

    X = np.full(num_firms, x_init)
    X_samples = []
    sim_length = sample_dates[-1] + 1
    # Use for loop to update X and collect samples
    for i in range(sim_length):
        if i in sample_dates:
            X_samples.append(X)
        X = update_cross_section(params, X)

    return X_samples
```

Let’s test it

```{code-cell} ipython3
x_init = 50
num_firms = 10_000
sample_dates = 10, 50, 250, 500

X_samples = shift_forward_and_sample(x_init, params, sample_dates)
```

Let’s plot the output.

```{code-cell} ipython3
fig, ax = plt.subplots()

for i, date in enumerate(sample_dates):
    ax.hist(X_samples, bins=50, 
            density=True, 
            histtype='step',
            label=f'cross-section when $t = {date}$')

ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

This model for inventory dynamics is asymptotically stationary, with a unique
stationary distribution.

In particular, the sequence of marginal distributions $ \{\psi_t\} $
converges to a unique limiting distribution that does not depend on
initial conditions.

That's why, by $ t=500 $, the distributions are barely changing.

If you test a few different initial conditions, you will see that they do not affect long-run outcomes.

+++

## Exercise: Restock frequency

Let’s study the probability that firms need to restock at least twice over periods $1, \ldots, 50$ when $ X_0 = 70 $.

We will do this by Monte Carlo:

* Set the number of firms to `1_000_000`.
* Calculate the fraction of firms that need to order twice or more in the first 50 periods.  

This proportion approximates the probability of the event when the sample size
is large.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for i in range(18):
    print("Solution below!")
```

```{code-cell} ipython3
def compute_freq(params,
                 x_init=70,
                 sim_length=50,
                 num_firms=1_000_000):
    s = params.s
    # Prepare initial arrays
    X = np.full(num_firms, x_init)
    # Restock counter starts at zero
    counter = np.zeros(num_firms)

    for i in range(sim_length):
        X = update_cross_section(params, X) 
        counter = np.where(X <= s, counter + 1, counter)
    return np.mean(counter > 1, axis=0)
```

```{code-cell} ipython3
freq = compute_freq(params)
print(f"Frequency of at least two stock outs = {freq}")
```

```{code-cell} ipython3

```
