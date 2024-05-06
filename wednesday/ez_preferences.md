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

# Recursive Utility: Solution Methods 

------

#### Prepared for the CBC Workshop May 2024
#### John Stachurski

------

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

jax.config.update("jax_enable_x64", True)
```

In (nonlinear) recusive utility models (e.g., Epstein-Zin preferences,
risk-sensitive preferences), lifetime utility is defined recursively.

In this lecture we discuss some methods for computing the solution of discrete
time recursive utility models.


$$
V_t = \left\{
        C_t^\rho + \beta (\mathbb E_t V_{t+1}^\gamma)^{\rho/\gamma} 
      \right\}^{1/\rho}
$$

One way to understand this expression is to think of it as



$$
V_t = \left\{
        C_t^\rho + \beta (\mathcal E_t V_{t+1})^\rho
      \right\}^{1/\rho}
$$

where $\mathcal E$ computes the ``risk-adjusted expectation'' 

$$
\mathcal E Y = (\mathbb E Y^\gamma)^{1/\gamma}
$$

* $\gamma$ governs risk aversion
* $\rho$ governs elasticity of intertemporal substitution

We suppose $C_t = c(X_t)$ where $X_t$ is a Markov process taking values in state
space $S$.

We guess the solution has the form $V_t = v(X_t)$ for all $t$, where $v$ is some
function over the state space $S$.

In this case we can write the above equation as

$$
v(X_t) = \left\{
        c(X_t)^\rho + \beta (\mathbb E_t v(X_{t+1})^\gamma)^{\rho/\gamma} 
      \right\}^{1/\rho}
$$

Let's suppose that $(X_t)$ is a Markov chain with transition matrix $P$.

Then the last equation tells us that we need to solve for $v$ in

$$
    v(x) 
    = \left\{
        c(X_t)^\rho + \beta (\sum_{x'} v(x')^\gamma P(x, x'))^{\rho/\gamma} 
      \right\}^{1/\rho}
$$


We define the operator $K$ sending $v$ into $Kv$ by

$$
    (Kv)(x) 
    = \left\{
        c(X_t)^\rho + \beta (\sum_{x'} v(x')^\gamma P(x, x'))^{\rho/\gamma} 
      \right\}^{1/\rho}
$$

Now we pick an initial guess of $v$ and iterate.

For the state process we use a Tauchen discretization of 

$$
    X_{t+1} = \alpha X_t + \sigma Z_{t+1},
    \qquad \{Z_t\} \text{ is IID and } N(0, 1)
$$

We assume that $c(x) = \exp(x)$.

```{code-cell} ipython3
c = jnp.exp
```

```{code-cell} ipython3
Model = namedtuple('Model', (ρ, β, α, σ, x_vals, c_vals, P))

def create_ez_model(ρ=2.2,
                    γ=0.5,
                    α=0.9,
                    β=0.99,
                    σ=0.1,
                    n=500):
    mc = qe.tauchen(n, α, σ)
    x_vals, P = jnp.exp(mc.state_values), mc.P
    c_vals = c(x_vals)
    P = jnp.array(P)
    return Model(ρ, β, α, σ, x_vals, c_vals, P)

def K(v, model):
    ρ, β, α, σ, x_vals, c_vals, P = model
    (c_vals**ρ + β * (P @ v**γ)**(1 / γ))**(1 / ρ)


def successive_approx(model, max_iter=10_000, tol=1e-5):
    model = create_ez_model()
    ρ, β, α, σ, x_vals, c_vals, P = model
    w = jnp.ones(len(x_vals))

    i, error = 0, tol + 1
    while error > tol and i < max_iter:
        v_new = K(v, model)
        error = np.max(np.abs(v_new - v))
        i += 1
        v = v_new
    return v


```
