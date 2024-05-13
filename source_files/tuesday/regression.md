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

# Linear Regression with Python

----

#### John Stachurski
#### Prepared for the CBC Computational Workshop (May 2024)

----

+++

Let's have a very quick look at linear regression in Python.

We'll also show how to download some data from [FRED](https://fred.stlouisfed.org/).

+++

Uncomment the next line if you don't have this library installed:

```{code-cell} ipython3
#!pip install pandas_datareader
```

Let's do some imports.

```{code-cell} ipython3
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

We use the `datetime` module from the standard library to pick start dates and end dates.

```{code-cell} ipython3
start = datetime.datetime(1947, 1, 1)
end = datetime.datetime(2019, 12, 1)
```

Now let's read in data on GDP and unemployment from FRED.

```{code-cell} ipython3
series = 'GDP', 'UNRATE'  
source = 'fred'
data = web.DataReader(series, source, start, end)
```

Data is read in as a `pandas` dataframe.

```{code-cell} ipython3
type(data)
```

```{code-cell} ipython3
data = data.dropna()
data
```

We'll convert both series into rates of change.

```{code-cell} ipython3
data = data.pct_change() * 100
data = data.reset_index().dropna()
data
```

Let's have a look at our data.

Notice in the code below that Matplotlib plays well with pandas.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x='UNRATE', y='GDP', data=data, color='k', alpha=0.5)
ax.set_xlabel('% change in unemployment rate')
ax.set_ylabel('% change in GDP')
plt.show()
```

If you want an interactive graph, you can use Plotly instead:

```{code-cell} ipython3
fig = px.scatter(data, x='UNRATE', y='GDP')
fig.update_layout(
    showlegend=False,
    autosize=False,
    width=600,
    height=400,
)

fig.show()
```

Let's fit a regression line, which can be used to measure [Okun's law](https://en.wikipedia.org/wiki/Okun%27s_law).

To do so we'll use [Statsmodels](https://www.statsmodels.org/stable/index.html).

```{code-cell} ipython3
model = smf.ols(formula='GDP ~ UNRATE', data=data)
ols = model.fit()
```

```{code-cell} ipython3
ols.summary()
```

```{code-cell} ipython3
ols.params
```

```{code-cell} ipython3
X = data['UNRATE']
fig, ax = plt.subplots()
plt.scatter(x='UNRATE', y='GDP', data=data, color='k', alpha=0.5)
plt.plot(X, ols.fittedvalues, label='OLS')
ax.set_xlabel('% change in unemployment rate')
ax.set_ylabel('% change in GDP')
plt.legend()
plt.show()
```

Next let's try using least absolute deviations, which means that we minimize

$$
\ell(\alpha, \beta) = \sum_{i=1}^n |y_i - (\alpha x_i + \beta_i)|
$$

over parameters $\alpha, \beta$.

This is a special case of quantile regression when the quantile is the median (0.5).

```{code-cell} ipython3
mod = smf.quantreg(formula="GDP ~ UNRATE", data=data)
lad = mod.fit(q=0.5)   # LAD model is a special case of quantile regression when q = 0.5
```

```{code-cell} ipython3
lad.summary()
```

```{code-cell} ipython3
lad.params
```

Let's compare the LAD regression line to the least squares regression line.

```{code-cell} ipython3
fig, ax = plt.subplots()
plt.scatter(x='UNRATE', y='GDP', data=data, color='k', alpha=0.5)
plt.plot(X, ols.fittedvalues, label='OLS')
plt.plot(X, lad.fittedvalues, label='LAD')
ax.set_xlabel('% change in unemployment rate')
ax.set_ylabel('% change in GDP')
plt.legend()
plt.show()
```

```{code-cell} ipython3

```
