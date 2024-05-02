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

```{code-cell} ipython3
#!pip install pandas_datareader
```

```{code-cell} ipython3
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

```{code-cell} ipython3
start = datetime.datetime(1947, 1, 1)
end = datetime.datetime(2019, 12, 1)
data = web.DataReader(['GDP', 
                       'UNRATE', 
                      ], 'fred', start, end)
```

```{code-cell} ipython3
data = data.dropna()
data
```

```{code-cell} ipython3
data = data.pct_change() * 100
data = data.reset_index().dropna()
data
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x='UNRATE', y='GDP', data=data, color='k', alpha=0.5)
ax.set_xlabel('% change in unemployment rate')
ax.set_ylabel('% change in GDP')
plt.show()
```

```{code-cell} ipython3
fig = px.scatter(data, x='UNRATE', y='GDP', 
                 trendline='ols', 
                 trendline_color_override='black')
fig.update_layout(
    showlegend=False,
    autosize=False,
    width=600,
    height=400,
)

fig.show()
```

```{code-cell} ipython3
X = data['UNRATE']
Y = data['GDP']
X_cons = sm.add_constant(X)
```

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
fig, ax = plt.subplots()
plt.scatter(x='UNRATE', y='GDP', data=data, color='k', alpha=0.5)
plt.plot(X, ols.fittedvalues, label='OLS')
ax.set_xlabel('% change in unemployment rate')
ax.set_ylabel('% change in GDP')
plt.legend()
plt.show()
```

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

```{code-cell} ipython3

```
