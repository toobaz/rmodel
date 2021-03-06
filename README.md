# RModel

This is a simple compatibility library that allows calling R estimators as if they were statsmodels estimators.

This allows to tackle, with very small code adaptations, the following use cases:
* from within a Python codebase, use estimators covered by R but not by statsmodels
* from within a Python codebase, use estimators for which the R implementation is preferred
* merge within a single table (which can be created for instance with `python-stargazer` or `statsmodels.iolib.summary2.summary_col`) both results produced by statsmodels and produced from R

## Current status

Only the `lm` R command has been tested so far; other models can be specified with the `command` parameter, and might or might not work.

## Examples

### With stargazer

The following example takes the [python stargazer](https://github.com/mwburke/stargazer) example (with the `from_formula()` statsmodels syntax) and extends it with a new column which is exactly like the previous but estimated in R, plus another similar one which passes a:

```python3
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from rmodel import RModel

diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

est = sm.OLS.from_formula('target ~ age + sex + bmi + bp', data=df).fit()
est2 = sm.OLS.from_formula('target ~ age + sex + bmi + bp + s1 + s2', data=df).fit()
est3 = RModel.from_formula('target ~ age + sex + bmi + bp + s1 + s2', data=df).fit()


stargazer = Stargazer([est, est2, est3])

as_html = stargazer.render_html()
```

<table style="text-align:center"><tr><td colspan="4" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="3"><em>Dependent variable:target</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td><td>(3)</td></tr><tr><td colspan="4" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Intercept</td><td>152.133<sup>***</sup></td><td>152.133<sup>***</sup></td><td>152.133<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(2.853)</td><td>(2.853)</td><td>(2.853)</td></tr><tr><td style="text-align:left">age</td><td>37.241<sup></sup></td><td>24.704<sup></sup></td><td>24.704<sup></sup></td></tr><tr><td style="text-align:left"></td><td>(64.117)</td><td>(65.411)</td><td>(65.411)</td></tr><tr><td style="text-align:left">bmi</td><td>787.179<sup>***</sup></td><td>789.742<sup>***</sup></td><td>789.742<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(65.424)</td><td>(66.887)</td><td>(66.887)</td></tr><tr><td style="text-align:left">bp</td><td>416.674<sup>***</sup></td><td>397.583<sup>***</sup></td><td>397.583<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(69.495)</td><td>(70.870)</td><td>(70.870)</td></tr><tr><td style="text-align:left">s1</td><td></td><td>197.852<sup></sup></td><td>197.852<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(143.812)</td><td>(143.812)</td></tr><tr><td style="text-align:left">s2</td><td></td><td>-169.251<sup></sup></td><td>-169.251<sup></sup></td></tr><tr><td style="text-align:left"></td><td></td><td>(142.744)</td><td>(142.744)</td></tr><tr><td style="text-align:left">sex</td><td>-106.578<sup>*</sup></td><td>-82.862<sup></sup></td><td>-82.862<sup></sup></td></tr><tr><td style="text-align:left"></td><td>(62.125)</td><td>(64.851)</td><td>(64.851)</td></tr><td colspan="4" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Observations</td><td>442</td><td>442</td><td>442</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.400</td><td>0.403</td><td>0.403</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.395</td><td>0.395</td><td>0.395</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>59.976 (df=437)</td><td>59.982 (df=435)</td><td>59.982 (df=435)</td></tr><tr><td style="text-align: left">F Statistic</td><td>72.913<sup>***</sup> (df=4; 437)</td><td>48.915<sup>***</sup> (df=6; 435)</td><td>48.915<sup>***</sup> (df=6; 435)</td></tr><tr><td colspan="4" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td>
 <td colspan="3" style="text-align: right">
  <sup>*</sup>p&lt;0.1;
  <sup>**</sup>p&lt;0.05;
  <sup>***</sup>p&lt;0.01
 </td></tr></table>


### With `summary_col`

The same can be done with `statsmodels`' `summary_col`:

```python3
table = summary_col([est, est2, est3])
```

### With R model loaded from disk

The following R code will run a simple regression in R and save the desired results:

```R
require("datasets")
data("cars")

res = lm('dist ~ speed', data=cars)

rsum = summary(res)

save(rsum, file='precious_results.RData')
```

The following Python code reloads the R results from disk, as a statsmodels-like object:
```python3
r_est = RModel.from_rda("precious_results.RData", 'rsum')
```

The following Python code retrieves the same data from R for comparison, and runs the analogous statsmodels model:
```python3
from rpy2.robjects import r
r("data(cars)")
cars = r['cars']

py_est = sm.OLS.from_formula('dist ~ speed', data=cars).fit()
```

We can now summarize the two models together:
```python3
stargazer = Stargazer([py_est, r_est])
as_html = stargazer.render_html()
```

<table style="text-align:center"><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left"></td><td colspan="2"><em>Dependent variable:dist</em></td></tr><tr><td style="text-align:left"></td><tr><td style="text-align:left"></td><td>(1)</td><td>(2)</td></tr><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align:left">Intercept</td><td>-17.579<sup>**</sup></td><td>-17.579<sup>**</sup></td></tr><tr><td style="text-align:left"></td><td>(6.758)</td><td>(6.758)</td></tr><tr><td style="text-align:left">speed</td><td>3.932<sup>***</sup></td><td>3.932<sup>***</sup></td></tr><tr><td style="text-align:left"></td><td>(0.416)</td><td>(0.416)</td></tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Observations</td><td>50</td><td>1</td></tr><tr><td style="text-align: left">R<sup>2</sup></td><td>0.651</td><td>0.651</td></tr><tr><td style="text-align: left">Adjusted R<sup>2</sup></td><td>0.644</td><td>0.644</td></tr><tr><td style="text-align: left">Residual Std. Error</td><td>15.380 (df=48)</td><td>15.380 (df=48)</td></tr><tr><td style="text-align: left">F Statistic</td><td>89.567<sup>***</sup> (df=1; 48)</td><td>89.567<sup>***</sup> (df=1; 48)</td></tr><tr><td colspan="3" style="border-bottom: 1px solid black"></td></tr><tr><td style="text-align: left">Note:</td>
 <td colspan="2" style="text-align: right">
  <sup>*</sup>p&lt;0.1;
  <sup>**</sup>p&lt;0.05;
  <sup>***</sup>p&lt;0.01
 </td></tr></table>

## Passing keyword arguments

Arguments to the R command can be transparently passed as keyword arguments.
In the following example, a Weighted OLS is estimated by passing the `weights`
argument to the `lm` R command.

```python3
# "age" is a standardized variable - we want positive weights:
df['weights'] = df['age'] - df['age'].min()
est4 = RModel.from_formula('target ~ age + sex + bmi + bp + s1 + s2', data=df, weights='weights').fit()
```
