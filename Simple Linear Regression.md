## 1.1 Regression Basics

### What is Regression Modelling

- Analyze and estimate the values of a response variable by using other variables it is **correlated to**
- Regression analysis is to investigate the relationship between two or more variables in a **non-deterministic way** (there is random error in the model)

---
### Response and Predicting Variables
Response Variable - also known as dependent variable and usually denoted by $Y$  
A random variable because it varies with changes in the predicting variables

Predicting or Explanatory Variables - also known as independent variables and usually denated by $X_1,X_2$  
A fixed variable because it does not varies with the response, it is set fixed before response is measured

In experimental/observational studies, we will fix the predicting variables and then measure the response variable

---
### Regression to Explain Variability in $Y$
A regression model aims to explain and hence predict the **total variability of $Y$** by using $X$. 

---
### Objectives in Regression Analysis
- Predicting how response variable behaves in different setting
- Modeling the relationship between response and predicting variables
- Testing hypotheses of association relationships

Linear models are simple and easy to understand and work well for a wide range of applications.  
The linear models is NOT a true representation of reality but it provides a useful representation of the reality.

---
### Linear Regression: General Model

**Simple linear regression**

$$
Y = \beta_0 + \beta_1X + \epsilon
$$

Relationship between two factors, in a non-determintistic way  
We estimate the relationship between the response and predicting variables as one straight line  
The line doesn't fit the point perfectly but the points are around the line

**Multple linear regression**

$$
Y= \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon
$$

Extension of simple linear regression to include more than one predicting variable
We estimate the relationship as a plane

**Polynomial regression**
$$Y=\beta_0 + \beta_1X+\beta_2X^2 + \epsilon$$
Captures more complex relationships between response and predicting variables  
We are captures nonlinear relationship
This also falls under the linear regression framework because it can be translated into linear regression, we can think of $X$ and $X^2$ as two different predicting variables and model using linear regression

**Linear regression is a very general model**

---

## 1.2 Estimation Method

### Simple Linear Regression
**Model**  
The goal is to fit a non-deterministic linear model (a line) that describes the linear relationship
$$Y=\beta_0 + \beta_1X + \epsilon$$
$\beta_0$ - intercept  
$\beta_1$ - slope  
$\epsilon$ - (random) deviance of the data from the linear model

**Data**  
We have $n$ pairs of data which consists of a value for response variable and a value for the predicting variables
$$\text{Data:}\{(x_1,y_1),(x_2,y_2)...(x_n,y_n)\}$$

**Model Assumption**  
1. Linearity/Mean Zero Assumption: $\mathbb{E}(\epsilon_i) = 0$
2. Constant Variance Assumption: $\text{Var}(\epsilon_i) = \sigma^2$
3. Independence Assumption: $\{\epsilon_1,...\epsilon_n\}$ are independent random variables
4. Normality for $\epsilon_i$ (Required only for statistical inference, CI, PI and Hypothesis testing)

<span style="color:red">
QUESTION: Does this mean that if 1 to 3 is satisfied, the goodness of fit is good. But the inference for CI, PI and testing may not be accurate
</span>

|Assumption|Explanation|Violation|
|:--:|:--|:--|
|Linearity/Mean Zero|Expected value of the errors is zero<br>it cannot be true that model is too high or too low <br>for certain subgroups in the population|Difficulty in estimating $\beta_0$<br> Model does not include a necessary systematic component|
|Constant Variance|Cannot be true that mode is more accurate for some <br>parts of the data and less accurate for other parts of the data|Estimates are not as efficient as they could be in <br>estimating the true parameters<br> Better estimates can be derived<br>Also results in poorly-calibrated prediction intervals|
|Independence|Deviances or in fact the response variables are independently <br>drawn from the data-generating process<br>it cannot be true that the model under-predicts $y$ for one <br>particular case will tell you anything about other cases.|Often occurs in data that are ordered in time (time series)<br>Misleading assessment of the strength of the regression|
|Normality|Errors are assumed to be normally distributed<br> Required for statistical inference such as Confidence and<br>Prediction intervals and Hypothesis testing|Hypothesis tests and confidence and prediction intervals<br>can be misleading|

**Model Paramters**  
Goal of modeling is to estimate model parameters from the observed data.  

$\beta_0$ - intercept  
$\beta_1$ - slope  
$\sigma^2$ - variance of the error terms  

Important to know that parameters in statistical models are unknown quantities and they stay unknown regardless of ow much data are observed.  
We estimate those parameters given the model assumption and the data.  
Estimation does not identify the true parameter value  
We are *estimating approximate of those parameters*

**Estimation Approach**  
The best line is one that makes the errors as small as possible given a criterion. The deviances of the data from the line (Model) is also called error terms

Minimize the sum of squared residuals or errors with respect to $\beta_0,\beta_1$
$$\sum^n_{i=1} (y_i-(\beta_0 - \beta_1x_i))^2$$
$$\hat{\beta}_0 = \bar{y} - \hat{\beta_1}\bar{x}$$
$$\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{\sum y_i(x_i-\bar{x})}{\sum (x_i - \bar{x})^2}$$
$\hat{\beta}_0, \hat{\beta}_1$ are estimated values and they are different from the actual parameter $\beta_0,\beta_1$

Least square error criterion is used in standard regression models but other criteria can be used. The estimated regression line would be different depending on the criterion used.

Minimization problem
$$\min_{\beta_0,\beta_1} \sum^n_{i=1} (y_i-(\beta_0+\beta_1x_i))^2$$
To solve, take first partial derivative of the objective function and equate it to zero
$$\frac{\partial}{\partial \beta_0}\sum^n_{i=1} (y_i-(\beta_0+\beta_1x_i))^2=0$$
$$\frac{\partial}{\partial \beta_1}\sum^n_{i=1} (y_i-(\beta_0+\beta_1x_i))^2=0$$
Solving the equations will give
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$
$$\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{\sum y_i(x_i-\bar{x})}{\sum (x_i - \bar{x})^2}$$

---
### Sums of squares definitions
$$S_{xx} = \sum_i (x_i-\bar{x})^2 = \sum_i x_i^2 - n\bar{x}^2$$
$$S_{xy} = \sum_i (x_i-\bar{x})(y_i-\bar{y}) = \sum_i x_iy_i-n\bar{x}\bar{y}$$
$$S_{yy} = \sum_i (y_i - \bar{y})^2 = \sum_i y_i^2 - n\bar{y}^2$$

---
### Fitted Values, Residuals and Mean Squared Error (MSE)
Given the estimates of $\beta_0,\beta_1$

Fitted values:
$$\hat{y_i} = \hat{\beta}_0 + \hat{\beta}_1x_i$$

Residuals (proxies for error/deviance term in the model):
$$r_i = \hat{\epsilon}_i = y_i - \hat{y}_i$$

Mean Squared error, the estimator for $\sigma^2$
$$\text{MSE} = \frac{SSE}{n-2} = \frac{\sum r_i^2}{n-2}$$
$\sigma^2$ is the variance of error term which is the third model parameter we want to estimate

---
### Variance Sampling Distribution
<span style="color:#008080">
*Note: Variance is call sampling distribution because variance is a statistics. A statistics (mean, variance etc) describes properties of a sample (of a certain size) and varies across samples. A parameter describes properties of a population.
</span>
$$\hat{\sigma}^2 = \frac{\sum \hat{\epsilon}^2}{n-2} \sim \chi^2_{n-2}$$
Assuming
$$\hat{\epsilon}_i \sim \epsilon_i \sim N(0,\sigma^2)$$
The sampling distribution of the estimator of the variance is chi-square with $n-2$ degrees of freedom.

- This distribution is under the assumption of normality of the error term
- $\hat{\epsilon}$ is proxy for deviance/error term and because we don't have the actual deviances $\epsilon$ since we do not have the actual $\beta_0,\beta_1$
- if we replace $\beta_0,\beta_1$ with $\hat{\beta}_0,\hat{\beta}_1$, we get $\hat{\epsilon}_i$ (aka residual)
- Since we are estimating $\sigma^2$ based on residuals $\hat{\epsilon}$, the estimator of the variance of the error term is now the sample variance

**What is sample variance estimation?**  
If we have $Z$ that is normally distributed
$$Z_1,Z_2...Z_n \sim N(\mu,\sigma^2)$$
The sample variance estimator, $S^2$
$$S^2 = \frac{\sum (Z_i-\bar{Z})^2}{n-1}$$
$\bar{Z}$ is used to replace the true parameter $\mu$

Sampling distribution of the sample variance estimator, $S^2$
$$S^2 \sim \chi^2_{n-1}$$
We lose a degree of freedom $(n-1)$ because of the replacement of 1 parameter $/mu$ with an estimator $\bar{Z}$
$$S^2 = \frac{\sum(z_i-\bar{z})^2}{n-1}, \quad \quad \sigma^2 = \sum(z_i-\bar{z})^2$$
$$\frac{(n-1)S^2}{\sigma^2}\sim \chi^2_{n-1}$$
Based on the definition of $S^2$ and $\sigma^2$, the above expression is also a $\chi^2_{n-1}$ distribution

**If $\epsilon_i$ is $z_i$ then it's sample variance estimator $S^2$ should have degree of freedom $n-1$ but why is it $n-2$?**
$$\epsilon = (y_i - (\beta_0 + \beta_1 x_i))$$
This is because the deviance/error term is replaced with residual $r_i(\hat{\epsilon})$ which contains two parameters $\beta_0,\beta_2$ with their estimators. This means the loss of two degrees of freedom insead of one (when $\mu$ is estimated by $\bar{Z}$)
Thus if
$$\epsilon \sim N(0,\sigma^2)$$
Then its variance sampling distribution 
$$\hat{\sigma}^2 = \text{MSE} \sim \chi^2_{n-2}$$

---
### Model Parameter Interpretation

Commonly interested in $\beta_1$
- positive value for $\beta_1$ is consistent with direct relationship between predicting variable $x$ and response variable $y$
- negative value for $\beta_1$ is consistent with an inverse relationship between predicting variable $x$ and response variable $y$
- Close to zero value of $\beta_1$ can be interpret as not a significant association between $x$ and $y$

---
### Model Estimate Interpretation
Least squares estimated coefficients 

$\hat{\beta}_1$ - estimated expected change in response variable associated with one unit of change in the predicting variable

$\hat{\beta}_0$ - estimated expected value of the response variable when the predicting variable equals to zero

However, whenever we make statistical statements about relationship, we will have to mention statistical significance.
- Statistically significantly positive/negative
- No statisitical significance

## 1.3 Statistical Inference

### Statistical Properties of $\beta_1$

$$\mathbb{E}(\hat{\beta}_1) = \beta_1$$
$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{S_{xx}}$$

---
**Unbiased Estimator**
$$\hat{\beta}_1=\frac{\sum(x_i-\bar{x})y_i}{S_{xx}}=c_iy_i$$
$$c_i = \frac{x_i - \bar{x}}{S_{xx}}\quad \text{is fixed since $x_i$ is fixed}$$
$$\mathbb{E}[\hat{\beta}_1] = \mathbb{E}[\sum c_iy_i] = \sum c_i \mathbb{E}[y_i]$$
$\mathbb{E}[\hat{\beta}_1]$ is a linear combination of random variable $y_i$, expectation of a linear combination of random variable is equal to the linear combination of the expectations
$$\sum c_i(\beta_0+\beta_1 x_i) = \beta_0 \sum c_i + \beta_1\sum c_i x_i$$
$$\sum c_i = 0$$
$$\sum c_ix_i = 1$$

Estimator is unbiased
$$\mathbb{E}[\hat{\beta}_1] = \beta_1$$

---
**Normally distributed**

$$\hat{\beta}_1 \sim N(\beta_1,\frac{\sigma^2}{S_{xx}})$$
Since $\mathbb{E}[\hat{\beta}_1]$ is a linear combination of $\{Y_1.Y_2...Y_n\}$  
Assume $\epsilon_i \sim N(0,\sigma^2)$, then 
$$\hat{\beta}_1 \sim N(\beta_1, \frac{\sigma^2}{S_{xx}})$$

Sampling distribution of $\hat{\beta}_1$ is not useful since $\sigma^2$ is unknown and if we replace it with MSE (which has a $\chi^2$ sampling distribution wit $n-2$ degrees of freedom)
$$\hat{\sigma}^2 = \text{MSE} = \frac{\sum \hat{\epsilon}_i^2}{n-2}\sim \chi^2_{n-1}$$
$$\frac{\hat{\beta}_1-\beta_1}{\sqrt{\frac{\text{MSE}}{S_{xx}}}} \sim t_{n-2}$$
Sampling distribution of $\hat{\beta}_1$ becomes a $t$-distribution with $n-2$ degrees of freedom (note, this comes from the variance estimator $\sigma^2$)

---
**Confidence Interval**  
$(1-\alpha)$ confidence interval
$$\hat{\beta}_1 \pm t_{\alpha/2,n-2} \sqrt{\frac{\text{MSE}}{S_{xx}}}$$
Center the confidence interval at the estimate value of $\beta_1$ ($\hat{\beta}_1$)  
Plus,minus the ($1-\alpha$) critical point ($t_{\alpha/2,n-2}$)  
Times standard deviation of the estimator for $\beta_1$ ($\sqrt{\frac{\text{MSE}}{S_{xx}}}$) 

---
**Testing Significance**  
Test statistical significance using the $t$-test (Because it is $t$-distribution)
$$H_0: \beta_1 = 0\quad \text{vs}\quad H_1:\beta_1 \neq 0$$
$t$-value is the difference between the data and the null hypothesis
$$t\text{-value} = \frac{\hat{\beta}_1-0}{\sqrt{\hat{\sigma}^2/S_{xx}}}=\frac{\hat{\beta}_1\sqrt{S_{xx}}}{\hat{\sigma}^2}$$
If $t$-value is large, we reject the null hypothesis, and we interpret it as $\beta_1$ is statistically significant. Statistical significance means that $\beta_1$ is statistically different from zero.

If there is no relationship then slope coefficient is zero, we cannot express $y$ as a linear function of $x$. The scatterplot between $y$ and $x$ would consist of scatter points around a constant line, which is the intercept.

---
**Review of testing significance**  
Test of statistical significance is a test for significance of relationship between $x$ and $y$.  
Compute the test statistics, $t$-value
$$t\text{-value} = t_0 = \frac{\hat{\beta}_1-0}{\sqrt{\hat{\sigma}^2/S_{xx}}}=\frac{\hat{\beta}_1}{se({\hat{\beta}_1})}\sim t_{(n-2)}$$
SE means standard error, it is standard deviation for a sampling distribution ($\hat{\beta}_1$ is a statistic/estimator)

Critical Region, Large $t$-value $\rightarrow$ reject $H_0$
$$|t\text{-value}| > t_{(\alpha/2,n-2)} \Rightarrow \text{Reject }H_0$$

P-Value, Small p-value $\rightarrow$ reject $H_0$
$$p = 2 \times Pr(t_{(\alpha/2,n-2)} > |t_0|)<\alpha \Rightarrow \text{Reject }H_0$$

---
**Interpretation of Significance**  
|Statistically Significant|Statistically Not Significant|
|:--:|:--:|
|Has linear relationship|Has no relationship|
|Has non-linear relationship|Has non-linear relationship|

---
**Testing at different level instead of zero**
$$H_0: \beta_1 = c\quad \text{vs}\quad H_1:\beta_1 \neq c$$
$$t\text{-value} = \frac{\hat{\beta}_1-c}{se({\hat{\beta}_1})}$$
Critical Region, Large $t$-value $\rightarrow$ reject $H_0$
$$|t\text{-value}| > t_{(\alpha/2,n-2)} \Rightarrow \text{Reject }H_0$$

P-Value, Small p-value $\rightarrow$ reject $H_0$
$$p = 2 \times Pr(t_{(\alpha/2,n-2)} > |t_0|)<\alpha \Rightarrow \text{Reject }H_0$$

---
**Testing if $\beta_1$ is positive or negative**  
This becomes a one-tailed test

Test if $\beta_1$ is positive
$$H_0: \beta_1 \leq 0 \quad \text{vs}\quad H_1:\beta_1 > 0$$
$$p = Pr(t_{(\alpha,n-2)} > t\text{-value})<\alpha \Rightarrow \text{Reject }H_0$$
P-value is the area on the right greater than $t$-value  
Reject $H_0$ if P-value is smaller than $\alpha$

Test if $\beta_1$ is negative
$$H_0: \beta_1 \geq 0 \quad \text{vs}\quad H_1:\beta_1 < 0$$
$$p = Pr(t_{(\alpha,n-2)} < t\text{-value})<\alpha \Rightarrow \text{Reject }H_0$$
P-value is the area on the left smaller than $t$-value  
Reject $H_0$ if P-value is smaller than $\alpha$


### Statistical Properties of $\beta_0$

**Unbiased estimator**  
Since
$$\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1\bar{x}$$
$$\mathbb{E}[\hat{\beta}_0] = \mathbb{E}[\hat{Y}] - \mathbb{E}[\hat{\beta}_1]\bar{x} = \beta_0$$
$\beta_0$ is also a linear combination of random variables, $Y$s, and since $Y$ are normally distributed so is $\hat{\beta}_0$  
Since the expectation of $\hat{\beta}_0$ is $\beta_0$ so it is an unbiased estimator of $\beta_0$  
$$\text{Var}(\hat{\beta}_0) = \sigma^2\bigg(\frac{1}{n}+\frac{\bar{x}}{S_{xx}}\bigg)$$
Sampling distribution is also a $t$-distribution when we replace the true variance $\sigma^2$ with its estimator MSE.

**Confidence Interval**
$$\hat{\beta}_0 \pm t_{(\alpha/2,n-2)}\sqrt{\text{MSE}\bigg(\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}}\bigg)}$$
Center of the estimator $\beta_0$ plus/minus the critical point from $t$-distribution times the standard error of $\hat{\beta}_0$

## 1.4 Regression Line: Estimation & Prediction

### Estimation vs Prediction

Estimated mean response:

**Estimation**
- If $x^*$ is one of the observation for the predicting variable, then we use estimation
- Estimated regression line for the value $x^*$ is interpreted as the **average estimated mean response for all settings** under which the predicting variable is equal to $x^*$
- Uncertainty comes from estimation of the regression line

**Prediction**
- If $x^*$ is a new observation of the predicting variable, then we use prediction
- Predicted regression line for the value $x^*$ is interpreted as the **estimated mean response for one setting** under which the predicting variable is equal to $x^*$
- Uncertainty comes from estimation of the regression line and from the newness of the observation

---
### Estimating the Regression Line (Mean Response)

At the selected value of $x^*$, the estimated "mean response" of $y$ is given by
$$\hat{y}|x^* = \hat{\beta}_0 + \hat{\beta}_1x^*$$

Since the estimators of $\beta_0,\beta_1$ are normally distribution so is $\hat{y}$ which means we can draw inference using $\hat{y}$ is we know its expected value and variance.

$$\mathbb{E}[\hat{Y}|X^*] = \beta+\beta_1x^*$$
$$\text{Var}[\hat{Y}|X^*] = \sigma^2\bigg(\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)$$
The expectation of $\hat{Y}$ is the regression line, thus the estimated regression line is an unbiased estimator (just like the estimators of the regression coefficients).  
The variance increases as $x^*$ (value of predicting variable) moves away from the range of the average predicting variable. Variance is smaller at the center of the average and increases as it goes away from teh average of the predicting data. This is translated that the uncertainty in the estimated regression line is going to be higher as $x^*$ moves away from the average. 

---
### Confidence Interval for Mean Response

$$\hat{y}|x^* \pm t_{\alpha/2,n-2}\sqrt{\hat{\sigma}^2\bigg(\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)}$$
The variance changes with $x^*$, confidence interval will be wider the further it is from the mean $\bar{x}$.  
By considering several values of $x^*$, we get a confidence band.  
Confidence bands show why extrapolation does not work since they become wider, too wide to be meaningful, at th ends of the range of the predicting variable.

---
### Predicting a New Response
While the prediction is the same as the estimator for the "mean response" which is $\hat{y}$:  
Prediction contains two sources of uncertainty:
- New $(n+1)$the observation
- Paramter estimates $(\beta_0,\beta_1)$

Variation of the estimated regression line is given:
$$\hat{\sigma}^2\bigg(\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)$$

Variation of a new measurement is $\sigma^2$

Together, the new observation is independent of the regression data, the total variation in predicting $\hat{y}|x^*$ is
$$\hat{\sigma}^2\bigg(\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)+\sigma^2 = \sigma^2 \bigg(1 + \frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)$$
Take note of the added '1' in the variance equation

$$\hat{y}|x^* \pm t_{t/2,n-2} \sqrt{\sigma^2 \bigg(1 + \frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}\bigg)}$$

The difference between the standar error from the confidence for the regression line is the additional of '1'

---
### Confidence Bands
Uncertainty in prediction is higher so the prediction bands are wider.  
For both bands, it is the narrowest at $x=\bar{x}$ and gets wider as we go away from it.  
Sample size also impacts the band, as $n$ increases, the bands get narrower because there is smaller variability.

---
### Estimation vs Prediction Intervals
- Prediction interval should not be confused with confidence interval (narrower)
- Prediction interval is used to provide an interval estimate for a prediction of $y$ for one member of the population with a particular value of $x^*$
- Confidence interval is used to provide an interval estimate for the true average value of $y$ for all members of the population with a particular value of $x^*$

---
### Regression through the Origin
Suppose we fail to reject $H_0: \beta_0 = 0$, so $\beta_0$ is not statistically significant

However, if we set $\beta_0 = 0$ (removing the intercept), we will not have the flexibility of sliding the line up and down.  
Usually, a model without an intercept gives worse result even if passing through origina makes sense.

While we may accept the model $y_i = \beta_1x_i + \epsilon_i$, it is not linear and we do not get precise CI and PI. This model become biased and get very wide as we get distance from $x=0$ and MSE will be higher.

## 1.5 Assumptions and Diagonstics

### Residual Analysis (Residual Plot)
Since we do not have the actual $\beta0,\beta_1$ to compute $\epsilon_i$, we use residual Values as proxy of the error terms. 
$$\epsilon_i \rightarrow \hat{\epsilon}_i = y_i-(\hat{\beta}_0 +\hat{\beta}_1x_i)$$
Plot of the residuals $\epsilon_i$ against fitted values $\hat{y}_i$ and predictive values $x_i$,  
if the scatter of $\epsilon_i$ is not random around zero line:
- relationship between $X$ and $Y$ is not linear (Linearity assumption)
- Variance of error terms not equal (Constant assumption)
- Response data not independent (Independence assumption)

Evaluation of the assumption using graphical diagnostics or hypothesis testing is call goodness-of-fit evaluation.

Important to differential between goodness-of-fit and approaches to evaluate model performance.

---
### Linearity/Mean Zero Assumption (Residual Analysis)
Linearity means the relationship between response and predicting variable is linear, which means the expectation of the error terms is zero
$$\mathbb{E}[\epsilon_i] = 0$$
- Residual plot should be random around the zero line

### Constant Variance Assumption (Residual Analysis)
Variance of the error terms is equal to $\sigma^2$ and the same across all error terms
$$\text{Var}(\epsilon_i) = \sigma^2$$
- Residual plot should show that the variance is the same across the fitted values.

### Independence Assumption (Residual Analysis)
Error terms are independent random variables
- Residuals plots should not show clusters of residual
- Residual analysis is to check for uncorrelated errors NOT to check for independence
- Independence is more complicated and not possible to evaluate if the data are from a randomized trial, then it is independent implicitly but most data are from observational studies
- We would have to evaluate if any transformation of the data would result in correlated data

### Normality Assumption (QQ-plot, Histogram)
Error terms are normallity distributed and this assumption is needed for statistical inference.
$$\epsilon \sim N(0,\sigma^2)$$
- Using Quantile-Quantile plot (plots the quantile of the empirical distribution of residual vs the quantiles of the normal distribution such that the points should form a straight line).
- Large departure indicates departure from normality
- Departure from a straight line could be in the form of a tail, which indicates of either a skewed distribution of heavy-tail distribution.
- Histogram plot used to evaluate shape of distribution, we will plot the histogram of residuals to identify departure from normality (skewness, modality if we have two or more modes, gaps in data etc)
- Use both QQ-plot and histogram to evaluate normality
  
---
### Variable Transformation (Address Assumption Violations)
If one or more assumption do not hold and does not have goodness-of-fit, it does not mean that regression is not useful

We can perform transformation to correct the violations, this is generally a trial-and-error process

**Linearity violation**  
Transform $X$ to improve linear assumption
- if relationship between $X$ and $Y$ is not exactly linear, we can model nonliner relationship by transforming $X$ by some nonlinear function such as $f(x) = x^a$ or $f(x) = \log(x)$

**Normality or Constant Variance violation**  
Transform $Y$ to improve assumption
- tranformation that normalizes or stabilize the variance of the response variable
- Common transformation is a power transformation of $y$ call Box-Cox transformation
$$y \rightarrow y^{\lambda}$$
Where $\lambda$ depends on how Var$(Y)$ changes as $X$ changes  
Note:$\sigma_y(x)$ means the Var$(Y)$ at $x$ value

|Var($Y$) vs $X$|$\lambda$|$y^*$|
|:--:|:--:|:--:|
|$\sigma_y(x)\propto $ consant|1| no transformation|
|$\sigma_y(x)\propto \sqrt{\mu_x}$|1/2| $\sqrt{y}$|
|$\sigma_y(x)\propto \mu_x$|0|$\ln(y)$|
|$\sigma_y(x)\propto 1/\mu_x$|-1|$\frac{1}{y}$|

- After transformation, fit the model again to evaluate the residual for departure from assumptions, if transformation do not address these departures then you may need to consider a different modeling approach

---
### Outliers in Regression

**Outlier**  
A data point far from the majority of the data (in $y$ and $x$) especially if it does not follow the general trend fo the rest of the data

**Leverage Point**  
A data point far from the mean of $x$

**Influential Point**  
- A data point far from the mean of $x$ and/or $y$ if it influences the regression model fit significantly
- They can change the value of the estimated parameters (magnitude or sign) and statistical significance.
- An outlier, including a leverage point, may or may not impact the regression fit, thus it may or may not be a influential point

We should not just discard outliers because sometimes the outlier belong to the data. Sometimes there are good reasons to exclude (error in data entry etc). You will have to perform statistical analysis with and without the outliers and inform how the outlier influeces the regression fit.

**Checking for outliers**  
Looking at standardized residuals
$$r_i^* = \frac{y_i-\hat{y}_1}{\sqrt{\text{MSE}}}$$
- Standardized residual bigger than 1 are large
- Standardized residual bigger than 2 are extremely large

**Effect of outliers**  
If the outliers are also influential points, the fitted regression line will have different coefficients (change in value or sign)

---
### Coefficient of Determiniation, $R^2$

$R^2$ quantifies the predictive power of the model, it is a statistic that summarized how well the predicting variable $x$ can be used to linearly predict the response variable $y$.
$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}}$$
$$\text{SSE} = \sum r_1^2$$
$$\text{SST} = \sum (y_i - \bar{y})^2$$
SSE is the sum of squared errors and SST is sum of square total  
$R^2$ is interpreted as the proportion of totaly variability in the response variable $y$ that can be explained by the linear regression that uses $x$. The interpretation applies only for linear relationships.

### Correlation Coefficient, $\rho$
$\rho$ summarized how well the $X$ are linearly related to $Y$
$$\begin{align*}\rho &= \text{cor}(X,Y)\\
&=\frac{S_{xy}}{\sqrt{S_{XX}}\sqrt{S_{YY}}}\\
&=\hat{\beta_1}\sqrt{\frac{S_{XX}}{S_{YY}}}
\end{align*}$$
$\rho$ can also be used to evaluate variour transformation of $X$ and $Y$ to improve the linearity assumption

In the context of **simple linear regression**
$$\rho^2 = R^2$$

---
### Goodness-of-Fit (GOF) vs $R^2$

GOF
- evaluation of model assumptions
- can be done using visual analytics (Residual analysis, QQ-plot, histogram)
- can be done using hypothesis testing
- can be used as a model performance criteria

$R^2$
- one of the different criteria we can use to evaluate model performance
- measures how much variability in the response is explained by the model

Different performance evaluation can give us different conclusion about the model  
Example: It is possible to have a model that fits the data well but the R-squared may be low, this means the predictors included in the model may not explain the variability in the response.


```R

```
