# Introduction to Statistical Methods

**Course Code**: S1-25_AIMLCZC418  
**Semester**: 1 (2025-26)  
**Category**: Core Course

---

## 📋 Course Overview

This course provides a comprehensive foundation in statistical methods essential for data analysis and machine learning. You'll learn probability theory, statistical inference, hypothesis testing, and how to apply these concepts to real-world data problems.

## 🎯 Learning Objectives

By the end of this course, you will be able to:

- ✅ Understand probability theory and random variables
- ✅ Perform statistical inference and estimation
- ✅ Conduct hypothesis testing
- ✅ Apply regression and correlation analysis
- ✅ Use statistical methods for data analysis
- ✅ Interpret statistical results correctly

---

## 📚 Course Content

### Module 1: Probability Theory

#### 1.1 Fundamentals of Probability
- Sample space and events
- Probability axioms
- Conditional probability
- Bayes' theorem
- Independence

!!! note "Bayes' Theorem"
    $$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$
    
    This fundamental theorem is crucial for Bayesian inference and many ML algorithms.

```python
# Example: Bayesian inference
def bayes_theorem(p_b_given_a, p_a, p_b):
    """
    Calculate P(A|B) using Bayes' theorem
    """
    return (p_b_given_a * p_a) / p_b

# Medical test example
p_disease = 0.01  # Prior probability of disease
p_positive_given_disease = 0.99  # Sensitivity
p_positive = 0.05  # Probability of testing positive

p_disease_given_positive = bayes_theorem(
    p_positive_given_disease, 
    p_disease, 
    p_positive
)

print(f"Probability of disease given positive test: {p_disease_given_positive:.4f}")
```

#### 1.2 Random Variables
- Discrete random variables
- Continuous random variables
- Probability mass functions (PMF)
- Probability density functions (PDF)
- Cumulative distribution functions (CDF)

#### 1.3 Expectation and Variance
- Expected value
- Variance and standard deviation
- Covariance and correlation
- Law of large numbers
- Central limit theorem

**Key Formulas**:

- Expected Value: $E[X] = \sum_{x} x \cdot P(X=x)$ (discrete) or $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$ (continuous)
- Variance: $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$
- Covariance: $\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$

### Module 2: Probability Distributions

#### 2.1 Discrete Distributions
- Bernoulli distribution
- Binomial distribution
- Poisson distribution
- Geometric distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Visualizing common distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Binomial distribution
n, p = 10, 0.5
x_binomial = np.arange(0, n+1)
pmf_binomial = stats.binom.pmf(x_binomial, n, p)
axes[0, 0].bar(x_binomial, pmf_binomial)
axes[0, 0].set_title('Binomial Distribution (n=10, p=0.5)')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('P(X=k)')

# Poisson distribution
mu = 3
x_poisson = np.arange(0, 15)
pmf_poisson = stats.poisson.pmf(x_poisson, mu)
axes[0, 1].bar(x_poisson, pmf_poisson)
axes[0, 1].set_title('Poisson Distribution (λ=3)')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('P(X=k)')

# Normal distribution
mu, sigma = 0, 1
x_normal = np.linspace(-4, 4, 100)
pdf_normal = stats.norm.pdf(x_normal, mu, sigma)
axes[1, 0].plot(x_normal, pdf_normal)
axes[1, 0].set_title('Normal Distribution (μ=0, σ=1)')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].grid(True)

# Exponential distribution
lambda_param = 1.5
x_exp = np.linspace(0, 5, 100)
pdf_exp = stats.expon.pdf(x_exp, scale=1/lambda_param)
axes[1, 1].plot(x_exp, pdf_exp)
axes[1, 1].set_title('Exponential Distribution (λ=1.5)')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('f(x)')
axes[1, 1].grid(True)

plt.tight_layout()
```

#### 2.2 Continuous Distributions
- Uniform distribution
- Normal (Gaussian) distribution
- Exponential distribution
- Chi-square distribution
- Student's t-distribution
- F-distribution

!!! info "Normal Distribution"
    The normal distribution is the most important distribution in statistics:
    
    $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$
    
    It appears naturally due to the Central Limit Theorem.

#### 2.3 Multivariate Distributions
- Joint distributions
- Marginal distributions
- Conditional distributions
- Multivariate normal distribution

### Module 3: Statistical Inference

#### 3.1 Point Estimation
- Method of moments
- Maximum likelihood estimation (MLE)
- Properties of estimators (bias, consistency, efficiency)
- Mean squared error

```python
import numpy as np
from scipy.optimize import minimize

# Example: Maximum Likelihood Estimation for Normal Distribution
def neg_log_likelihood(params, data):
    """
    Negative log-likelihood for normal distribution
    """
    mu, sigma = params
    n = len(data)
    return 0.5 * n * np.log(2 * np.pi * sigma**2) + \
           np.sum((data - mu)**2) / (2 * sigma**2)

# Generate sample data
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 1000)

# Find MLE estimates
initial_guess = [0, 1]
result = minimize(neg_log_likelihood, initial_guess, args=(data,))
mu_mle, sigma_mle = result.x

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_mle:.4f}, σ={sigma_mle:.4f}")
print(f"Sample estimates: μ={np.mean(data):.4f}, σ={np.std(data, ddof=1):.4f}")
```

#### 3.2 Interval Estimation
- Confidence intervals
- Confidence level and margin of error
- CI for means and proportions
- Bootstrap methods

#### 3.3 Hypothesis Testing
- Null and alternative hypotheses
- Type I and Type II errors
- p-values
- Significance level (α)
- Power of a test

**Hypothesis Testing Framework**:

1. State hypotheses ($H_0$ and $H_1$)
2. Choose significance level (α)
3. Calculate test statistic
4. Determine p-value
5. Make decision (reject or fail to reject $H_0$)

```python
from scipy import stats

# Example: One-sample t-test
# H0: μ = 100, H1: μ ≠ 100
sample_data = [98, 102, 95, 105, 100, 97, 103, 99, 101, 96]
hypothesized_mean = 100

# Perform t-test
t_statistic, p_value = stats.ttest_1samp(sample_data, hypothesized_mean)

print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"Reject H0 (p-value {p_value:.4f} < {alpha})")
else:
    print(f"Fail to reject H0 (p-value {p_value:.4f} >= {alpha})")
```

### Module 4: Regression and Correlation

#### 4.1 Correlation Analysis
- Pearson correlation coefficient
- Spearman rank correlation
- Kendall's tau
- Interpretation and limitations

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Correlation analysis
# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'height': np.random.normal(170, 10, 100),
    'weight': np.random.normal(70, 15, 100),
    'age': np.random.randint(20, 60, 100),
    'income': np.random.normal(50000, 20000, 100)
})

# Calculate correlation matrix
corr_matrix = data.corr()

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pearson correlation
from scipy.stats import pearsonr

r, p_value = pearsonr(data['height'], data['weight'])
print(f"Pearson correlation: r={r:.4f}, p-value={p_value:.4f}")
```

#### 4.2 Simple Linear Regression
- Least squares estimation
- Regression coefficients
- Goodness of fit (R²)
- Residual analysis
- Assumptions of linear regression

**Simple Linear Regression Model**:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $\beta_0$: intercept
- $\beta_1$: slope
- $\epsilon$: error term

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example: Simple linear regression
X = np.random.randn(100, 1) * 10
y = 3 * X + 5 + np.random.randn(100, 1) * 2

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Linear Regression: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}')
plt.legend()
plt.grid(True)
plt.show()

print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
```

#### 4.3 Multiple Linear Regression
- Multiple predictors
- Multicollinearity
- Variable selection
- Model diagnostics

### Module 5: Advanced Topics

#### 5.1 ANOVA (Analysis of Variance)
- One-way ANOVA
- Two-way ANOVA
- Post-hoc tests
- Assumptions and violations

#### 5.2 Non-parametric Methods
- Sign test
- Wilcoxon test
- Mann-Whitney U test
- Kruskal-Wallis test
- When to use non-parametric tests

#### 5.3 Chi-Square Tests
- Goodness of fit test
- Test of independence
- Contingency tables

```python
from scipy.stats import chi2_contingency

# Example: Chi-square test of independence
# Observed frequencies
observed = np.array([
    [30, 20, 10],  # Group A
    [15, 25, 20]   # Group B
])

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected frequencies:\n{expected}")

if p_value < 0.05:
    print("\nReject H0: Variables are dependent")
else:
    print("\nFail to reject H0: Variables are independent")
```

---

## 🛠️ Practical Applications in ML

### Application 1: Feature Selection using Correlation

```python
import pandas as pd
from sklearn.datasets import load_boston

# Load data
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

# Calculate correlation with target
correlations = df.corr()['PRICE'].sort_values(ascending=False)
print("Feature correlations with target:")
print(correlations)

# Select highly correlated features
high_corr_features = correlations[abs(correlations) > 0.5].index
print(f"\nHighly correlated features: {list(high_corr_features)}")
```

### Application 2: A/B Testing

```python
from scipy.stats import ttest_ind

# Example: A/B test for website conversion rates
group_a = np.random.binomial(1, 0.10, 1000)  # Control group
group_b = np.random.binomial(1, 0.12, 1000)  # Treatment group

# Calculate conversion rates
conv_a = np.mean(group_a)
conv_b = np.mean(group_b)

# Perform t-test
t_stat, p_value = ttest_ind(group_a, group_b)

print(f"Group A conversion rate: {conv_a:.4f}")
print(f"Group B conversion rate: {conv_b:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference detected!")
else:
    print("No significant difference.")
```

### Application 3: Outlier Detection using Statistical Methods

```python
def detect_outliers_iqr(data):
    """Detect outliers using IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using z-score method"""
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Example
data = np.concatenate([np.random.normal(100, 15, 1000), [200, 250, -50]])
outliers_iqr = detect_outliers_iqr(data)
outliers_zscore = detect_outliers_zscore(data)

print(f"Outliers detected (IQR): {np.sum(outliers_iqr)}")
print(f"Outliers detected (Z-score): {np.sum(outliers_zscore)}")
```

---

## 📖 Key Concepts

!!! abstract "Essential Concepts"
    - **p-value**: Probability of obtaining results at least as extreme as observed, assuming H₀ is true
    - **Confidence Interval**: Range of values likely to contain the true parameter
    - **Central Limit Theorem**: Distribution of sample means approaches normal as n increases
    - **Type I Error**: Rejecting H₀ when it's true (false positive)
    - **Type II Error**: Failing to reject H₀ when it's false (false negative)

---

## 📚 Resources

### Recommended Books
- *Statistical Inference* by Casella and Berger
- *Probability and Statistics* by DeGroot and Schervish
- *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman
- *Practical Statistics for Data Scientists* by Bruce and Bruce

### Python Libraries
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and statistics
- **Pandas**: Data manipulation
- **Statsmodels**: Statistical modeling
- **Seaborn**: Statistical visualization

### Online Resources
- [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
- [Seeing Theory: Visual Introduction to Probability and Statistics](https://seeing-theory.brown.edu/)

---

## 💡 Important Tips

!!! warning "Common Pitfalls"
    - Don't confuse correlation with causation
    - Always check assumptions before applying tests
    - Consider practical significance, not just statistical significance
    - Be aware of multiple testing problems
    - Understand the context of your data

---

## 📝 Notes Section

### Week 1-2: Probability Theory
*Add your notes here*

### Week 3-4: Statistical Distributions
*Add your notes here*

### Week 5-6: Hypothesis Testing and Inference
*Add your notes here*

---

## 🎓 Assignments and Projects

### Assignment 1: Probability and Distributions
*Details to be added*

### Assignment 2: Hypothesis Testing
*Details to be added*

### Project: Statistical Analysis of Real-World Dataset
*Details to be added*

---

## 📊 Progress Tracker

- [ ] Module 1: Probability Theory
- [ ] Module 2: Probability Distributions
- [ ] Module 3: Statistical Inference
- [ ] Module 4: Regression and Correlation
- [ ] Module 5: Advanced Topics
- [ ] Assignments Completed
- [ ] Final Project

---

*Last Updated: October 2025*
