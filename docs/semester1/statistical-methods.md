# Introduction to Statistical Methods

**Course Code**: S1-25_AIMLCZC418  
**Semester**: 1 (2025-26)  
**Category**: Core Course

---

## � Course Syllabus

### Module 1: Probability Theory

- Fundamentals of Probability
- Random Variables
- Expectation and Variance
- Bayes' Theorem

### Module 2: Probability Distributions

- Discrete Distributions (Bernoulli, Binomial, Poisson)
- Continuous Distributions (Normal, Exponential, Chi-square)
- Multivariate Distributions

### Module 3: Statistical Inference

- Point Estimation (MLE, Method of Moments)
- Interval Estimation
- Hypothesis Testing

### Module 4: Regression and Correlation

- Correlation Analysis
- Simple Linear Regression
- Multiple Linear Regression

### Module 5: Advanced Topics

- ANOVA
- Non-parametric Methods
- Chi-Square Tests

---

## 📝 Notes Section
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

## � Notes Section

### Week 1-2: Probability Theory
*Add your notes here*

### Week 3-4: Statistical Distributions
*Add your notes here*

### Week 5-6: Hypothesis Testing and Inference
*Add your notes here*

---

## 📄 Exam Papers

### Previous Year Questions
*To be added*

### Sample Questions
*To be added*

### Important Topics for Exam
- Bayes' Theorem
- Hypothesis Testing
- Linear Regression
- ANOVA
- Statistical Distributions

---

## 🎓 Assignments and Projects

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
