# Mathematical Foundations for Machine Learning

**Course Code**: S1-25_AIMLCZC416  
**Semester**: 1 (2025-26)  
**Category**: Core Course

---

## 📋 Course Overview

This course provides the essential mathematical foundations required for understanding and implementing machine learning algorithms. It covers linear algebra, calculus, and optimization theory - the three pillars of modern ML.

## 🎯 Learning Objectives

By the end of this course, you will be able to:

- ✅ Understand and apply linear algebra concepts in ML contexts
- ✅ Master multivariable calculus and gradient computations
- ✅ Apply optimization techniques for model training
- ✅ Understand the mathematical basis of ML algorithms
- ✅ Perform matrix operations and transformations

---

## 📚 Course Content

### Module 1: Linear Algebra

#### 1.1 Vectors and Vector Spaces
- Vector operations and properties
- Vector spaces and subspaces
- Linear independence and basis
- Inner products and norms

!!! note "ML Application"
    Vectors represent data points, features, and model parameters in ML.

#### 1.2 Matrices and Matrix Operations
- Matrix multiplication and properties
- Transpose and inverse
- Determinants and rank
- Special matrices (identity, diagonal, symmetric)

#### 1.3 Eigenvalues and Eigenvectors
- Computing eigenvalues and eigenvectors
- Diagonalization
- Spectral theorem
- Applications in PCA and dimensionality reduction

```python
import numpy as np

# Example: Computing eigenvalues and eigenvectors
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

#### 1.4 Matrix Decompositions
- Singular Value Decomposition (SVD)
- QR decomposition
- LU decomposition
- Applications in ML

### Module 2: Multivariable Calculus

#### 2.1 Partial Derivatives
- Definition and computation
- Gradient vector
- Directional derivatives
- Chain rule

!!! tip "Important for ML"
    Gradients are fundamental to training neural networks through backpropagation.

#### 2.2 Optimization Basics
- Local and global minima/maxima
- Critical points
- Second derivative test
- Hessian matrix

#### 2.3 Taylor Series and Approximations
- First-order approximations
- Second-order approximations
- Applications in optimization

### Module 3: Optimization Theory

#### 3.1 Convex Optimization
- Convex sets and functions
- Convex optimization problems
- KKT conditions
- Duality

#### 3.2 Gradient Descent Methods
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Convergence analysis

```python
# Example: Simple gradient descent implementation
def gradient_descent(f, df, x0, learning_rate=0.01, iterations=1000):
    """
    f: objective function
    df: gradient of f
    x0: initial point
    """
    x = x0
    history = [x]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example usage
f = lambda x: x**2
df = lambda x: 2*x
minimum, path = gradient_descent(f, df, x0=10.0)
print(f"Minimum found at: {minimum}")
```

#### 3.3 Advanced Optimization
- Momentum methods
- AdaGrad, RMSprop, Adam
- Newton's method
- Conjugate gradient method

### Module 4: Probability and Statistics Basics

#### 4.1 Probability Fundamentals
- Random variables
- Probability distributions
- Expectation and variance
- Common distributions (Gaussian, Bernoulli, etc.)

#### 4.2 Linear Regression from a Mathematical Perspective
- Least squares method
- Normal equations
- Geometric interpretation
- Regularization (Ridge, Lasso)

---

## 🛠️ Practical Applications

### Application 1: Principal Component Analysis (PCA)

PCA uses eigendecomposition to find principal components:

$$\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$$

Where $\mathbf{C}$ is the covariance matrix.

```python
from sklearn.decomposition import PCA
import numpy as np

# Sample data
X = np.random.randn(100, 5)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

### Application 2: Linear Regression

Solving using normal equations:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Application 3: Gradient-Based Optimization

All neural network training relies on gradient descent and its variants.

---

## 📖 Key Formulas

!!! abstract "Essential Formulas"
    
    **Gradient**: $\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$
    
    **Gradient Descent Update**: $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)$
    
    **Matrix-Vector Product**: $\mathbf{Ax} = \sum_{j=1}^{n} a_{ij}x_j$
    
    **Eigenvalue Equation**: $\mathbf{Av} = \lambda\mathbf{v}$

---

## 📚 Resources

### Recommended Books
- *Linear Algebra and Its Applications* by Gilbert Strang
- *Matrix Computations* by Gene H. Golub
- *Numerical Optimization* by Nocedal and Wright
- *Mathematics for Machine Learning* by Deisenroth, Faisal, and Ong

### Online Resources
- [MIT OCW: Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Khan Academy: Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)

### Tools and Libraries
- NumPy for numerical computing
- SciPy for scientific computing
- SymPy for symbolic mathematics
- MATLAB/Octave for matrix computations

---

## 💡 Important Concepts

!!! warning "Key Takeaways"
    - Linear algebra provides the language for ML
    - Calculus enables us to optimize models
    - Understanding gradients is crucial for deep learning
    - Matrix operations are computationally expensive - choose wisely
    - Convex optimization guarantees finding global optima

---

## 📝 Notes Section

### Week 1-2: Linear Algebra Foundations
*Add your notes here*

### Week 3-4: Calculus and Gradients
*Add your notes here*

### Week 5-6: Optimization Methods
*Add your notes here*

---

## 🎓 Assignments and Projects

### Assignment 1: Matrix Operations
*Details to be added*

### Assignment 2: Gradient Descent Implementation
*Details to be added*

### Project: PCA from Scratch
*Details to be added*

---

## 📊 Progress Tracker

- [ ] Module 1: Linear Algebra
- [ ] Module 2: Multivariable Calculus
- [ ] Module 3: Optimization Theory
- [ ] Module 4: Probability Basics
- [ ] Assignments Completed
- [ ] Final Project

---

*Last Updated: October 2025*
