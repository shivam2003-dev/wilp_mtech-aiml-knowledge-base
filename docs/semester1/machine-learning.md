# Machine Learning

**Course Code**: S1-25_AIMLCZG565  
**Semester**: 1 (2025-26)  
**Category**: Core Course

---

## 📋 Course Overview

This comprehensive course covers the fundamental concepts, algorithms, and techniques in machine learning. You'll learn both supervised and unsupervised learning methods, model evaluation, and practical implementation strategies for real-world problems.

## 🎯 Learning Objectives

By the end of this course, you will be able to:

- ✅ Understand core ML concepts and terminology
- ✅ Implement supervised learning algorithms
- ✅ Apply unsupervised learning techniques
- ✅ Evaluate and compare ML models
- ✅ Handle real-world data challenges
- ✅ Select appropriate algorithms for different problems
- ✅ Build end-to-end ML pipelines

---

## 📚 Course Content

### Module 1: Introduction to Machine Learning

#### 1.1 What is Machine Learning?
- Definition and scope
- Types of machine learning
  - Supervised learning
  - Unsupervised learning
  - Semi-supervised learning
  - Reinforcement learning
- Applications and use cases
- ML workflow and lifecycle

!!! info "ML Definition"
    Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel (1959)

```python
# ML Problem Framework
"""
1. Define the problem
2. Gather data
3. Prepare data
4. Choose model
5. Train model
6. Evaluate model
7. Tune hyperparameters
8. Deploy model
"""
```

#### 1.2 Mathematical Foundations Review
- Linear algebra essentials
- Probability and statistics
- Calculus for optimization
- Information theory basics

#### 1.3 Python for Machine Learning
- NumPy for numerical computing
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Scikit-learn for ML

```python
# Essential imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
```

### Module 2: Supervised Learning - Regression

#### 2.1 Linear Regression
- Simple linear regression
- Multiple linear regression
- Polynomial regression
- Regularization (Ridge, Lasso, ElasticNet)

**Linear Regression Model**:
$$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = \mathbf{w}^T\mathbf{x}$$

**Cost Function (MSE)**:
$$J(\mathbf{w}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Example: Linear Regression
# Generate sample data
X = np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualization
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
```

#### 2.2 Regularization Techniques
- L1 Regularization (Lasso): $J(\mathbf{w}) = MSE + \alpha\sum_{i=1}^{n}|w_i|$
- L2 Regularization (Ridge): $J(\mathbf{w}) = MSE + \alpha\sum_{i=1}^{n}w_i^2$
- ElasticNet: Combination of L1 and L2

```python
# Comparison of regularization techniques
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Generate data with many features
X = np.random.randn(100, 20)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} R² Score: {score:.4f}")
```

#### 2.3 Non-linear Regression
- Polynomial regression
- Decision tree regression
- Support Vector Regression (SVR)
- Ensemble methods for regression

### Module 3: Supervised Learning - Classification

#### 3.1 Logistic Regression
- Binary classification
- Sigmoid function: $\sigma(z) = \frac{1}{1+e^{-z}}$
- Log loss / Cross-entropy
- Multinomial logistic regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Binary classification (setosa vs others)
y_binary = (y == 0).astype(int)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

#### 3.2 Decision Trees
- Tree construction algorithms (ID3, C4.5, CART)
- Splitting criteria (Gini impurity, entropy)
- Pruning techniques
- Advantages and limitations

**Gini Impurity**:
$$G = 1 - \sum_{i=1}^{C}p_i^2$$

**Entropy**:
$$H = -\sum_{i=1}^{C}p_i\log_2(p_i)$$

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train decision tree
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names)
plt.show()

# Evaluate
y_pred_dt = dt_classifier.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
```

#### 3.3 Support Vector Machines (SVM)
- Linear SVM
- Kernel trick (polynomial, RBF, sigmoid)
- Soft margin and C parameter
- Multi-class SVM

```python
from sklearn.svm import SVC

# Train SVM with different kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    accuracy = svm.score(X_test_scaled, y_test)
    print(f"SVM ({kernel} kernel) Accuracy: {accuracy:.4f}")
```

#### 3.4 K-Nearest Neighbors (KNN)
- Distance metrics (Euclidean, Manhattan, Minkowski)
- Choosing K
- Weighted KNN
- Curse of dimensionality

```python
from sklearn.neighbors import KNeighborsClassifier

# Find optimal K
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# Plot K vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN: K vs Accuracy')
plt.grid(True)
plt.show()

optimal_k = k_range[np.argmax(scores)]
print(f"Optimal K: {optimal_k}")
print(f"Best Accuracy: {max(scores):.4f}")
```

#### 3.5 Naive Bayes
- Bayes theorem in classification
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

### Module 4: Ensemble Methods

#### 4.1 Bagging
- Bootstrap Aggregating
- Random Forest
- Extra Trees

```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_classifier.fit(X_train, y_train)

# Feature importance
importances = rf_classifier.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

print("Random Forest Accuracy:", rf_classifier.score(X_test, y_test))
```

#### 4.2 Boosting
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# AdaBoost
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_classifier.fit(X_train, y_train)
print("AdaBoost Accuracy:", ada_classifier.score(X_test, y_test))

# Gradient Boosting
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_classifier.fit(X_train, y_train)
print("Gradient Boosting Accuracy:", gb_classifier.score(X_test, y_test))
```

#### 4.3 Stacking and Voting
- Voting classifiers (hard and soft)
- Stacking classifiers
- Blending

### Module 5: Unsupervised Learning

#### 5.1 Clustering
- K-Means clustering
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture Models (GMM)

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.tight_layout()
plt.show()

# Elbow method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
```

#### 5.2 Dimensionality Reduction
- Principal Component Analysis (PCA)
- t-SNE
- UMAP
- Autoencoders

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load high-dimensional data
from sklearn.datasets import load_digits
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_digits)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
axes[0].set_title('PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()

print(f"Explained variance ratio (PCA): {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
```

#### 5.3 Association Rule Learning
- Apriori algorithm
- Market basket analysis
- FP-Growth

### Module 6: Model Evaluation and Selection

#### 6.1 Performance Metrics
- Classification metrics
  - Accuracy, Precision, Recall, F1-Score
  - ROC curve and AUC
  - Confusion matrix
- Regression metrics
  - MSE, RMSE, MAE, R²

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Example: ROC Curve
log_reg.fit(X_train_scaled, y_train)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

#### 6.2 Cross-Validation
- K-fold cross-validation
- Stratified K-fold
- Leave-one-out cross-validation
- Time series cross-validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# K-Fold Cross-Validation
cv_scores = cross_val_score(
    LogisticRegression(),
    X_train_scaled,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    LogisticRegression(),
    X_train_scaled,
    y_train,
    cv=skf
)
print(f"Stratified K-Fold Mean Score: {scores.mean():.4f}")
```

#### 6.3 Hyperparameter Tuning
- Grid Search
- Random Search
- Bayesian Optimization

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search Example
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
print("Test set score:", grid_search.score(X_test_scaled, y_test))
```

### Module 7: Feature Engineering

#### 7.1 Feature Scaling
- Standardization
- Normalization
- Robust scaling

#### 7.2 Feature Selection
- Filter methods (correlation, chi-square, mutual information)
- Wrapper methods (forward/backward selection, RFE)
- Embedded methods (Lasso, tree-based importance)

```python
from sklearn.feature_selection import SelectKBest, chi2, RFE

# SelectKBest
selector = SelectKBest(chi2, k=2)
X_selected = selector.fit_transform(X_train, y_train)

print("Selected features:", iris.feature_names[selector.get_support()])

# Recursive Feature Elimination
rfe = RFE(LogisticRegression(), n_features_to_select=2)
rfe.fit(X_train_scaled, y_train)

print("RFE selected features:", iris.feature_names[rfe.support_])
```

#### 7.3 Feature Creation
- Polynomial features
- Interaction features
- Domain-specific features

---

## 📖 Key Concepts

!!! abstract "ML Fundamentals"
    - **Bias-Variance Tradeoff**: Balance between underfitting and overfitting
    - **No Free Lunch Theorem**: No single algorithm works best for all problems
    - **Curse of Dimensionality**: Performance degrades with too many features
    - **Regularization**: Prevent overfitting by penalizing complexity
    - **Cross-Validation**: Reliable way to estimate model performance

---

## 🛠️ Complete ML Pipeline

```python
# Complete ML Pipeline Example
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define pipeline
numeric_features = ['feature1', 'feature2']
categorical_features = ['category1']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit and predict
# full_pipeline.fit(X_train, y_train)
# predictions = full_pipeline.predict(X_test)
```

---

## 📚 Resources

### Essential Libraries
- **Scikit-learn**: General ML library
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Categorical boosting
- **MLxtend**: ML extensions

### Recommended Books
- *Hands-On Machine Learning* by Aurélien Géron
- *Pattern Recognition and Machine Learning* by Christopher Bishop
- *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman
- *Machine Learning: A Probabilistic Perspective* by Kevin Murphy

### Online Courses
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai: Introduction to Machine Learning](https://course18.fast.ai/ml)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

---

## 💡 Best Practices

!!! tip "ML Best Practices"
    1. Always split data before any processing
    2. Scale features for distance-based algorithms
    3. Use cross-validation for model selection
    4. Start simple, then increase complexity
    5. Monitor both training and validation metrics
    6. Handle imbalanced data appropriately
    7. Document your experiments
    8. Version control your code and data

---

## 📝 Notes Section

### Week 1-2: ML Fundamentals and Regression
*Add your notes here*

### Week 3-4: Classification Algorithms
*Add your notes here*

### Week 5-6: Ensemble Methods and Unsupervised Learning
*Add your notes here*

---

## 🎓 Assignments and Projects

### Assignment 1: Regression Analysis
*Details to be added*

### Assignment 2: Classification Challenge
*Details to be added*

### Project: End-to-End ML Pipeline
*Details to be added*

---

## 📊 Progress Tracker

- [ ] Module 1: Introduction to ML
- [ ] Module 2: Supervised Learning - Regression
- [ ] Module 3: Supervised Learning - Classification
- [ ] Module 4: Ensemble Methods
- [ ] Module 5: Unsupervised Learning
- [ ] Module 6: Model Evaluation
- [ ] Module 7: Feature Engineering
- [ ] Assignments Completed
- [ ] Final Project

---

*Last Updated: October 2025*
