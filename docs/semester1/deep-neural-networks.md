# Deep Neural Networks

**Course Code**: S1-25_AIMLCZG511  
**Semester**: 1 (2025-26)  
**Category**: Core Course

---

## 📋 Course Overview

This course provides a comprehensive introduction to deep neural networks, covering fundamental architectures, training techniques, and practical implementations. You'll learn how to design, train, and deploy deep learning models for various applications.

## 🎯 Learning Objectives

By the end of this course, you will be able to:

- ✅ Understand the architecture and components of neural networks
- ✅ Implement backpropagation and gradient descent
- ✅ Design and train deep neural networks
- ✅ Apply regularization and optimization techniques
- ✅ Work with popular deep learning frameworks
- ✅ Solve real-world problems using deep learning

---

## 📚 Course Content

### Module 1: Neural Network Fundamentals

#### 1.1 Perceptron and Multilayer Perceptrons
- Single neuron model
- Activation functions
- Perceptron learning algorithm
- Multi-layer perceptrons (MLPs)

!!! info "Historical Context"
    The perceptron, introduced in 1958, was the first algorithm for supervised learning of binary classifiers.

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
    
    def predict(self, x):
        # Add bias term
        x = np.insert(x, 0, 1)
        # Compute weighted sum
        z = np.dot(self.weights, x)
        # Apply activation (step function)
        return 1 if z > 0 else 0
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                # Update weights
                xi_with_bias = np.insert(xi, 0, 1)
                self.weights += self.learning_rate * (target - prediction) * xi_with_bias
```

#### 1.2 Activation Functions
- Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- ReLU: $\text{ReLU}(x) = \max(0, x)$
- Leaky ReLU, ELU, Swish
- Softmax for multi-class classification

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Visualization
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
```

#### 1.3 Forward Propagation
- Layer-by-layer computation
- Matrix operations in neural networks
- Computational graphs

### Module 2: Backpropagation and Training

#### 2.1 Loss Functions
- Mean Squared Error (MSE) for regression
- Binary Cross-Entropy for binary classification
- Categorical Cross-Entropy for multi-class classification
- Custom loss functions

!!! note "Loss Function Selection"
    The choice of loss function depends on your problem type:
    - Regression → MSE, MAE
    - Binary Classification → Binary Cross-Entropy
    - Multi-class Classification → Categorical Cross-Entropy

#### 2.2 Backpropagation Algorithm
- Chain rule in neural networks
- Gradient computation
- Weight updates
- Computational efficiency

**Mathematical Formulation**:

For a layer $l$ with activation $a^{(l)}$ and weights $W^{(l)}$:

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)})$$

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

```python
def backward_propagation(X, y, parameters, cache):
    """
    Implement backward propagation for a neural network
    """
    m = X.shape[1]
    W1, W2 = parameters['W1'], parameters['W2']
    A1, A2 = cache['A1'], cache['A2']
    
    # Backward propagation
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # tanh derivative
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients
```

#### 2.3 Optimization Algorithms
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent
- Momentum
- RMSprop
- Adam optimizer

### Module 3: Deep Learning Architectures

#### 3.1 Convolutional Neural Networks (CNNs)
- Convolutional layers
- Pooling layers
- Famous architectures: LeNet, AlexNet, VGG, ResNet
- Applications in computer vision

#### 3.2 Recurrent Neural Networks (RNNs)
- Vanilla RNNs
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Applications in sequence modeling

#### 3.3 Advanced Architectures
- Autoencoders
- Generative Adversarial Networks (GANs)
- Transformer architecture
- Attention mechanisms

### Module 4: Regularization and Best Practices

#### 4.1 Regularization Techniques
- L1 and L2 regularization
- Dropout
- Batch Normalization
- Data augmentation
- Early stopping

```python
import tensorflow as tf
from tensorflow import keras

# Example: Building a regularized neural network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', 
                      kernel_regularizer=keras.regularizers.l2(0.001),
                      input_shape=(input_dim,)),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

#### 4.2 Hyperparameter Tuning
- Learning rate selection
- Batch size effects
- Network depth and width
- Grid search and random search
- Bayesian optimization

#### 4.3 Training Best Practices
- Weight initialization strategies
- Learning rate scheduling
- Gradient clipping
- Transfer learning
- Model checkpointing

---

## 🛠️ Practical Implementation

### Example 1: Simple Neural Network with NumPy

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = {}
        self.initialize_parameters()
    
    def initialize_parameters(self):
        for i in range(1, len(self.layers)):
            self.parameters[f'W{i}'] = np.random.randn(
                self.layers[i], self.layers[i-1]) * 0.01
            self.parameters[f'b{i}'] = np.zeros((self.layers[i], 1))
    
    def forward(self, X):
        A = X
        caches = {'A0': X}
        
        for i in range(1, len(self.layers)):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            Z = np.dot(W, A) + b
            A = self.relu(Z) if i < len(self.layers) - 1 else self.sigmoid(Z)
            caches[f'Z{i}'] = Z
            caches[f'A{i}'] = A
        
        return A, caches
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def compute_loss(self, AL, y):
        m = y.shape[1]
        loss = -np.sum(y * np.log(AL) + (1-y) * np.log(1-AL)) / m
        return loss
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            AL, caches = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(AL, y)
            losses.append(loss)
            
            # Backward propagation (simplified)
            # ... implement gradients ...
            
            # Update parameters
            # ... implement updates ...
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# Usage
nn = NeuralNetwork([784, 128, 64, 10])  # Input: 784, Hidden: 128, 64, Output: 10
```

### Example 2: Image Classification with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training loop
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
```

---

## 📖 Key Concepts

!!! abstract "Essential Concepts"
    - **Universal Approximation Theorem**: Neural networks can approximate any continuous function
    - **Vanishing Gradients**: Deep networks can suffer from gradients becoming too small
    - **Exploding Gradients**: Gradients can become too large, causing instability
    - **Overfitting**: Model performs well on training but poorly on test data
    - **Transfer Learning**: Using pre-trained models for new tasks

---

## 📚 Resources

### Frameworks and Libraries
- **TensorFlow/Keras**: Industry-standard deep learning framework
- **PyTorch**: Research-friendly framework with dynamic graphs
- **JAX**: High-performance numerical computing
- **MXNet**: Scalable deep learning framework

### Recommended Books
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Neural Networks and Deep Learning* by Michael Nielsen
- *Hands-On Machine Learning* by Aurélien Géron
- *Deep Learning with Python* by François Chollet

### Online Courses
- [Fast.ai: Practical Deep Learning](https://www.fast.ai/)
- [Stanford CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [deeplearning.ai Specialization](https://www.deeplearning.ai/)

### Tools
- Google Colab for GPU access
- Weights & Biases for experiment tracking
- TensorBoard for visualization
- Netron for model visualization

---

## 💡 Important Tips

!!! tip "Best Practices"
    - Start with simple models and gradually increase complexity
    - Always validate on a separate dataset
    - Use appropriate activation functions (ReLU for hidden layers)
    - Apply regularization to prevent overfitting
    - Monitor both training and validation metrics
    - Use transfer learning when possible
    - Save model checkpoints regularly

---

## 📝 Notes Section

### Week 1-2: Neural Network Basics
*Add your notes here*

### Week 3-4: Backpropagation and Optimization
*Add your notes here*

### Week 5-6: CNN and RNN Architectures
*Add your notes here*

---

## 🎓 Assignments and Projects

### Assignment 1: Implement Backpropagation from Scratch
*Details to be added*

### Assignment 2: Image Classification with CNN
*Details to be added*

### Project: Build and Train a Deep Learning Model
*Details to be added*

---

## 📊 Progress Tracker

- [ ] Module 1: Neural Network Fundamentals
- [ ] Module 2: Backpropagation and Training
- [ ] Module 3: Deep Learning Architectures
- [ ] Module 4: Regularization and Best Practices
- [ ] Assignments Completed
- [ ] Final Project

---

*Last Updated: October 2025*
