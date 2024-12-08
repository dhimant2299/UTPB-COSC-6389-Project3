# 📷 CNN Visualization for Glasses Classification

### 🧠 A Convolutional Neural Network (CNN) project to classify images of glasses vs. no-glasses with a real-time visualization interface created without using any libraries related to neural networks.

---

## ✨ Features

### 📊 Interactive GUI
A Tkinter-based GUI that showcases:

- **The CNN Structure**:
  - Input, Convolutional, Pooling, and Fully Connected layers.
- **Real-Time Training Metrics**:
  - **Loss** and **Accuracy** graphs that update dynamically.
- **Dynamic Weight Visualization**:
  - Connections are color-coded to represent weight strength after each epoch.

---

## 🛠 Comprehensive CNN Architecture

1. **Convolutional Layer**: Extracts spatial features from the input using filters.  
   **Formula**:  
   \[
   Y[i, j] = \sum_{k=0}^{K} \sum_{l=0}^{L} X[i+k, j+l] \cdot W[k, l] + b
   \]
   Where:
   - \( Y[i, j] \): Output feature map.
   - \( X[i, j] \): Input image/window.
   - \( W[k, l] \): Filter weights.
   - \( b \): Bias term.

2. **MaxPooling Layer**: Reduces the spatial dimensions while retaining the most important features.  
   **Formula**:  
   \[
   Y[i, j] = \max \left( X[i:i+P, j:j+P] \right)
   \]
   Where \( P \) is the pooling window size.

3. **Fully Connected Layer**: Maps high-level features to class probabilities.  
   **Formula**:  
   \[
   Z = W \cdot X + b
   \]
   Followed by **Softmax Activation**:
   \[
   \text{Softmax}(Z_i) = \frac{e^{Z_i}}{\sum_{j} e^{Z_j}}
   \]

---

## 📂 Dataset
A collection of labeled images, classified into two categories:
- **Glasses**
- **No-Glasses**

---

## 📌 Customizable
- **Hyperparameters**:
  - Learning rate
  - Number of epochs
- **Network Layers**:
  - Easily modify the number of filters, pooling size, and fully connected neurons.

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- Required libraries: `numpy`, `scipy`, `pillow`, `matplotlib`, `tkinter`, `scikit-learn`

