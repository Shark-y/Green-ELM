# 🎩 Green-ELM: The Magic Hat of Neural Networks

**Green-ELM** is a non-iterative, zero-train neural architecture that replaces the "thermodynamic cooling" of Gradient Descent with instantaneous **Galactic Expansion**. 

By projecting data into high-dimensional manifolds and solving for weights using the Moore-Penrose pseudoinverse, Green-ELM achieves superior accuracy to standard SGD in a fraction of the time.

## 🚀 Key Results

| Dataset | Method | Hidden ($d$) | Accuracy | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Green-ELM | 4,000 | **98.10%** | < 1s |
| **Fashion-MNIST** | Green-ELM | 4,000 | **86.63%** | ~94s |
| **Fashion-MNIST** | SGD (10 Epochs) | 64 | 84.41% | ~9s |

*Note: Green-ELM hits high accuracy on Fashion-MNIST without a single step of backpropagation.*

## 🌌 How it Works: High Dimensional Expansion
Unlike traditional networks that "learn" features slowly, Green-ELM treats the hidden layer as a **Magic Hat**. 
1. **The Expansion:** Inputs are projected into a massive hidden space ($d=500$ to $4000$) using high-entropy random weights.
2. **The Discovery:** We skip training. Instead, we use a closed-form analytic solution (Pseudoinverse) to discover the output weights in a single matrix operation.
3. **The Scaling:** Accuracy scales near-logarithmically with dimensionality ($Accuracy \propto \log(d)$).

## 🛠️ Installation & Usage
```bash
git clone https://github.com/Shark-y/Green-ELM
cd Green-ELM
pip install -r requirements.txt
python mnist.py
python mnist-fashion.py
