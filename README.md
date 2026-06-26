# 🎩 VoodooNet: The Magic Hat of Neural Networks

**VoodooNet** is a non-iterative, zero-train neural architecture that replaces the "thermodynamic cooling" of Gradient Descent with instantaneous **Galactic Expansion**. 

By projecting data into high-dimensional manifolds and solving for weights using the Moore-Penrose pseudoinverse, VoodooNet achieves superior accuracy to standard SGD in a fraction of the time.

## 🚀 Key Results

| Dataset | Method | Hidden ($d$) | Accuracy | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | VoodooNet | 4,000 | **98.10%** | < 1s |
| **Fashion-MNIST** | VoodooNet | 4,000 | **86.63%** | ~94s |
| **Fashion-MNIST** | SGD (10 Epochs) | 64 | 84.41% | ~9s |

*Note: VoodooNet hits "Hyper-Galactic" accuracy on Fashion-MNIST without a single step of backpropagation.*

## 🌌 How it Works: Galactic Expansion
Unlike traditional networks that "learn" features slowly, VoodooNet treats the hidden layer as a **Magic Hat**. 
1. **The Expansion:** Inputs are projected into a massive hidden space ($d=500$ to $4000$) using high-entropy random weights.
2. **The Discovery:** We skip training. Instead, we use a closed-form analytic solution (Pseudoinverse) to discover the output weights in a single matrix operation.
3. **The Scaling:** Accuracy scales near-logarithmically with dimensionality ($Accuracy \propto \log(d)$).

## 🛠️ Installation & Usage
```bash
git clone https://github.com/Shark-y/VoodooNet
cd VoodooNet
pip install -r requirements.txt
python mnist.py
python mnist-fashion.py
