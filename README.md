# DL Mastery: Deep Learning from Zero to Hero

**A free, comprehensive, hands-on deep learning course featuring an interactive browser playground and 17 Google Colab notebooks.**

[![Playground](https://img.shields.io/badge/Playground-Live-brightgreen)](https://dlmastery.github.io/neural-network-playground/)
[![License](https://img.shields.io/badge/License-Open_Source-blue)]()
[![Colab](https://img.shields.io/badge/Run_on-Google_Colab-F9AB00)]()

---

## Video Links (18 Videos -> 1 Site + 17 Notebooks):

- [Resource 0: Neural Network Playground](#resource-0-neural-network-playground)
- [Resource 1: NumPy Foundations](#resource-1-numpy-foundations)
- [Resource 2: PyTorch Tensors from Zero to Hero](#resource-2-pytorch-tensors-from-zero-to-hero)
- [Resource 3: TensorFlow Tensor Operations](#resource-3-tensorflow-tensor-operations)
- [Resource 4: JAX Deep Learning Tutorial](#resource-4-jax-deep-learning-tutorial)
- [Resource 5: Calculus for Deep Learning](#resource-5-calculus-for-deep-learning)
- [Resource 6: Probability Fundamentals for Deep Learning](#resource-6-probability-fundamentals-for-deep-learning)
- [Resource 7: Probability for Deep Learning](#resource-7-probability-for-deep-learning)
- [Resource 8: Linear Algebra for Deep Learning](#resource-8-linear-algebra-for-deep-learning)
- [Resource 9: Neural Networks from Scratch](#resource-9-neural-networks-from-scratch)
- [Resource 10: Why Deep Learning Works — Geometric Intuition](#resource-10-why-deep-learning-works--geometric-intuition)
- [Resource 11: PyTorch Neural Networks Tutorial](#resource-11-pytorch-neural-networks-tutorial)
- [Resource 12: PyTorch Advanced Tutorial](#resource-12-pytorch-advanced-tutorial)
- [Resource 13: Keras/TensorFlow Neural Networks Tutorial](#resource-13-kerastensorflow-neural-networks-tutorial)
- [Resource 14: Keras/TensorFlow Advanced Tutorial](#resource-14-kerastensorflow-advanced-tutorial)
- [Resource 15: NumPy Foundations for Deep Learning](#resource-15-numpy-foundations-for-deep-learning)
- [Resource 16: JAX Deep Learning Tutorial (Comprehensive)](#resource-16-jax-deep-learning-tutorial-comprehensive)
- [Resource 17: JAX Neural Networks Tutorial](#resource-17-jax-neural-networks-tutorial)

---
## Recommended Learning Path

```
PHASE 1: EXPLORE (30 minutes)
└── Resource 0: Playground
    ├── Train an MLP on the XOR problem
    ├── Train a CNN on MNIST, explore the Explainer
    ├── Generate text with the Transformer
    ├── Classify graph nodes with GNN
    ├── Generate digits with Diffusion
    └── Browse Design Spaces reference cards

PHASE 2: FOUNDATIONS (2–3 hours)
├── Resource 1: NumPy Foundations
├── Resource 15: NumPy Foundations for Deep Learning
└── Pick your framework:
    ├── Resource 2: PyTorch Tensors (recommended for research)
    ├── Resource 3: TensorFlow Tensors (recommended for production)
    └── Resource 4: JAX Tutorial (recommended for Google/TPU work)

PHASE 3: MATHEMATICS (2–3 hours)
├── Resource 8: Linear Algebra for Deep Learning
├── Resource 5: Calculus for Deep Learning
├── Resource 6: Probability Fundamentals (intuition-first)
└── Resource 7: Probability for Deep Learning (rigorous)

PHASE 4: BUILD FROM SCRATCH (1–2 hours)
├── Resource 9: Neural Networks from Scratch (pure NumPy)
└── Resource 10: Why Deep Learning Works (geometric intuition)

PHASE 5: MASTER YOUR FRAMEWORK (2–3 hours)
├── PyTorch path:    Resource 11 → Resource 12
├── TF/Keras path:   Resource 13 → Resource 14
└── JAX path:        Resource 16 → Resource 17
```

---
## Resource 0: Neural Network Playground

**URL:** [https://dlmastery.github.io/neural-network-playground/](https://dlmastery.github.io/neural-network-playground/)

**What it is:** A fully interactive browser-based environment for training and visualizing 6 neural network architectures. Everything runs client-side — no server, no account, no data leaves your browser.

### Model 1: Neural Net (MLP)

Binary classification on 2D data with a fully customizable architecture.

**Features:**
- Input features: x₁, x₂, plus toggleable engineered features (x₁², x₂², x₁x₂, sin, cos)
- Customizable hidden layers: add/remove layers, add/remove neurons per layer
- Activation functions: ReLU, Sigmoid, Tanh, Leaky ReLU
- Adjustable learning rate (0.00001 to 1.0) and regularization (0.0 to 0.1)
- Real-time decision boundary visualization during training
- Live epoch counter, loss, and accuracy display
- Parameter count and layer count display

**Example Presets:**
- **XOR Problem:** 2 inputs → 4 neurons → 1 output, demonstrates why hidden layers are needed
- **Spiral Classification:** Interleaved spirals requiring deeper networks and feature engineering

**What you learn:** How network architecture (depth, width), activation functions, learning rate, and regularization affect learning. Visual intuition for decision boundaries.

---

### Model 2: CNN

Image classification on real datasets with layer-by-layer architecture building.

**Features:**
- Datasets: MNIST Digits (28×28 grayscale, 10 classes), Fashion-MNIST (28×28, 10 classes), CIFAR-10 (32×32 color, 10 classes)
- Layer palette: Conv2D, MaxPool, Dense, Dropout — drag and arrange
- Training controls: learning rate, optimizer (Adam/SGD/RMSprop), batch size (16/32/64/128)
- Backend selection: WebGL (GPU), WebGPU, CPU (WASM)
- Real-time training metrics: epoch, loss, train accuracy, test accuracy

**CNN Explainer (8 interactive tabs):**

| Tab | What It Shows |
|-----|--------------|
| **Overview** | High-level CNN pipeline from input to prediction |
| **Draw** | Draw your own digit with mouse/touch, see live classification with confidence scores and processed 28×28 input |
| **Filters** | Learned convolution kernels (weights) for each Conv2D layer, blue/red coloring for positive/negative values |
| **Activations** | Feature maps showing what each filter detects in the input, bright areas = strong activation |
| **Flatten** | Visual before/after of 2D feature maps → 1D vector conversion, explains why flatten is needed before Dense layers |
| **Softmax** | Side-by-side comparison of raw logits vs. probabilities after softmax, shows the exponential amplification effect |
| **Saliency** | Heatmap overlay showing which input pixels most influence the prediction, warm colors = high importance |
| **Convolution** | Animated visualization of a filter sliding across the input, showing element-wise products at each position |

**What you learn:** How convolution detects features, how pooling creates translation invariance, how softmax produces probabilities, what the network actually "sees" at each layer.

---

### Model 3: Transformer

Text generation with real language models running in-browser.

**Features:**
- Models: DistilGPT-2 (88MB, 82M parameters, 6 layers) and GPT-2 (167MB, full model)
- Adjustable temperature (0.1–2.0) and max generation length
- Token counter and generation status display
- Backend: WebGPU (GPU) or CPU (WASM)
- Architecture info panel: parameter count, layers, hidden size, attention heads, vocab size

**Transformer Explainer (6 interactive tabs):**

| Tab | What It Shows |
|-----|--------------|
| **Overview** | What transformers are, key innovation (attention), applications (GPT, BERT, T5, ViT) |
| **Tokenization** | Live interactive tokenizer — type any text and watch BPE split it into subword tokens with IDs |
| **Embeddings** | Token embeddings (768-dim) + position embeddings = combined embeddings, visual breakdown |
| **Attention** | Self-attention mechanism: Query/Key/Value computation, attention scores, weighted value aggregation, with formula |
| **Multi-Head** | Multiple parallel attention heads, each learning different relationships (syntax, semantics, coreference) |
| **Generation** | Autoregressive token-by-token generation, sampling strategies (greedy, temperature, top-k, nucleus/top-p) |

**Live PicoGPT:** A tiny transformer (2 layers, 4 heads, 64 dim) trained on Shakespeare that you can step through token by token, watching embeddings flow through transformer blocks, attention patterns form, and predictions emerge.

**Attention Visualization:** Select any layer (1–6) and head (1–12) to see the full attention pattern matrix after generating text.

**What you learn:** How tokenization works, how attention enables parallel sequence processing, how multi-head attention captures different types of relationships, how autoregressive generation produces text.

---

### Model 4: GNN (Graph Neural Network)

Node classification on graph-structured data.

**Features:**
- Datasets: Zachary's Karate Club (34 nodes, 78 edges, 2 classes) and Synthetic Graph
- Architectures: GCN (Graph Convolution Network) and GAT (Graph Attention Network)
- Configurable number of layers (1–5)
- Interactive graph visualization with node coloring by predicted class
- Training metrics: epoch, loss, accuracy
- Click-to-inspect node details: connections, embedding values, predicted class

**Message Passing Visualization:**
- Animated step-by-step message passing
- Play, step forward, and reset controls
- Layer-by-layer visualization showing information aggregation
- Watch nodes collect neighbor information, transform it, and pass it forward

**Node Embeddings:** 2D projection of learned node representations, showing how nodes cluster by class as training progresses.

**What you learn:** How graph neural networks propagate information through edges, the difference between GCN (uniform neighbor weighting) and GAT (learned attention-weighted neighbors), how message passing enables nodes to learn from their local structure.

---

### Model 5: Diffusion

Image generation via iterative denoising.

**Features:**
- Generate MNIST digits from random noise in a 4×4 grid
- Class selection: Random or specific digit (0–9)
- Adjustable denoising steps (1–50) and guidance scale (1.0–10.0)
- Backend: WebGPU, WebGL, CPU
- U-Net model info: ~100K parameters, 28×28×1 input, 1000 timesteps

**Training Mode:**
- Train your own diffusion model from scratch in the browser
- Adjustable learning rate and epochs
- Real-time loss curve

**Denoising Timeline:** Click any generated digit to see the complete denoising trajectory — pure noise on the left, clean image on the right, every intermediate step between.

**Diffusion Explainer (7 interactive tabs):**

| Tab | What It Shows |
|-----|--------------|
| **What is Diffusion?** | Core concept: learning to reverse noise addition. Applications: DALL-E 2, Stable Diffusion, Midjourney, Sora |
| **Forward Process** | Interactive noise slider showing gradual image corruption with the mathematical formula q(x_t \| x_{t-1}) |
| **Reverse Process** | Animated denoising with play control, showing the model predicting and subtracting noise step by step |
| **U-Net** | Architecture diagram: encoder (downsample), decoder (upsample), skip connections between corresponding layers |
| **Noise Schedules** | Comparison of linear vs. cosine schedules with visual curves. Linear: simple ramp. Cosine: preserves more signal longer |
| **Sampling** | DDPM vs. DDIM race — DDPM (50 steps, stochastic, higher quality) vs. DDIM (10 steps, deterministic, faster) |
| **Guidance** | Classifier-free guidance slider: low (1–2) = diverse/fuzzy, high (7–10) = sharp/less variety, with formula |

**What you learn:** How noise addition creates a training target, how the U-Net predicts noise to remove, how noise schedules affect quality, the speed-quality tradeoff between DDPM and DDIM, how guidance controls the fidelity-diversity tradeoff.

---

### Model 6: Design Spaces

Architecture reference cards for all model types.

**Contents:**
- One card per architecture (Neural Net, CNN, Transformer, GNN, Diffusion)
- Each card includes:
  - **When to Use:** Data type and problem type recommendations
  - **Key Innovation:** The core idea that makes the architecture work
  - **Typical Size:** Parameter count ranges
  - **Scaling Guide:** How performance changes with scale
  - **Key Papers:** Seminal references

**View modes:** Card view and Table view. Copy config button for quick reference.

**What you learn:** High-level architectural decision-making — which architecture to choose for which problem, trade-offs, and where to read more.

---

### Additional Playground Features

**PyTorch Code Reference:** Accessible from the toolbar. Clean, copyable PyTorch implementations of:
- MLP (nn.Sequential with Linear/activation/Sigmoid)
- CNN (Conv2d, MaxPool2d, Flatten, Linear)
- Transformer (self-attention with Q/K/V and softmax)
- GNN (GCN layer with adjacency matrix message passing)
- Diffusion (forward noising, reverse denoising with U-Net)
- Standard training loop template

**Example Configurations:** Pre-built setups with recommended hyperparameters:
- XOR Problem (Neural Net)
- Spiral Classification (Neural Net)
- MNIST Digit Recognition (CNN)
- Text Generation (Transformer)
- Image Generation (Diffusion)

**Requirements:** Modern browser. No installation, account, or server. 100% client-side.

---

## Resource 1: NumPy Foundations

**File:** `1_Numpy_foundations.ipynb`  
**Duration:** ~25 minutes  
**Framework:** NumPy  
**Prerequisites:** Basic Python

**Description:** Complete introduction to NumPy as the foundation of all deep learning computation.

**Contents:**

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| 1–2 | Array Basics | Creation (zeros, ones, arange, random), indexing, slicing |
| 3 | Operations | Element-wise arithmetic, math functions (exp, log, sin — building blocks of activations) |
| 4 | Broadcasting | Rules (right-to-left comparison), bias addition pattern (z = X @ W + b), normalization |
| 5 | Matrix Multiplication | @ operator, shape rules (inner dims must match), batch computation |

**Deep Learning Connections:**
- Element-wise operations → activation functions (ReLU = `np.maximum(0, x)`)
- Broadcasting → bias addition across a batch
- Matrix multiplication → the layer operation (`output = input @ weights`)

---

## Resource 2: PyTorch Tensors from Zero to Hero

**File:** `2_pytorch_tensors_from_zero_to_hero.ipynb`  
**Duration:** ~30 minutes  
**Framework:** PyTorch  
**Prerequisites:** Resource 1

**Description:** Comprehensive guide to PyTorch tensor operations, from creation to patterns used in production models.

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| 1 | What is a Tensor? | Scalars, vectors, matrices, higher-order tensors, DL data shapes |
| 2 | Creating Tensors | From lists, NumPy, special functions (zeros/randn/eye), weight init patterns |
| 3–4 | Attributes & Indexing | shape/dtype/device, basic/advanced/boolean indexing |
| 5–6 | Operations & Reshaping | Arithmetic, broadcasting, in-place ops, view/reshape/squeeze/unsqueeze/permute |
| 7 | Linear Algebra | matmul (@), dot product, batched operations, decompositions |
| 8 | Einstein Summation | einsum notation, matrix multiply, transpose, batch operations, trace |
| 9–10 | DL Operations & Patterns | Softmax, log-softmax, layer norm, batch norm from raw ops |

**Key takeaway:** PyTorch tensors = NumPy arrays + GPU acceleration + automatic differentiation tracking.

---

## Resource 3: TensorFlow Tensor Operations

**File:** `3_tensorflow_tensor_operations_tutorial.ipynb`  
**Duration:** ~30 minutes  
**Framework:** TensorFlow / Keras 3  
**Prerequisites:** Resource 1

**Description:** Complete TensorFlow tensor operations guide, mirroring the PyTorch notebook for the TF/Keras ecosystem.

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| 1 | Tensor Basics | tf.constant (immutable) vs tf.Variable (mutable), creation, dtypes, attributes |
| 2 | Indexing & Reshaping | Slicing, tf.gather/gather_nd, tf.reshape/squeeze/expand_dims/transpose |
| 3 | Math & Broadcasting | Element-wise ops, broadcasting rules (same as NumPy), strict type promotion |
| 4 | Linear Algebra | tf.linalg.matmul, @ operator |
| 5 | Einstein Summation | tf.einsum (identical notation to PyTorch) |
| 6 | Reduction Operations | tf.reduce_mean/sum/max with axis arguments |
| 7–8 | DL Patterns & Exercises | Softmax, cross-entropy, normalization from scratch + tf.nn equivalents |

**Key differences from PyTorch:** Immutable tensors, tf.Variable for weights, stricter type promotion, tf.gather for advanced indexing.

---

## Resource 4: JAX Deep Learning Tutorial

**File:** `4_jax_deep_learning_tutorial.ipynb`  
**Duration:** ~35 minutes  
**Framework:** JAX  
**Prerequisites:** Resource 1

**Description:** Introduction to JAX — NumPy with GPU/TPU acceleration, automatic differentiation, JIT compilation, and automatic vectorization.

**Contents:**

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| 1 | Introduction | JAX philosophy: pure functions, composable transformations |
| 2 | Tensor Fundamentals | jax.numpy, explicit PRNG keys (split/subkey), immutable arrays (.at[].set()) |
| 3 | Operations | Same as NumPy but on accelerators |
| 4–5 | Linear Algebra & einsum | Standard operations with jnp |
| 6 | Automatic Differentiation | `grad()`, composable: `grad(grad(f))` for second derivatives |
| 7 | JIT Compilation | `jit()` for XLA acceleration, 10–100x speedup, tracing semantics |
| 8 | Vectorization | `vmap()` for automatic batching, per-example gradients |
| 9–11 | Neural Networks | Building NNs from scratch with PyTrees, common DL patterns, practical examples |

**JAX's unique features:** Explicit PRNG, composable transforms (`jit(vmap(grad(f)))`), functional programming paradigm, XLA compilation.

---

## Resource 5: Calculus for Deep Learning

**File:** `5_Calculus_for_deep_learning.ipynb`  
**Duration:** ~25 minutes  
**Framework:** NumPy (for visualization)  
**Prerequisites:** High school algebra

**Description:** All the calculus needed for deep learning: derivatives, chain rule, gradients, gradient descent, and backpropagation. Visualized and coded.

**Contents:**

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| 1 | Derivatives | Rate of change, sensitivity of output to input |
| 2 | Chain Rule | Composing derivatives, mathematical foundation of backpropagation |
| 3 | Partial Derivatives & Gradients | Multi-variable derivatives, gradient vector, direction of steepest ascent |
| 4 | Gradient Descent | Learning algorithm: w_new = w_old - lr × gradient, learning rate effects |
| 5 | Backpropagation | Efficient chain rule: forward pass saves values, backward pass reuses them, O(n) |

**Core Equations:**
- Forward: `z = Wx + b`, `a = activation(z)`
- Loss: `L = loss_fn(prediction, target)`
- Backward: `∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W`
- Update: `W = W - η · ∂L/∂W`

---

## Resource 6: Probability Fundamentals for Deep Learning

**File:** `6_probability_fundamentals_for_deep_learning.ipynb`  
**Duration:** ~30 minutes  
**Framework:** NumPy, Matplotlib  
**Prerequisites:** Basic arithmetic

**Description:** Intuition-first probability course. Every concept follows: story → visual → math → code → deep learning connection.

**Contents:**

| Chapter | Topic | Deep Learning Connection |
|---------|-------|------------------------|
| 0–2 | Probability basics, sample spaces, events, rules | Neural networks output probabilities over classes |
| 3 | Conditional probability | P(class \| input) — the core of classification |
| 4 | Bayes' theorem | Bayesian neural networks, belief updating |
| 5–6 | Random variables, distributions | Bernoulli (sigmoid), Categorical (softmax), Gaussian (weight init) |
| 7 | Expectation & variance | Batch normalization controls variance |
| 8 | Maximum Likelihood Estimation | Training = maximizing likelihood of observed data |
| 9 | Information theory | Cross-entropy loss = negative log-likelihood |
| 10 | Everything together | Complete classification pipeline using all concepts |

**Key insight:** Minimizing cross-entropy loss IS the same as maximizing likelihood — same equation, negative sign.

---

## Resource 7: Probability for Deep Learning

**File:** `7_probability_for_deep_learning.ipynb`  
**Duration:** ~35 minutes  
**Framework:** NumPy, SciPy, Matplotlib  
**Prerequisites:** Resource 6 or basic probability

**Description:** Rigorous probability theory for deep learning. Formal definitions, proofs, and advanced topics including KL divergence and Monte Carlo methods.

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| 1 | Fundamentals | Kolmogorov axioms, sample spaces, events |
| 2 | Conditional Probability & Bayes | Formal derivation, medical test example (false positive rates) |
| 3 | Random Variables | PMF, PDF, CDF, formal definitions |
| 4 | Expected Value & Variance | E[X], Var(X), moments |
| 5 | Key DL Distributions | Gaussian, Uniform, Bernoulli, Categorical, mixtures — with DL applications |
| 6 | Information Theory | Entropy (H), cross-entropy, KL divergence (VAEs, knowledge distillation) |
| 7 | Maximum Likelihood | Formal MLE derivation, connection to cross-entropy minimization |
| 8 | Sampling & Monte Carlo | Monte Carlo methods, dropout as approximate Bayesian inference |
| 9 | Practical Exercises | Hands-on problems |

**Key insight:** Dropout is a form of approximate Bayesian inference through Monte Carlo sampling of sub-networks.

---

## Resource 8: Linear Algebra for Deep Learning

**File:** `8_linear_algebra_for_deep_learning.ipynb`  
**Duration:** ~25 minutes  
**Framework:** NumPy  
**Prerequisites:** Basic arithmetic

**Description:** The linear algebra you need for deep learning — nothing more, nothing less. Every concept maps to a neural network component.

**Contents:**

| Section | Topic | Neural Network Use |
|---------|-------|-------------------|
| 1–3 | Vectors & matrices | Features/embeddings (vectors), weights/batches (matrices) |
| 4 | Dot products | Single neuron computation: w · x + b |
| 5 | Matrix multiplication | Layer operation: output = input @ weights |
| 5 | Transpose | Attention (Q @ K.T), backpropagation (gradient flow) |
| 6 | Eigenvalues/eigenvectors | PCA, gradient explosion/vanishing analysis |
| 7 | Norms & distances | L2 regularization (weight decay), L1 sparsity |

**Core equation:** `Y = activation(X @ W + b)` — every term is linear algebra.

**Quick reference:** dot product → neuron activation, matmul → layer transformation, transpose → attention + backprop, norms → regularization, eigenvalues → training stability.

---

## Resource 9: Neural Networks from Scratch

**File:** `9_neural_networks_from_scratch.ipynb`  
**Duration:** ~40 minutes  
**Framework:** NumPy only (no deep learning frameworks)  
**Prerequisites:** Resources 1, 5, 8

**Description:** Build a complete neural network from scratch using only NumPy. Full forward propagation, backpropagation, and training.

**Contents:**

| Chapter | Topic | What You Build |
|---------|-------|---------------|
| 1 | The Neuron | Mathematical model: z = w·x + b, a = σ(z) |
| 2 | Activation Functions | Sigmoid, tanh, ReLU, leaky ReLU, softmax — each with its derivative |
| 3 | Loss Functions | MSE, binary cross-entropy, categorical cross-entropy |
| 4 | Dense Layer | Matrix form: Z = X @ W + b, A = activation(Z) |
| 5 | Backpropagation | Chain rule, gradient equations for all layers, weight updates |
| 6 | Complete Network | `NeuralNetwork` class: `add_layer()`, `forward()`, `backward()`, `train()` |
| 7 | Training Demos | XOR (why hidden layers needed), spiral classification, MNIST (>95% accuracy) |

**Key gradient equations:**
- Output: `dZ = predictions - targets` (cross-entropy + softmax)
- Hidden: `dZ = (dA_next @ W_next.T) * activation'(Z)`
- Weights: `dW = X.T @ dZ / batch_size`
- Biases: `db = mean(dZ, axis=0)`

---

## Resource 10: Why Deep Learning Works — Geometric Intuition

**File:** `10_why_deep_learning_works_geometric_intuition.ipynb`  
**Duration:** ~30 minutes  
**Framework:** NumPy, Matplotlib  
**Prerequisites:** Resource 9

**Description:** Why deep-and-narrow networks beat wide-and-shallow networks, explained through the space-folding insight.

**Contents:**

| Part | Topic | Key Insight |
|------|-------|------------|
| 1 | Universal Approximation Theorem | Existence ≠ findability. One wide layer CAN approximate anything but may need billions of neurons |
| 2 | The Challenge | Complex classification boundary that's hard to separate |
| 3 | Single Neuron | One neuron = one straight line. Can't handle complex boundaries |
| 4 | ReLU as Space-Folding | ReLU creates a "fold" — negative inputs collapse, positive pass through. Each neuron = one fold |
| 5 | Wide vs Deep | **Wide:** n neurons → ~n regions (LINEAR). **Deep:** n neurons × L layers → up to n^L regions (EXPONENTIAL) |

**The key comparison:**
- Wide: 100,000 neurons × 1 layer ≈ 100,000 regions
- Deep: 32 neurons × 5 layers = 160 total neurons → up to 32^5 = 33,554,432 regions
- **160 neurons beats 100,000 neurons because depth creates exponential expressivity**

**Experimental proof:** The notebook trains both architectures and shows the deep network achieves better accuracy, cleaner boundaries, and faster convergence with fewer total parameters.

---

## Resource 11: PyTorch Neural Networks Tutorial

**File:** `11_pytorch_neural_networks_tutorial.ipynb`  
**Duration:** ~35 minutes  
**Framework:** PyTorch  
**Prerequisites:** Resources 1, 2, 9

**Description:** Complete PyTorch journey from tensor fundamentals through autograd to production-ready nn.Module code.

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| I | Tensor Fundamentals | Recap with deep learning context |
| II | Einstein Summation | einsum patterns for DL |
| III | Autograd | requires_grad, computation graphs, .backward(), .grad |
| IV | NN from Primitives | Manual weights with requires_grad + autograd (no nn.Module) |
| V | nn.Module API | nn.Linear, model.parameters(), optimizers (Adam/SGD) |
| VI | Complete Training | DataLoaders, train/val loops, metrics, model saving |
| VII | Comparison | NumPy vs PyTorch primitives vs nn.Module |

**Progression:** Manual everything → autograd replaces backprop → nn.Module replaces weight management → DataLoader replaces batching. Same math at every level.

---

## Resource 12: PyTorch Advanced Tutorial

**File:** `12_pytorch_advanced_tutorial.ipynb`  
**Duration:** ~40 minutes  
**Framework:** PyTorch  
**Prerequisites:** Resource 11

**Description:** Advanced PyTorch: custom autograd, building operations from scratch, custom layers, modern architectures (ResNet, SE, Transformer), and custom training loops.

**Contents:**

| Part | Topic | What You Build |
|------|-------|---------------|
| I | Advanced Autograd | Nested gradients, Jacobians, Hessians, custom autograd.Function, gradient hooks |
| II | Ops from Scratch | Convolution (sliding window), max pooling, batch normalization |
| III | Primitive Layers | Custom layers using only nn.Parameter |
| IV | nn.Module Layers | Proper subclassing with forward(), parameter registration |
| V | Architectures | ResNet residual block, Squeeze-and-Excitation block, Transformer encoder block |
| VI | Custom Training | Gradient clipping, accumulation, mixed precision, learning rate scheduling |
| VII | Practical Demos | Combined custom architecture + custom training loop |

---

## Resource 13: Keras/TensorFlow Neural Networks Tutorial

**File:** `13_keras_tensorflow_neural_networks_tutorial.ipynb`  
**Duration:** ~35 minutes  
**Framework:** TensorFlow / Keras 3  
**Prerequisites:** Resources 1, 3, 9

**Description:** Complete TensorFlow/Keras 3 journey mirroring the PyTorch notebook. From GradientTape to model.fit().

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| I | TF Tensor Fundamentals | Recap with DL context |
| II | Einstein Summation | tf.einsum |
| III | GradientTape | `with tf.GradientTape()`, tape.gradient(), tf.Variable auto-watching |
| IV | NN from Primitives | Manual tf.Variable weights + GradientTape |
| V | Keras 3 API | Sequential, Functional, multi-backend (TF/JAX/PyTorch) |
| VI | Complete Training | model.fit(), callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard) |
| VII | Comparison | NumPy vs TF primitives vs Keras |

**Keras 3 highlight:** Multi-backend support — same Keras code runs on TensorFlow, JAX, or PyTorch.

---

## Resource 14: Keras/TensorFlow Advanced Tutorial

**File:** `14_keras_tensorflow_advanced_tutorial.ipynb`  
**Duration:** ~40 minutes  
**Framework:** TensorFlow / Keras 3  
**Prerequisites:** Resource 13

**Description:** Advanced TF/Keras: custom gradients, building ops from scratch, custom Keras layers, modern architectures, and custom training loops.

**Contents:**

| Part | Topic | What You Build |
|------|-------|---------------|
| I | Advanced GradientTape | Nested tapes, Jacobians, @tf.custom_gradient decorator |
| II | Ops from Scratch | Convolution, pooling, batch normalization using raw TF ops |
| III | Primitive Layers | Custom layers using only tf.Variable |
| IV | Custom Keras Layers | build() + call() + get_config() — lazy initialization, serialization |
| V | Architectures | ResNet block, SE block, Transformer encoder (Functional API) |
| VI | Custom Training | GradientTape loops, tf.distribute for multi-GPU |
| VII | Practical Demos | Combined custom architecture + custom training |

**TF/Keras edge:** tf.data.Dataset for fast preprocessing pipelines. TF Lite for mobile deployment. TF.js for browser deployment.

---

## Resource 15: NumPy Foundations for Deep Learning

**File:** `15_numpy_foundations_for_deep_learning.ipynb`  
**Duration:** ~25 minutes  
**Framework:** NumPy  
**Prerequisites:** Resource 1

**Description:** Focused deep-dive into the specific NumPy operations that map directly to neural network components. Bridges "I know NumPy" to "I see neural network operations."

**Contents:**

| Section | NumPy Operation | Neural Network Component |
|---------|----------------|------------------------|
| 3 | Element-wise functions (exp, maximum, tanh) | Activation functions (sigmoid, ReLU, tanh) + their derivatives |
| 4 | Broadcasting | Bias addition (z = X @ W + **b**), normalization ((X - μ) / σ) |
| 5 | Matrix multiplication (@) | The layer operation (output = input @ weights), shape tracing |

**Key mapping:** Every element-wise op → activation function. Every broadcast → bias addition or normalization. Every matmul → a neural network layer.

---

## Resource 16: JAX Deep Learning Tutorial (Comprehensive)

**File:** `16_jax_deep_learning_tutorial.ipynb`  
**Duration:** ~35 minutes  
**Framework:** JAX  
**Prerequisites:** Resources 1, 4

**Description:** Complete JAX reference — from tensor creation through building and training neural networks. More comprehensive than Resource 4.

**Contents:**

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| 1–3 | JAX Arrays & Ops | Full jax.numpy treatment, explicit PRNG deep-dive, operations |
| 4–5 | Linear Algebra & einsum | DL-specific examples (batch matmul for attention, outer products) |
| 6 | Autodiff | grad, value_and_grad, composability (grad(grad(f))) |
| 7 | JIT | XLA compilation, tracing, benchmarks (10–100x speedup) |
| 8 | vmap | Automatic vectorization, per-example gradients |
| 9–11 | Neural Networks | Full NN from scratch with PyTrees, DL patterns (attention, conv, norm) |

---

## Resource 17: JAX Neural Networks Tutorial

**File:** `17_jax_neural_networks_tutorial.ipynb`  
**Duration:** ~35 minutes  
**Framework:** JAX, Flax, Optax  
**Prerequisites:** Resources 4, 9, 16

**Description:** From raw JAX to production Flax/Optax code. The JAX equivalent of PyTorch's Resources 11+12 combined.

**Contents:**

| Part | Topic | Key Concepts |
|------|-------|-------------|
| I | JAX Fundamentals | Recap: arrays, PRNG, immutability |
| II | Transformations | jit, vmap, grad, composition: `jit(vmap(grad(loss)))` |
| III | Einsum | JAX einsum (identical syntax to NumPy/PyTorch/TF) |
| IV | Autodiff | grad, value_and_grad, jacobian, hessian via jacfwd(jacrev(f)) |
| V | Functional NNs | PyTree parameters, pure forward functions, JIT-compiled training |
| VI | Flax Linen | nn.Module, @nn.compact, model.init(), model.apply() |
| VII | Complete Training | Optax optimizers, composable chains (clip + adam + schedule), best practices |

**Framework decision guide:**

| Choose JAX when | Choose PyTorch when |
|----------------|-------------------|
| TPU support needed | Larger ecosystem/community |
| Per-example gradients (research) | Extensive pretrained models |
| Functional programming preference | OOP/stateful programming preference |
| Cutting-edge research (DeepMind style) | Production deployment priority |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Playground** | HTML/CSS/JS, WebGL/WebGPU, TensorFlow.js, ONNX Runtime Web |
| **Notebooks** | Google Colab (Python 3, Jupyter) |
| **Frameworks** | NumPy, PyTorch, TensorFlow/Keras 3, JAX/Flax/Optax |
| **Playground Models** | MLP, CNN, Transformer (DistilGPT-2/GPT-2), GCN/GAT, Diffusion U-Net |
| **Visualization** | Matplotlib, custom canvas renderers, D3.js-style graphics |

---

## Key Features Across All Notebooks

- **Visual ASCII diagrams** — see the math before you code it
- **"Deep Learning Connection" callouts** — every concept links to real neural networks
- **Framework comparison tables** — NumPy vs PyTorch vs TensorFlow vs JAX side by side
- **Progressive complexity** — each notebook builds on previous ones
- **Fully runnable** — every cell executes on free Colab; outputs pre-rendered
- **No black boxes** — every operation explained before being abstracted
- **Consistent structure** — fundamentals → einsum → autodiff → primitives → high-level API → training

---

## FAQ

**Q: Do I need a GPU?**  
A: No. All notebooks run on CPU. GPU speeds up the framework notebooks (2, 3, 4, 11–17) but is not required. The playground runs on WebGL/WebGPU in your browser.

**Q: Which framework should I learn?**  
A: If you're doing research or starting out, PyTorch (Resources 2, 11, 12). If you're deploying to production/mobile, TensorFlow/Keras (Resources 3, 13, 14). If you're working with TPUs or prefer functional programming, JAX (Resources 4, 16, 17). The math notebooks (5–10) are framework-agnostic.

**Q: Can I skip the math notebooks?**  
A: You can, but you'll be a much better practitioner if you don't. At minimum, do Resource 5 (Calculus) and Resource 8 (Linear Algebra). Resource 9 (from scratch) is the single most valuable notebook in the course.

**Q: What order should I do the notebooks?**  
A: See [Recommended Learning Path](#recommended-learning-path). The short version: Playground → NumPy → your framework's tensors → math → from scratch → your framework's NN tutorials.

**Q: Does the playground send my data anywhere?**  
A: No. Everything runs client-side in your browser. No server, no account, no tracking.

**Q: How is Resource 4 different from Resource 16?**  
A: Resource 4 is a focused introduction to JAX. Resource 16 is the comprehensive reference with deeper coverage, more examples, and practical patterns. Start with 4, use 16 as your reference.

**Q: How is Resource 6 different from Resource 7?**  
A: Resource 6 builds intuition with stories and visuals. Resource 7 provides mathematical rigor with formal definitions and proofs. Both cover the same territory from different angles. Start with 6 for understanding, use 7 for precision.

---

## License & Attribution

All materials are open-source and free for educational use. The playground runs entirely client-side — no data leaves your browser. Notebooks run on Google Colab's free tier.

Created for the DL Mastery deep learning education initiative.

---

*Happy learning. Go build something amazing.*
