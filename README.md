# Handwritten Digit Recognition with a Simple Neural Network

A beginner-friendly deep learning project that classifies handwritten digits (0–9) from the **MNIST** dataset using a fully connected neural network built with **TensorFlow/Keras**. The project focuses not only on accuracy, but also on making sure the model **generalises** to unseen data (i.e. doesn't overfit).

## Overview

| | |
|---|---|
| **Dataset** | MNIST: 70,000 grayscale 28×28 images (60,000 train / 10,000 test) |
| **Task** | Multi-class image classification (10 classes) |
| **Tools** | Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib |
| **Environment** | Google Colab / Jupyter Notebook |

## Model Architecture

```
Input (784) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
```

- **Optimizer:** Adam
- **Loss:** Sparse categorical cross-entropy
- **Batch size:** 32
- **Epochs:** up to 10 (with early stopping)
- **Validation split:** 20%

## How Overfitting Is Controlled

1. **Dropout (0.2):** randomly disables 20% of hidden neurons during training so the network can't simply memorise the training set.
2. **EarlyStopping:** monitors validation loss (`patience=3`) and restores the best weights if validation performance stops improving.
3. **Generalisation check:** compares train, validation and test accuracy and reports the train–test gap.

## Project Workflow

### 1. Load and Explore the Data
- Load MNIST and inspect array shapes and pixel range
- Visualise the first 10 training images with their labels
- **Normalise** pixel values from 0–255 to 0–1 (faster, more stable gradient descent)
- **Flatten** each 28×28 image into a 784-length vector

### 2. Build and Train the Network
- Build and compile the model, then train with a validation split
- Plot training vs. validation accuracy and loss curves

### 3. Evaluate, Predict and Experiment
- Evaluate on the held-out test set
- Predict and display test images (predicted vs. actual)
- Inspect misclassified images
- **Experiment:** change a single variable (hidden neurons 128 → 256) and compare the two models fairly

## Results

| Model | Hidden Neurons | Parameters | Test Accuracy | Train–Test Gap |
|-------|:-------------:|:----------:|:-------------:|:--------------:|
| Model 1 | 128 | 101,770 | **97.90%** | 1.08 pp |
| Model 2 | 256 | 203,530 | 97.76% | 0.95 pp |

### Key Takeaways
- **Model 1 (128 neurons) performed slightly better** than the larger Model 2, even though Model 2 has roughly twice as many parameters.
- Doubling the hidden layer size reduced test accuracy by 0.14 percentage points, a sign of diminishing returns on a simple dataset like MNIST.
- The train–test gap is small for both models (about 1 pp), so neither shows major overfitting. Dropout and early stopping help keep it in check.

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib jupyter
```

### Run the Notebook

```bash
git clone https://github.com/Devapriya-20/Handwritten-text-recognition-using-DL
cd Handwritten-text-recognition-using-DL

```

or open it directly in **Google Colab** and choose **Runtime → Run all**. MNIST is downloaded automatically through `keras.datasets`, so no manual data setup is needed.

## Repository Structure

```
├── Handwritten_Digit_Recognition_project.ipynb   # Main notebook
└── README.md
```


## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes and Christopher J.C. Burges
- [TensorFlow/Keras](https://www.tensorflow.org/) for the deep learning framework
