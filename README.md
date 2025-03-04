# Neural MNIST

## Overview
**MNIST Neural Network Classifier** is a multi-class classification model designed to recognize **handwritten digits (0-9)** from grayscale images using a **neural network**. The model is trained using **supervised learning** and leverages **forward propagation and backpropagation** for optimization.

## Features
- **Reads dataset** from files containing training images and labels.
- **Shuffles and splits** data into training and testing sets based on a given percentage.
- **Initializes parameter matrices** dynamically for propagation.
- **Performs forward propagation** to compute predictions.
- **Calculates the cost function** to evaluate model performance.
- **Applies backpropagation** to compute gradients and update parameters.
- **Optimizes model over multiple iterations** to achieve high accuracy in digit recognition.

## Implementation Details
- **Input Layer:** 400 neurons (20×20 pixel grayscale images flattened).
- **Hidden Layer:** 25 neurons to increase model complexity and performance.
- **Output Layer:** 10 neurons (one for each digit from 0 to 9).
- **Optimization:** Uses **backpropagation** for gradient calculation and parameter updates.
- **Accuracy Calculation:** After training, forward propagation is applied one last time to determine the model’s recognition accuracy.

## Notes
- The model uses a small **neural network architecture** optimized for MNIST digit classification.
- The training process involves computing cost function and gradients iteratively to minimize classification errors.
- Proper initialization of weight matrices is crucial for **convergence and performance**.

