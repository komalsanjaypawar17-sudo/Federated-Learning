# Handwritten Digit Recognition with a CNN (MNIST)

A Convolutional Neural Network (CNN) built with Keras (TensorFlow) that recognizes handwritten digits from 0 to 9.

On the 10,000 test images it had never seen during training, the model reached **[YOUR ACCURACY]% accuracy**.

## Overview

To a computer, a handwritten digit is just a grid of numbers. This project trains a neural network to turn that grid into the correct digit. It is a classic introduction to image classification and deep learning.

## Dataset

- **MNIST**: 70,000 grayscale images of handwritten digits, loaded directly through `keras.datasets`
- 60,000 training images and 10,000 test images
- Each image is 28 × 28 pixels
- 10 classes (digits 0 to 9)

## How It Works

1. **Load the data**: MNIST is downloaded automatically by Keras.
2. **Preprocess**: images are reshaped to `28 × 28 × 1`, converted to `float32`, and scaled from 0-255 to 0-1. Labels are one-hot encoded.
3. **Build the model**: a small CNN (see below).
4. **Train**: 10 epochs, batch size 128, Adam optimizer, categorical cross-entropy loss.
5. **Evaluate**: loss and accuracy are measured on the unseen test set.
6. **Predict**: the digit with the highest probability is chosen for each test image.

## Model Architecture

| Layer | Details |
|---|---|
| Conv2D | 64 filters, 3 × 3 kernel, ReLU activation |
| MaxPooling2D | 2 × 2 pool |
| Flatten | converts feature maps to a single vector |
| Dense | 64 units, ReLU activation |
| Dense | 10 units, Softmax activation |

The final Softmax layer outputs a probability for each digit, for example "92% sure this is a 7".

## Results

| Metric | Value |
|---|---|
| Test accuracy | [YOUR ACCURACY]% |
| Test loss | [YOUR LOSS] |

*Results may vary slightly from run to run because of random weight initialization.*

## Tech Stack

- Python 3
- Keras (TensorFlow)
- NumPy
- Matplotlib

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the project

```bash
python mnist_cnn.py
```

If you are using a Jupyter notebook, open it and run all cells.

## Project Structure

```
.
├── mnist_cnn.py      # training and evaluation script (rename to match your file)
└── README.md
```

## Possible Improvements

- Add a second convolutional layer and Dropout to reduce overfitting
- Plot training and validation accuracy curves
- Display sample predictions and misclassified digits
- Try the same approach on a more challenging dataset such as Fashion-MNIST

## Author

**Komal Pawar**
M.Sc. Data Science student


## License

This project is open source and available under the [MIT License](LICENSE).
