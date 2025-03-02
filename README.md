# CIFAR-10 CNN Classifier

## Overview
This repository contains a university project for the **Fundamentals of Machine Vision** course. The project implements a **Convolutional Neural Network (CNN)** for classifying images from the **CIFAR-10** dataset. The model is trained using **PyTorch** and incorporates **k-fold cross-validation** to improve generalization. The dataset consists of 60,000 images across 10 classes, with 80% allocated for training and 20% for testing.

## Features
- **Preprocessing**: Images are converted to grayscale and normalized.
- **CNN Architecture**: Three convolutional layers with batch normalization and pooling layers.
- **Activation Function**: ReLU applied after convolutional layers.
- **Pooling**: Max pooling in early layers and average pooling before the fully connected layers.
- **Classification**: Two fully connected (FC) layers for output prediction.
- **Cross-validation**: Uses 5-fold cross-validation for model evaluation.
- **Evaluation Metrics**: Measures accuracy and F1-score for performance assessment.

## Dataset
The model is trained and tested on the **CIFAR-10** dataset, which consists of 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Preprocessing Steps
- Convert images to grayscale
- Normalize pixel values to the range [-1, 1]
- Apply data augmentation (sharpening, flipping, etc.)

## Installation
To use this repository, ensure you have **Python 3.x** installed along with the required dependencies.

### Dependencies
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Usage
### Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/abolfazmz81/CIFAR-10_CNN_Classifier.git
   cd CIFAR-10_CNN_Classifier
   ```
2. Run Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Execute the cells step by step to train the model and evaluate its performance.

## Model Architecture
The CNN architecture follows this structure:

| Layer | Type | Filters | Kernel Size | Stride |
|--------|-----------------|----------|-------------|--------|
| L1 | Convolution (Conv1) | 16 | (3x3) | (1x1) |
|  | Batch Normalization | --- | --- | --- |
|  | ReLU | --- | --- | --- |
|  | Max Pooling (2x2) | --- | (2x2) | (1x1) |
| L2 | Convolution (Conv2) | 32 | (3x3) | (1x1) |
|  | Batch Normalization | --- | --- | --- |
|  | ReLU | --- | --- | --- |
|  | Max Pooling (2x2) | --- | (2x2) | (1x1) |
| L3 | Convolution (Conv3) | 64 | (3x3) | (1x1) |
|  | Batch Normalization | --- | --- | --- |
|  | ReLU | --- | --- | --- |
|  | Average Pooling (2x2) | --- | (2x2) | (1x1) |
| L4 | Fully Connected (FC1) | --- | --- | --- |
|  | ReLU | --- | --- | --- |
| L5 | Fully Connected (FC2) | --- | --- | --- |

## Results
The model's performance is evaluated using **accuracy** and **F1-score**. These metrics help in understanding how well the model generalizes to unseen data.

## Acknowledgments
- CIFAR-10 dataset: [CIFAR-10 Official Website](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch framework: [PyTorch](https://pytorch.org/)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

