# PyTorch-Neural-Network-Implementation

This project focuses on implementing and comparing various aspects of deep learning, from data loading to training neural networks, using both PyTorch's built-in functionalities and custom implementations. The tasks include calculating execution time, comparing custom and PyTorch data loaders, implementing and training neural networks, and developing a custom back-propagation algorithm.

## Project Structure

- `data/`: Contains the MNIST dataset and any related files.
- `models/`: Contains the implementations of neural network architectures.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `scripts/`: Python scripts for data loading, training models, and plotting results.
- `results/`: Directory to save model checkpoints, loss logs, and performance graphs.
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn

## Tasks

### Task 1: Custom Data Loader vs. PyTorch Data Loader

#### Downloading MNIST Dataset in Google Colab

```python
import torch
from torchvision import datasets, transforms

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
