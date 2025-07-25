"""
JAX Neural Network Implementation for MNIST Classification

This script implements a fully-connected neural network using JAX for automatic
differentiation and optimization. The network is trained on the MNIST dataset
to classify handwritten digits.

Key features:
- Pure JAX implementation with automatic differentiation
- JIT compilation for performance
- Vectorized operations using vmap
- MNIST dataset integration via PyTorch
"""

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
import time

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

# Network architecture: input -> hidden -> hidden -> output
layer_sizes = [784, 512, 512, 10]  # 784 input (28x28 flattened), 2 hidden layers, 10 outputs
step_size = 0.01                   # Learning rate
num_epochs = 8                     # Number of training epochs
batch_size = 128                   # Batch size for training
n_targets = 10                     # Number of classes (digits 0-9)

# =============================================================================
# NETWORK INITIALIZATION
# =============================================================================

def random_layer_params(m, n, key, scale=1e-2):
    """
    Initialize weights and biases for a single layer.
    
    Args:
        m: Number of input features
        n: Number of output features  
        key: JAX random key for reproducibility
        scale: Scale factor for weight initialization
    
    Returns:
        Tuple of (weights, biases) for the layer
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    """
    Initialize all layers for a fully-connected neural network.
    
    Args:
        sizes: List of layer sizes [input_size, hidden1_size, ..., output_size]
        key: JAX random key for reproducibility
    
    Returns:
        List of (weights, biases) tuples for each layer
    """
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Initialize network parameters
params = init_network_params(layer_sizes, random.key(0))

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    """Rectified Linear Unit activation function."""
    return jnp.maximum(0, x)

# =============================================================================
# FORWARD PASS (PREDICTION)
# =============================================================================

def predict(params, image):
    """
    Forward pass through the neural network for a single image.
    
    Args:
        params: Network parameters (list of weight/bias tuples)
        image: Flattened input image (784-dimensional)
    
    Returns:
        Logits for each class (10-dimensional)
    """
    activations = image
    # Forward pass through hidden layers
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    # Final layer (no activation function, just logits)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)  # Log-softmax for numerical stability

# Test single prediction
random_flattened_image = random.normal(random.key(1), (28 * 28,))
preds = predict(params, random_flattened_image)
print(f"Single prediction shape: {preds.shape}")

# =============================================================================
# BATCHED OPERATIONS
# =============================================================================

# Create batched version for handling multiple images at once
batched_predict = vmap(predict, in_axes=(None, 0))

# Test batched prediction
random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
batched_preds = batched_predict(params, random_flattened_images)
print(f"Batched prediction shape: {batched_preds.shape}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def one_hot(x, k, dtype=jnp.float32):
    """
    Create a one-hot encoding of x of size k.
    
    Args:
        x: Array of class indices
        k: Number of classes
        dtype: Data type for the output
    
    Returns:
        One-hot encoded array
    """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    """
    Calculate classification accuracy.
    
    Args:
        params: Network parameters
        images: Batch of input images
        targets: One-hot encoded target labels
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

# =============================================================================
# LOSS FUNCTION AND OPTIMIZATION
# =============================================================================

def loss(params, images, targets):
    """
    Compute cross-entropy loss.
    
    Args:
        params: Network parameters
        images: Batch of input images
        targets: One-hot encoded target labels
    
    Returns:
        Average cross-entropy loss
    """
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    """
    Update network parameters using gradient descent.
    
    Args:
        params: Current network parameters
        x: Batch of input images
        y: One-hot encoded target labels
    
    Returns:
        Updated network parameters
    """
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def numpy_collate(batch):
    """
    Custom collate function to convert PyTorch tensors to numpy arrays.
    This is needed because we're using JAX with PyTorch's DataLoader.
    """
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    """
    Convert PIL image to flattened numpy array.
    
    Args:
        pic: PIL image
    
    Returns:
        Flattened numpy array of shape (784,)
    """
    return np.ravel(np.array(pic, dtype=jnp.float32))

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Prepare full datasets for evaluation
print("Preparing evaluation datasets...")
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Load test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

# =============================================================================
# TRAINING LOOP
# =============================================================================

print(f"Starting training for {num_epochs} epochs...")
print(f"Network architecture: {layer_sizes}")
print(f"Batch size: {batch_size}, Learning rate: {step_size}")
print("-" * 60)

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training loop over batches
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x, y)
    
    epoch_time = time.time() - start_time

    # Evaluate on full datasets
    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} sec")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("-" * 60)

print("Training completed!")