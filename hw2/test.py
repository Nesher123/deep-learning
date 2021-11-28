import torch
import unittest
import hw2.layers as layers
from hw2.grad_compare import compare_layer_to_torch

test = unittest.TestCase()


def test_block_grad(block: layers.Layer, x, y=None, delta=1e-3):
    diffs = compare_layer_to_torch(block, x, y)

    # Assert diff values
    for diff in diffs:
        test.assertLess(diff, delta)


N = 100
in_features = 200
num_classes = 10
eps = 1e-6

# Test Sequential
# Let's create a long sequence of layers and see
# whether we can compute end-to-end gradients of the whole thing.

seq = layers.Sequential(
    layers.Linear(in_features, 100),
    layers.Linear(100, 200),
    layers.Linear(200, 100),
    layers.ReLU(),
    layers.Linear(100, 500),
    layers.LeakyReLU(alpha=0.01),
    layers.Linear(500, 200),
    layers.ReLU(),
    layers.Linear(200, 500),
    layers.LeakyReLU(alpha=0.1),
    layers.Linear(500, 1),
    layers.Sigmoid(),
)
x_test = torch.randn(N, in_features)

# Test forward pass
z = seq(x_test)
test.assertSequenceEqual(z.shape, [N, 1])

# Test backward pass
test_block_grad(seq, x_test)
