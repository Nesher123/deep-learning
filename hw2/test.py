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

# # Test LeakyReLU
# alpha = 0.1
# lrelu = layers.LeakyReLU(alpha=alpha)
# x_test = torch.randn(N, in_features)
#
# # Test forward pass
# z = lrelu(x_test)
# test.assertSequenceEqual(z.shape, x_test.shape)
# test.assertTrue(torch.allclose(z, torch.nn.LeakyReLU(alpha)(x_test), atol=eps))
#
# # Test backward pass
# test_block_grad(lrelu, x_test)

# Test ReLU
relu = layers.ReLU()
x_test = torch.randn(N, in_features)

# Test forward pass
z = relu(x_test)
test.assertSequenceEqual(z.shape, x_test.shape)
test.assertTrue(torch.allclose(z, torch.relu(x_test), atol=eps))

# Test backward pass
test_block_grad(relu, x_test)
