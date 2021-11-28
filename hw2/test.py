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

# Test Linear
out_features = 1000
fc = layers.Linear(in_features, out_features)
x_test = torch.randn(N, in_features)

# Test forward pass
z = fc(x_test)
test.assertSequenceEqual(z.shape, [N, out_features])
torch_fc = torch.nn.Linear(in_features, out_features, bias=True)
torch_fc.weight = torch.nn.Parameter(fc.w)
torch_fc.bias = torch.nn.Parameter(fc.b)
test.assertTrue(torch.allclose(torch_fc(x_test), z, atol=eps))

# Test backward pass
test_block_grad(fc, x_test)

# Test second backward pass
x_test = torch.randn(N, in_features)
z = fc(x_test)
z = fc(x_test)
test_block_grad(fc, x_test)

# Test CrossEntropy
cross_entropy = layers.CrossEntropyLoss()
scores = torch.randn(N, num_classes)
labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

# Test forward pass
loss = cross_entropy(scores, labels)
expected_loss = torch.nn.functional.cross_entropy(scores, labels)
test.assertLess(torch.abs(expected_loss - loss).item(), 1e-5)
print('loss=', loss.item())

# Test backward pass
test_block_grad(cross_entropy, scores, y=labels)
