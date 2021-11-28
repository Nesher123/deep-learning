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
