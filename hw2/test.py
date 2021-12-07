import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.cnn as cnn

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

test_params = [
    dict(
        in_size=(3, 100, 100), out_classes=10,
        channels=[32] * 4, pool_every=2, hidden_dims=[100] * 2,
        conv_params=dict(kernel_size=3, stride=1, padding=1),
        activation_type='relu', activation_params=dict(),
        pooling_type='max', pooling_params=dict(kernel_size=2),
    ),
    dict(
        in_size=(3, 100, 100), out_classes=10,
        channels=[32] * 4, pool_every=2, hidden_dims=[100] * 2,
        conv_params=dict(kernel_size=5, stride=2, padding=3),
        activation_type='lrelu', activation_params=dict(negative_slope=0.05),
        pooling_type='avg', pooling_params=dict(kernel_size=3),
    ),
]

for i, params in enumerate(test_params):
    torch.manual_seed(seed)

    net = cnn.ConvClassifier(**params)
    print(f"\n=== test {i=} ===")
    print(net)

    test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
    test_out = net(test_image)
    print(f'{test_out=}')

    expected_out = torch.load(f'tests/assets/expected_conv_out_{i:02d}.pt')
    diff = torch.norm(test_out - expected_out).item()
    print(f'{diff=:.3f}')
    test.assertLess(diff, 1e-3)
