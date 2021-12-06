import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.optimizers as optimizers
import hw2.layers as layers
import hw2.answers as answers
from torch.utils.data import DataLoader
import hw2.training as training
from cs3600.plot import plot_fit

seed = 42
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

# Overfit to a very small dataset of 20 samples
batch_size = 10
max_batches = 2
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

# Get hyperparameters
hp = answers.part2_overfit_hp()

torch.manual_seed(seed)

# Build a model and loss using our custom MLP and CE implementations
model = layers.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
loss_fn = layers.CrossEntropyLoss()

# Use our custom optimizer
optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

# Run training over small dataset multiple times
trainer = training.LayerTrainer(model, loss_fn, optimizer)
best_acc = 0

for i in range(20):
    res = trainer.train_epoch(dl_train, max_batches=max_batches)
    best_acc = res.accuracy if res.accuracy > best_acc else best_acc

test.assertGreaterEqual(best_acc, 98)
# Define a larger part of the CIFAR-10 dataset (still not the whole thing)
batch_size = 50
max_batches = 100
in_features = 3 * 32 * 32
num_classes = 10
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size // 2, shuffle=False)


# Define a function to train a model with our Trainer and various optimizers
def train_with_optimizer(opt_name, opt_class, fig):
    torch.manual_seed(seed)

    # Get hyperparameters
    hp = answers.part2_optim_hp()
    hidden_features = [128] * 5
    num_epochs = 10

    # Create model, loss and optimizer instances
    model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
    loss_fn = layers.CrossEntropyLoss()
    optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

    # Train with the Trainer
    trainer = training.LayerTrainer(model, loss_fn, optimizer)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

    fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
    return fig


fig_optim = None
fig_optim = train_with_optimizer('vanilla', optimizers.VanillaSGD, fig_optim)
