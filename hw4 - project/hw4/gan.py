import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        channel_list = [128, 256, 512, 1024]
        kernel_list = [4] * len(channel_list)
        stride_list = [2] * (len(channel_list) - 1) + [1]
        padding_list = [1] * (len(channel_list) - 1) + [0]

        for idx, in_c, out_c, ker, stride, padding in zip(range(len(channel_list)), [in_size[0]] + channel_list,
                                                          channel_list, kernel_list, stride_list, padding_list):
            modules.append(nn.Conv2d(in_c, out_c, ker, stride, padding, bias=False))

            if idx < len(channel_list) - 1:
                modules.append(nn.Dropout2d(0.1))
                modules.append(nn.BatchNorm2d(out_c))
                modules.append(nn.LeakyReLU())

        self.feature_extractor = nn.Sequential(*modules)
        size_list = [1, in_size[0], in_size[1], in_size[2]]
        temp = torch.rand(size_list)
        output = self.feature_extractor(temp)
        output = output.view(1, -1)
        classifier_module = [nn.Linear(output.shape[1], channel_list[-1]),
                             nn.Linear(channel_list[-1], channel_list[-1]),
                             nn.Linear(channel_list[-1], 1)]
        self.classifier = nn.Sequential(*classifier_module)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        y = self.classifier(features.view(batch_size, -1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # hint (you dont have to use....)
        from .autoencoder import DecoderCNN
        modules = []
        channel_list = [1024, 512, 128, 64, out_channels]
        kernel_list = [featuremap_size] * len(channel_list)
        stride_list = [2] * len(channel_list)
        padding_list = [0] + [1] * (len(channel_list) - 1)
        self.featuremap_size = featuremap_size

        for idx, in_c, out_c, kernel, stride, padding in zip(range(len(channel_list)), [z_dim] + channel_list,
                                                             channel_list, kernel_list, stride_list, padding_list):
            modules.append(nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False))

            if idx < len(channel_list) - 1:
                modules.append((nn.Dropout2d(0.1)))
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Tanh())

        self.generator = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n, self.z_dim, device=device)

        if with_grad:
            samples = self.forward(z)
        else:
            with torch.no_grad():
                samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.view(z.shape[0], -1, 1, 1)
        x = self.generator(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    distribution_error = label_noise * torch.rand_like(y_data) + (data_label - 0.5 * label_noise)
    classification_error = label_noise * torch.rand_like(y_data) + (1 - data_label - 0.5 * label_noise)
    loss_module = nn.BCEWithLogitsLoss()
    loss_data = loss_module(y_data, distribution_error)
    loss_generated = loss_module(y_generated, classification_error)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    y_compare = torch.ones_like(y_generated) * data_label
    loss_module = nn.BCEWithLogitsLoss()
    loss = loss_module(y_generated, y_compare)
    # ========================
    return loss


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    n = x_data.shape[0]
    data = x_data
    x_discrininator = dsc_model(gen_model.sample(n, False))
    dsc_loss = dsc_loss_fn(dsc_model(data), x_discrininator)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    x_discriminator = dsc_model(gen_model.sample(n, True))
    gen_loss = gen_loss_fn(x_discriminator)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved
