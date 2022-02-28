r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=8, h_dim=32, z_dim=8, x_sigma2=0.0001, learn_rate=0.0001, betas=(0.9, 0.9),
    )
    # ========================
    return hypers


# What does the  𝜎2  hyperparameter (x_sigma2 in the code) do? Explain the effect of low and high values.
part2_q1 = r"""
As in the field of Statistics, sigma^2 represents the **Variance** of a distribution.<br>
Here, it is the Likelihood Variance of the Normal distribution.

As the value decreases, we put MORE weight on the VAE's reconstruction loss term and the results will be closer to one 
another and the original inputs (more rigid).<br>
As the value increases, we put LESS weight on the VAE's reconstruction loss term and the results will be more creative 
(or flexible).
"""

# 1. Explain the purpose of both parts of the VAE loss term - reconstruction loss and KL divergence loss.
# 2. How is the latent-space distribution affected by the KL loss term?
# 3. What's the benefit of this effect?
part2_q2 = r"""
**1**.<br>
The purpose of the reconstruction loss term is to penalize the network for creating outputs that are too different from 
the original inputs.<br>
To increase the probability and make sure that the reconstructed images will be SIMILAR to the original.

Regarding the KL divergence, it measures how much the original and the reconstructed images diverge from each other 
between two probability distributions.<br>
It's purpose is to improve the approximation of the posterior distribution to generate better latent space samples.<br>
By minimizing this loss we optimize the probability dist.s' parameters to be as close as it can to the target dist.


**2**.<br>
The larger the KL loss term, the more the latent-space distribution becomes (standard) Normal.<br>
It is a regularization that encourages the encoder to be evenly distributed around the center of the latent space.

**3**.<br>
The benefit of this effect is to avoid overfitting of the model, and to get better generalized results - because the 
distribution covers more areas of the latent space.
"""

# In the formulation of the VAE loss, why do we start by maximizing the evidence distribution, $p(\bb{X})$?
part2_q3 = r"""
In the formulation of the VAE loss, we start by maximizing the evidence distribution, $p(\bb{X})$, in order to make the 
model sample data with a high probability of being similar to our dataset.
"""

# In the VAE encoder, why do we model the log of the latent-space variance corresponding to an input, 𝜎2𝛼,
# instead of directly modelling this variance?
part2_q4 = r"""
In the VAE encoder, we model the **log** of the latent-space variance corresponding to an input, 𝜎2𝛼, instead of 
directly modelling this variance because log because it brings stability and ease of training. 
For example, the standard deviation values are usually very small 1>>𝜎>0.
The optimization has to work with very small numbers, where the floating point arithmetic and the poorly defined 
gradient bring <u>numerical instabilities</u>.

By taking the log, we map the numerically unstable very small numbers in [1,0] interval to [log(1), -inf], and have a 
easier/simpler intervals to work with.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You can add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You can add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # hypers = dict(
    #     batch_size=32, z_dim=128,
    #     data_label=1, label_noise=0.28,
    #     discriminator_optimizer=dict(
    #         type='Adam',
    #         weight_decay=0.02,
    #         betas=(0.5, 0.999),
    #         lr=0.0002,
    #     ),
    #     generator_optimizer=dict(
    #         type='Adam',
    #         weight_decay=0.02,
    #         betas=(0.5, 0.999),
    #         lr=0.00021,
    #     ),
    # )
    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.28,
        discriminator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5, 0.999),
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5, 0.999),
            lr=0.00021,
        ),
    )
    # ========================
    return hypers


# Explain in detail why during training we sometimes need to maintain gradients when sampling from the GAN,
# and other times we don’t.
# When are they maintained and why? When are they discarded and why?
part3_q1 = r"""
The GAN consists of two models: Discriminator & Generator.<br>
Therefore, we also split the training parts since we dont want one training to affect the other.

The reason for discarding the gradients in the discriminator training is that we dont want the discriminator 
training to affect the generator training.<br>
We want the generator gradients to be affected only by its own loss, and to not update them during discriminator train.

While optimizing the discriminator we're adding some fake inputs to the original inputs.<br>
Both inputs create a loss, but we want to maintain the discriminator' gradients solely according to the original inputs.
<br>
The fake data is only used for testing the performance and we don't want to update the discriminator accordingly.
"""

# 1. When training a GAN to generate images, should we decide to stop training solely based on the fact that the
# Generator loss is below some threshold? Why or why not?
# 2. What does it mean if the discriminator loss remains at a constant value while the generator loss decreases?
part3_q2 = r"""
**1**.<br>
When training a GAN to generate images, we should **NOT** decide to stop training solely based on the fact that the 
Generator's loss is below some threshold because our eventual goal is for the generator to be good enough so that the 
discriminator would not be able to differentiate between a real and a fake image.<br> 
In case we stop solely based on a threshold, we make the discriminator's job easier and it might be able to 
differentiate, so we stopped prematurely.<br>

Since the Generator's training may take quite some time, the early stopping according to the generator's loss alone is 
not recommended.

We should bare in mind that the performance of the GAN depends on the 2 parts (Generator AND Discriminator), so looking 
only at the generator's performance is clearly not enough.


**2**.<br>
When the discriminator loss remains at a constant value while the generator loss decreases it means that the generator 
is getting better at "fooling" the discriminator by making more realistic images.<br>
The fact that the discriminator loss remains at a constant value actually means its classification is ALSO getting 
better at detecting fake images, but not at the the same velocity as the generator.
"""

# Compare the results you got when generating images with the VAE to the GAN results.
# What’s the main difference and what’s causing it?
part3_q3 = r"""
We saw that <u>GANs are better at generating (more realistic) images</u>.<br>
Furthermore, during the encoding process of the VAEs, important data is lost naturally, by removing and compressing 
parts of it, and therefore it yields inferior results compared to the GAN.

The 2 methods have different approaches as well:
The VAE learning approach is to compress the data correctly in order to be able to reconstruct it later, meaning it 
focuses directly on the data.
The GAN approach is to train an entity which is able to distinguish between real and fake data, and another, con-artist 
entity.<br>
It "cares" relies more about the two adversaries' expertise, and not on the features of the data.

The background parts in the VAE are more blurry and the face area is more detailed, while the GAN has much more 
similarities in the background parts.<br>

The differences between the methods are also related to their training process:
VAEs have a final target (improve accuracy), while GANs' training process is more complicated - 2 models in a 
competition to "ruin" the other's progress.
"""

# ==============
