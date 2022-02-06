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
        batch_size=64, h_dim=128, z_dim=128, x_sigma2=0.5, learn_rate=0.01, betas=(0.9, 0.9),
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


part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

# ==============
