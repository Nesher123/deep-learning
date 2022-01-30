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


part2_q1 = r"""
**Your answer:**

"""

part2_q2 = r"""
**Your answer:**

"""

part2_q3 = r"""
**Your answer:**


"""

part2_q4 = r"""
**Your answer:**

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
