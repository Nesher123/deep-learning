r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. The shape of the Jacobian tensor of the output of the layer w.r.t. the input X would be 
[in_features, out_features, N] = [1024, 2048, 128].
<br>
2. 1GB of RAM/GPU memory will be required to store the above Jacobian.
"""


# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.1, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.2, 0.05, 0.01, 0.0001, 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
When training a model with the cross-entropy loss function, it **IS** possible for the test loss to increase for a few 
epochs while the test accuracy also increases!<br>


"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""

part3_q5 = r"""
**Your answer:**

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
