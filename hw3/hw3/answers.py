r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=256, seq_len=64,
        h_dim=128, n_layers=3, dropout=0.3,
        learn_rate=0.01, lr_sched_factor=0.5, lr_sched_patience=2
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Have"
    temperature = 0.5
    # ========================
    return start_seq, temperature


# Why do we split the corpus into sequences instead of training on the whole text?
part1_q1 = r"""
We split the corpus into sequences instead of training on the whole text because:<br>
a. The learning process is more generalized this way, and prevents overfitting 
and can improve the robustness of model when available training data is limited.<br>
b. It might be too large to fit in memory and result in very long training times 
(see https://arxiv.org/ftp/arxiv/papers/1708/1708.05604.pdf).<br>
&emsp; More troubling, attempting to back-propagate across very long input sequences may result in vanishing gradients, 
and in turn, an un-learnable model.
"""

# How is it possible that the generated text clearly shows memory longer than the sequence length?
part1_q2 = r"""
The generated text shows memory longer than the sequence length because of the hidden states ,H 
("This can be thought of as the memory of that layer").<br>
These pass previous knowledge and therefore have more memory in each layer.
"""

# Why are we not shuffling the order of batches when training?
part1_q3 = r"""
We are NOT shuffling the order of batches when training because each batch rely on its previous batch's hidden state 
(the "memory").<br>
Naturally, the continuity of text is important for context and logic, and if we shuffle words we can understand the text 
differently (or not understand at all)...<br>
Here, "understanding" means generating similar texts, so we learn the text as it is, otherwise the resulted text will 
be irrelevant.
"""

# 1. Why do we lower the temperature for sampling (compared to the default of 1.0)?
# 2. What happens when the temperature is very high and why?
# 3. What happens when the temperature is very low and why?
part1_q4 = r"""
1. When sampling, we would prefer to control the distributions and make them less uniform to increase the chance of 
sampling the char(s) with the <u>highest scores</u> compared to the others.<br>
Low temperature results in less uniform distributions and more structured sampling (which we want).<br>
<br>
2. When the temperature is very high the probabilities are more uniform and the model will therefore choose random 
chars for each word, and the generated text might have no meaning...<br>
<br>
3. When the temperature is very low the probabilities are not uniform at all and the model will therefore always choose 
the chars that are more common (more "safe" to choose) from training.<br>
This result in less generalized generated text and thus less errors but more similar outputs all the time."""
# ==============
