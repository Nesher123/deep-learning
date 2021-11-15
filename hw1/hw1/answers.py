r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False.
The test set is unknown during the training phase.
We assume that if the training data is good enough (sufficiently large number of samples, indicative of the real world behaviour, etc.), 
then by finding the parameteres that give us the minimal error **during training**, we will get a good estimate for the test data ("ground truth").

2. False.
Usually we want a larger percentage of data to train upon.
Meaning, to best estimate results on the test data, we should have a richer training data, and as explained above, we would like a large-enough 
training set.
Usually, an 80-20 % split in favor of the training is a good rule-of-thumb.

3. True.
The test set should always be treated as if it doesn't exist until a decision is made for the model's hyperparameters.
As if we commit to the paramteres we chose and "use them in production" :)
That is why we allocate some of the data as validation set, such that we use IT to test each fold's performance 
and select the best hyperparameters selection (those who's average loss is minimal).

4. True.
See section 3 above.
The validation set is our "sanity check" for questionning our training set's output and help us prevent a scenario of overfitting.
"""

part1_q2 = r"""
**Your answer:**
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
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

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
"""

part4_q2 = r"""
**Your answer:**
"""

part4_q3 = r"""
**Your answer:**
"""

# ==============
