r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False.
The test set does **NOT** allow us to estimate our in-sample error.
The test set is unknown during the training phase.
We assume that if the training data is good enough (sufficiently large number of samples, indicative of the real world behaviour, etc.), 
then by finding the parameteres that give us the minimal error **during training**, we will get a good estimate for the test data ("ground truth").
The **validation** set allows us to estimate our in-sample error.

2. False.
Usually we want a larger percentage of data to train upon.
Meaning, to best estimate results on the test data, we should have a richer training data, and as explained above, we would like a large-enough 
training set.
Usually, an 80-20 % split in favor of the training is a good rule-of-thumb.

3. True.
The test-set should **NOT** be used during cross-validation.
The test set should always be treated as if it doesn't exist until a decision is made for the model's hyperparameters.
As if we commit to the paramteres we chose and "use them in production" :)
That is why we allocate some of the data as validation set, such that we use IT to test each fold's performance 
and select the best hyperparameters selection (those who's average loss is minimal).

4. True.
After performing cross-validation, we use the validation-set performance of each fold as a proxy for the model's generalization error.
See section 3 above.
The validation set is our "sanity check" for questioning our training set's output and help us prevent a scenario of overfitting.
"""

part1_q2 = r"""
The friend's approach is **NOT** justified.
It is the analogous to looking at the answers to come up with a solution.
A kind of "cheating".
By basing the parameters on the test set and taking the lowest error in there, the friend is treating the training set as training, 
and might cause overfitting for the particular test set.
The correct approach would to choose the 𝜆 values which produced the best results on the **validation** set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Increasing k may not necessariy lead to improved generalization for unseen data.
Remember that k is the number of nearset neighbors we look at.
k ranges from 1 (sensitive to noise) to the number of samples in our data (too generalized).
By taking larger k's we may base our classification solely on our data's majority votes, which is wrong of course - 
if some class is more prevalent than others, and the proximity of nearset neighbors looses its importance.
The weighted nearest neighbour classifier (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#The_weighted_nearest_neighbour_classifier) 
is a possible adjustments to improve the discussed fallback, but still not error-proof.
Choosing the best k is a good scenario for using Cross Validation as we did in this exercise.
"""

part2_q2 = r"""
1. Training our model solely on the training set may lead to overfit.
Meaning, we may get very low error rate, but it is not indicative of data we have not yet seen, 
and the error we will get on the test set will be high.
The validation set is **validating** and reassuring we keep questioning ourselves and are working as thoroughly as possible before 
choosing a final answer as the model's parameters.

2. As in part1_q2, we should **NOT** base our model's parameters on the test set.
This is simply a misuse of the test set, because the real world may behave differently and we will overfit this test set.
K-fold CV should be used in order to select the best model with respect to validation set accuracy, 
i.e for each i in K, train on K-i folds, check average error on validation set and pick the hyperparameters which yield the lowest error.
These are the parameters that supposedly give us the best generalization for the test data (and future data to come). 
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
