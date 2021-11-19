r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False.<br>
The test set does **NOT** allow us to estimate our in-sample error.<br>
The test set is unknown during the training phase.<br>
We assume that if the training data is good enough (sufficiently large number of samples, indicative of the real world behaviour, etc.), 
then by finding the parameteres that give us the minimal error **during training**, we will get a good estimate for the test data ("ground truth").<br>
The **validation** set allows us to estimate our in-sample error.

2. False.<br>
Usually we want a larger percentage of data to train upon.<br>
Meaning, to best estimate results on the test data, we should have a richer training data, and as explained above, we would like a large-enough 
training set.<br>
Usually, an 80-20 % split in favor of the training is a good rule-of-thumb.

3. True.<br>
The test-set should **NOT** be used during cross-validation.<br>
The test set should always be treated as if it doesn't exist until a decision is made for the model's hyperparameters.<br>
As if we commit to the paramteres we chose and "use them in production" :)<br>
That is why we allocate some of the data as validation set, such that we use IT to test each fold's performance 
and select the best hyperparameters selection (those who's average loss is minimal).

4. True.<br>
After performing cross-validation, we use the validation-set performance of each fold as a proxy for the model's generalization error.<br>
See section 3 above.<br>
The validation set is our "sanity check" for questioning our training set's output and help us prevent a scenario of overfitting.
"""

part1_q2 = r"""
The friend's approach is **NOT** justified.<br>
It is the analogous to looking at the answers to come up with a solution.<br>
A kind of "cheating".<br>
By basing the parameters on the test set and taking the lowest error in there, the friend is treating the training set as training, 
and might cause overfitting for the particular test set.<br>
The correct approach would to choose the 𝜆 values which produced the best results on the **validation** set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Increasing k may not necessariy lead to improved generalization for unseen data.<br>
Remember that k is the number of nearset neighbors we look at.<br>
k ranges from 1 (sensitive to noise) to the number of samples in our data (too generalized).<br>
By taking larger k's we may base our classification solely on our data's majority votes, which is wrong of course - 
if some class is more prevalent than others, and the proximity of nearset neighbors looses its importance.
The weighted nearest neighbour classifier (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#The_weighted_nearest_neighbour_classifier) 
is a possible adjustments to improve the discussed fallback, but still not error-proof.<br>
Choosing the best k is a good scenario for using Cross Validation as we did in this exercise.
"""

part2_q2 = r"""
1. Training our model solely on the training set may lead to overfit.<br>
Meaning, we may get very low error rate, but it is not indicative of data we have not yet seen, 
and the error we will get on the test set will be high.<br>
The validation set is **validating** and reassuring we keep questioning ourselves and are working as thoroughly as possible before 
choosing a final answer as the model's parameters.<br>

2. As in part1_q2, we should **NOT** base our model's parameters on the test set.<br>
This is simply a misuse of the test set, because the real world may behave differently and we will overfit this test set.<br>
K-fold CV should be used in order to select the best model with respect to validation set accuracy, 
i.e for each i in K, train on K-i folds, check average error on validation set and pick the hyperparameters which yield the lowest error.<br>
These are the parameters that supposedly give us the best generalization for the test data (and future data to come). 
"""

# ==============
# Part 3 answers

part3_q1 = r"""
The selection of Δ > 0 is arbitrary for the SVM loss L(W) because we only care that it is positive.<br>
By doing so, we require classifications to be at least Δ-different for each classification score.<br>
W can be of different magnitudes (it is a vector), and by enforcing Δ-difference, we simply scale W, without effecting 
the final classifications themselves.
"""

part3_q2 = r"""
1. It seems like the linear model is actually learning the weights per digit, but that some weights are very 
similar.<br>
Those weights can be translated to the similarity of matrices for each digit - varying angles and boldness of the 
typed digit.<br>
Maybe also the ratio of white and black pixels...<br>
It also seems like the model fails when the image is very different from other images with the same digit 
(especially the first and last errors).<br>
For example, the first mis-classification where 5 is interpreted as 6 occurs probably because the weights for such 
class is already similar.<br>
Same for the second error (line 2) where 6 is classified as 4 - even for me it is difficult to classify actually!<br>
It is worth mentioning that the classifier never failed when given image was 0 or 8<br>
Also, it never classifies 0, 1, 5, or 8 when it is mis-classifying...<br>

2. This interpretation is different from KNN since KNN basically says "if you're close to coordinate x, then the 
classification will be similar to observed outcomes at x".<br>
SVM tries to generalize and learn the **representation** of each class.
"""

part3_q3 = r"""
1. 
    Based on the graph of the training set loss, we would say that the learning rate we chose is **good**.<br>
    When training for the same number of epochs with a **too low** learning rate, our loss would not converge to the 
    validation set loss, and the total accuracy would be lower.

    Choosing a **too high** learning rate may overshoot minimal points and may also cause the loss to increase between 
    epochs.<br>
    <img src="https://miro.medium.com/max/918/1*7WRRrBoUDhLf2AYlzrFRZg.png" width="800" height="400">

2. Based on the graph of the training and test set accuracy, we would say that the model **slightly overfitted the 
training set**.<br>
The results on the training set are very good (in this case) which usually means overfitting.<br>
We can also notice the training set's accuracy exceeds the validation set accuracy at some moments.<br>
In general, the higher the training set accuracy, the higher the overfitting, and vice versa.
"""

# ==============
# Part 4 answers

part4_q1 = r"""
To recall, linear regression has four assumptions as follows:<br>
1. Linear relationship between predictors and the target variable, meaning the pattern must in the form of a 
straight-line (or a hyperplane in case of multiple linear regression)<br>
2. Homoscedasticity, i.e., constant variance of the residuals<br>
3. Independent observations. This is actually equivalent to independent residuals<br>
4. Normality of residuals, i.e., the residuals follow the normal distribution<br>

We can check the first three assumptions in the above via the residual plot!

Assumption 1: Linear relationship<br>
This assumption is validated if there is no discerning, nonlinear pattern in the residual plot.<br>
If the residual points' pattern is no-hozontal (has a U-shape for example) then the true relationship is nonlinear.  

Assumption 2: Constant variance<br>
This assumption is validated if the residuals are scattered evenly (about the same distance) with respect to the 
zero-horizontal line throughout the x-axis in the residual plot.<br>

Assumption 3: Independent Observations<br>
This assumption is validated if there is no discerning pattern between several consecutive residuals in the residual 
plot.

Finally, one other reason this is a good residual plot is, that independent of the value of an independent variable 
(x-axis), the residual errors are approximately distributed in the same manner.<br>
In other words, we do not see any patterns in the value of the residuals as we move along the x-axis.<br>
Hence, this satisfies our earlier assumption that regression model residuals are independent and normally distributed.

In conclusion, a good example of a residual plot is:<br>
<img src="https://miro.medium.com/max/430/1*40E7lY7o39jddXBKQypeTA.png" width="800" height="400">

We can clearly see that the final plot after CV is better then the plot for the top-5 features since the dots are 
closer to y = 0 axis.
"""

part4_q2 = r"""
1. Adding non-linear features to our data helps us finding a **linear relationship** between the **transformations** of 
X and Y.<br>
We can thereby obtain a non-linear model in our original data by combining a linear method with non-linear 
transformation of our original data.<br>
The key to understanding what is going on is that we are producing a linear model in a high dimensional space where the 
data coordinates are given by non-linear transforms of the original input features. This results in a linear surface in 
the higher dimensional space.<br>
(see https://www.futurelearn.com/info/courses/advanced-machine-learning/0/steps/49532)

So it is still a linear regression model but of x' and y' and not x and y...


2. No, we cannot fit any non-linear function of the original features with this approach because the features may be 
completely uncorrelated in the first place (0 correlation between x & y).predict


3. If we want to plot a decision boundary for non-linear features, then we may get a non-hyperplane in the **original** 
dimension because **the linear separation occurs in a higher dimension**.<br/>
Thus, the decision boundary may be hyperbolic or of any degree actually (N-1 dimensions, to be precise, where N is the 
dimension in which the linear separation occurs).<br/>
(see https://www.fatalerrors.org/images/blog/01da298452fd594ea4b9c1e7e76d2e74.jpg)

For the non-linear features themselves, we WILL get a hyperplane representing the decision boundary (in dimension N).
"""

part4_q3 = r"""
1. `np.logspace` simply gives values on a larger scale (orders of magnitude different) compared to `np.linspace`.<br/>
Since we are using CV, we want to better-tune the hyperparameters, and using np.linspace with lambdas would not have a 
significant impact on the loss at every iteration and the final decision for hyperparameters.

    For example:<br/>
    `print(np.linspace(0.02, 2.0, num=20))`
    >>>[ 0.02        0.12421053  0.22842105  0.33263158  0.43684211  0.54105263
      0.64526316  0.74947368  0.85368421  0.95789474  1.06210526  1.16631579
      1.27052632  1.37473684  1.47894737  1.58315789  1.68736842  1.79157895
      1.89578947  2.        ]
  
    `print(np.logspace(0.02, 2.0, num=20))`
    >>>[   1.04712855    1.33109952    1.69208062    2.15095626    2.73427446
      3.47578281    4.41838095    5.61660244    7.13976982    9.07600522
      11.53732863   14.66613875   18.64345144   23.69937223   30.12640904
      38.29639507   48.68200101   61.88408121   78.6664358   100.        ]

2. K-fold CV is fitting the data **K times** (3 times in our case).<br/>
Including the hyperparameters - we have 3 degrees and 20 lambdas, which yields 60 different combinations.<br/>
So overall the model fitted to data |lambda_values| * |degrees| * |folds| = 20 * 3 * 3 = 180 times.
"""
