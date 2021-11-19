import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        # Weights is of size (D + 1) x C
        self.weights = torch.normal(mean=0, std=weight_std, size=(n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        # Class scores is of size N X C, and contains the class scores per sample
        class_scores = torch.matmul(x, self.weights)
        # So, pred is the largest score for a sample; Use argmax to get indices, which correspond to the class
        y_pred = torch.argmax(class_scores, dim=1)

        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = torch.sum(y == y_pred).item() / len(y)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1,
              weight_decay=0.001,
              max_epochs=100,
              ):
        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            self._train_batch(data_loader=dl_train, loss_fn=loss_fn, result=train_res, learn_rate=learn_rate,
                              weight_decay=weight_decay)
            self._train_batch(data_loader=dl_valid, loss_fn=loss_fn, result=valid_res, learn_rate=learn_rate,
                              weight_decay=weight_decay)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def _train_batch(self, data_loader: DataLoader, loss_fn: ClassifierLoss, result: namedtuple, learn_rate=0.1,
                     weight_decay=0.001):
        accuracies, losses = [], []
        data_loader.num_workers = 0  # TODO: Is this a good fix? It works but not sure
        x, y = next(iter(data_loader))
        # get class predictions and save accuracy
        y_pred, scores = self.predict(x)
        accuracies.append(LinearClassifier.evaluate_accuracy(y, y_pred))
        # calculate hinge loss given scores and prediction
        hinge_loss = loss_fn.loss(x, y, scores, y_pred)
        # calculate regularization loss
        regularization_loss = (weight_decay / 2) * torch.pow(self.weights.norm(), 2)
        # save total loss
        losses.append(hinge_loss + regularization_loss)
        # calculate the gradient
        grad = loss_fn.grad() + weight_decay * self.weights
        # update weights
        self.weights -= learn_rate * grad

        # update the result
        result.accuracy.append(sum(accuracies) / len(accuracies))
        result.loss.append(sum(losses) / len(losses))

        return result

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        weights = self.weights

        if has_bias:
            weights = weights[1:]

        C = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        w_images = weights.view((-1, C, H, W))
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp.update({
        'weight_std': 0.005,
        'weight_decay': 0.005,
        'learn_rate': 0.005
    })
    # ========================

    return hp
