import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import collections
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

# specify data paths
dataset_path = "./ROI_dataset.dat"
model_path = "./model_ROI.pth"


def save_model(model, path):
    """
    Description: saves model to path
    Arguments:   valid PyTorch neural network and valid file path
    Returns:     nothing
    """
    torch.save(model, path)

def load_model(path):
    """
    Description: loads model from path
    Arguments:   valid file path
    Returns:     PyTorch neural network
    """
    model = torch.load(path)
    model.eval()
    return model

def prepare_data(dataset):
    """
    Description: splits dataset to inputs and outputs and
                 converts numpy arrays to PyTorch tensors
    Arguments:   dataset (including inputs and targets)
    Returns:     inputs and outputs PyTorch tensors
    """
    inputs = torch.from_numpy(dataset[:, :3])
    targets = torch.from_numpy(dataset[:, 3:])
    return inputs, targets

def predict_hidden(dataset):
    """
    Description: retrieves the model saved at model_path prediction from
                 an input dataset
    Arguments:   test_dataset (including inputs and targets)
    Returns:     predictions
    """
    # 1. prepare data
    input, targets = prepare_data(dataset)

    # 2.load model
    model = load_model(model_path)

    # 3. Calculate probabilities and turn into predictions
    pred = model(input)
    predictions = np.zeros_like(pred.data)

    predictions[np.arange(len(pred)), pred.argmax(1)] = 1

    # 3. return predictions
    return predictions

def evaluate_architecture(predictions, dataset):
    """
    Description: evaluates the performance of model depending on predictions
                 and targets
    Arguments:   predictions and dataset (including inputs and targets)
    Prints:         (Confusion Matrix,
                     Precision,
                     Recall,
                     F1 Measuer,
                     Classification Rate
                    )
    """

    # 1. Change the format of predictions and targets
    predictions = torch.max(torch.tensor(predictions), 1)[1].data.numpy()
    targets = torch.max(torch.tensor(dataset[:,3:]), 1)[1].data.numpy()

    # 2. Print quality metrics and return them
    print("confusion_matrix")
    conf_matrix = confusion_matrix(predictions, targets)
    print(conf_matrix)
    print('\n' + "F1 measure")
    print(F1_measure(conf_matrix))
    print('\n' + "Recall Rate")
    print(recall_rate(conf_matrix))
    print('\n' + "Precision Rate")
    print(precision_rate(conf_matrix))
    print('\n' + "Classification Rate")
    print(classification_rate(conf_matrix))

def make_model(n_l, n_h, n_in, n_out):
    """
    Description:  makes a PyTorch model designed by arguments passed
    Arguments:
        n_l     - number of layers
        n_h     - number of neurons in hidden layers
        n_in    - number of neurons in input layer
        n_out   - number of neurons in output layer
    Returns:      initialised PyTorch model
    """
    m = collections.OrderedDict()

    # 1. input layer
    m['layer1'] = nn.Linear(n_in, n_h)

    # 2. hidden layers
    for i in range(n_l - 1):
        m['layer' + str(i + 1) + 'activation'] = nn.ReLU()
        m['layer' + str(i + 2)] = nn.Linear(n_h, n_h)
    m['layer' + str(n_h - 2) + 'activation'] = nn.ReLU()

    # 3. output layer
    m['layer' + str(n_h + 4)] = nn.Linear(n_h, n_out)
    return nn.Sequential(m)

def train_model(dataset, n_l, n_h, n_in, n_out, s_lr, n_e, b_s, w):
    """
    Description:  trains a neural network based on arguments passed
    Arguments:
        dataset  - inputs and targets
        n_l     - number of layers
        n_h     - number of neurons in hidden layers
        n_in    - number of neurons in input layer
        n_out   - number of neurons in output layer
        s_lr    - starting learning rate
        n_e     - number of epochs
        b_s     - batch size
        w       - class weights applied to the loss function
    Returns:      trained PyTorch model
    """

    # 1. prepare_data
    inputs, targets = prepare_data(dataset)

    # 2. build model
    model =  make_model(n_l, n_h, n_in, n_out)

    # 3. Set learning criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight = w)
    optimizer = torch.optim.Adam(model.parameters(), lr = s_lr)

    # 4. Train model
    for epoch in range(n_e):

        # 4.1 implement mini_batches
        k = int(len(inputs)/b_s)
        split_inputs = np.array_split(inputs, k)
        split_targets = np.array_split(targets, k)

        # 4.2 for each mini_batch
        for mini_batch in range(k):

            # Forward Propagation
            pred = model(split_inputs[mini_batch])
            target_mini = split_targets[mini_batch]
            target_idx = torch.max(target_mini, 1)[1]

            # Compute and print loss
            loss = criterion(pred, target_idx)

            # Zero the gradients
            optimizer.zero_grad()

            # perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()

    # 5. return model
    return model

def confusion_matrix(predictions, targets):
    """
    Returns: A confusion matrix
    Assumes: Assumes valid predictions and targets
    Example output: [[ 50  0  0  0]
                     [ 0  50  0  0]
                     [ 0  0  50  0]
                     [ 0  0  0  50]
    """
    # 1. Initialize empty confusion matrix according to number of labels
    number_labels = 4
    conf_matrix = np.zeros([number_labels, number_labels], dtype=int)

    # 2. Fill the confusion matrix according to predictions and targets
    for i in range(len(predictions)):
        conf_matrix[targets[i]][predictions[i]] += 1

    return conf_matrix

def classification_rate(conf_matrix):
    """
    Returns: an array of classification rates for each of the classes
    Assumes: a valid confusion matrix
    Example output: [ 1.  1.  1.  1.]
    """
    # 1. Initialize classification rate array with number of labels
    class_rate = []
    true_predictions  = 0

    # 2. Calculate sum of true predictions (sum of diagnoal)
    for i in range(0, len(conf_matrix)):
        true_predictions += conf_matrix[i][i]

    # 3. Calculate sum of false predictions (FP and FN)
    for i in range(0, len(conf_matrix)):
        false_predictions = 0
        for j in range(0, len(conf_matrix)):
                if i != j:
                    false_predictions += conf_matrix[j][i]
                    false_predictions += conf_matrix[i][j]
        # 4. Calculate classification rate per class
        if (false_predictions + true_predictions) != 0:
            class_rate.append(float(true_predictions)/(false_predictions + true_predictions))
    return np.asarray(class_rate)

def recall_rate(conf_matrix):
    """
    Returns: an array of average recall rates per class in a confusion matrix
             TP/(TP + FN)
    Assumes: a valid confusion matrix
    Example output: [0.91428571 0.92957746 0.91456311 0.91365462]
    """
    # 1. Initialize recall rates array with number of labels
    class_recall = np.zeros(len(conf_matrix))

    # 2. Copy number of TP divided by TP + FN per class into class_recall array
    for i in range(0, len(conf_matrix)):
        if sum(conf_matrix[i,:]) == 0:
            class_recall[i] = None
        else:
            class_recall[i] += float(conf_matrix[i][i]) / float(sum(conf_matrix[i,:]))
    return class_recall

def precision_rate(conf_matrix):
    """
    Returns: an array of average precision rates per class in a confusion matrix
             TP/(TP + FP)
    Assumes: a valid confusion matrix
    Example output: [0.896 0.924 0.942 0.91]
    """
    # 1. Initialize precision rates array with number of labels
    class_precision = np.zeros(len(conf_matrix))

    # 2. Copy number of TP divided by TP + FP per class into class_precision array
    for i in range(0, len(conf_matrix)):
        if sum(conf_matrix[:,i]) > 0:
            class_precision[i] += float(conf_matrix[i][i]) / float(sum(conf_matrix[:,i]))
        else:
            class_precision[i] = None
    return class_precision

def F1_measure(conf_matrix):
    """
    Returns: an array of F1 scores per class in a confusion matrix
    Assumes: a valid confusion matrix
    Example output: [0.896 0.924 0.942 0.91]
    """
    # 1. Initialize F1 measures array with number of labels
    F1_measure = np.zeros(len(conf_matrix))

    # 2. Compute and return F1 measure per class
    precision = precision_rate(conf_matrix)
    recall = recall_rate(conf_matrix)
    for i in range(0, len(conf_matrix)):
        F1_measure[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    return F1_measure

def accuracy(preds, targets):
    """
    Returns: the prediction accuracy given predictions and targets
    Assumes: predictions within
    Example output: [0.9]
    """
    correct = 0
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            correct += 1

    return float(correct)/len(preds)

def main():
    """
    Description: evaluates the model saved at model_path using dataset
                 from dataset_path. Prints metrics to standard output
    Arguments:   none
    Returns:     none
    """

    # 1. load test dataset and model
    dataset = np.loadtxt(dataset_path, dtype = 'float32')

    # 2. find predictions
    predictions = predict_hidden(dataset)

    # 3. evaluate predictions
    evaluate_architecture(predictions, dataset)

if __name__ == "__main__":
    main()
