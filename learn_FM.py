import numpy as np
import torch
import torch.nn as nn
import collections
import math

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# specify data paths
dataset_path = "./FM_dataset.dat"
model_path = "./model_FM.pth"

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


def predict_hidden(test_dataset):
    """
    Description: returns the model saved at model_path prediction from
                 an input dataset
    Arguments:   test_dataset (including inputs and targets)
    Returns:     predictions
    """
    # 1. prepare data
    inputs, targets = prepare_data(test_dataset)

    # 2.load model
    model = load_model(model_path)

    # 3. calculate and return predictions
    return model(inputs)

def evaluate_architecture(predictions, data_val):
    """
    Description: evaluates the performance of model depending on predictions
                 and targets
    Arguments:   predictions and dataset (including inputs and targets)
    Returns:     tuple of scores:
                    (mean squared error,
                     root mean squared error,
                     mean absolute error,
                     r squared score,
                     explained variance score
                    )
    """
    # 1. extract targets and convert data to numpy array
    inputs, targets = prepare_data(data_val)
    targets = targets.numpy()
    predictions = predictions.data.numpy()

    # f = open("predictions.txt", "a")
    # for i in range(len(predictions)):
    #     print("predictions: ", predictions[i].astype(int))
    #     print("targets    : ", targets[i].astype(int))
    #     print("---------------------------------------------")
    #     f.write("\npredictions: " + np.array2string(predictions[i].astype(int)))
    #     f.write("\ntargets    : " + np.array2string(targets[i].astype(int)))
    #     f.write("\n---------------------------------------------")

    # 2. calculate scores
    rsquared = r2_score(targets, predictions)
    v_score = explained_variance_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    return (mse, math.sqrt(mse), mae, rsquared, v_score)


def print_scores(scores):
    """
    Description: prints metrics to standard output
    Arguments:   scores
    Returns:     none
    """
    print(
           "SCORES: " + "\n" +
           "MSE: " + str(scores[0]) + "\n" +
           "RMSE: " + str(scores[1]) + "\n" +
           "MAE: " + str(scores[2]) + "\n" +
           "R-Squared: " + str(scores[3]) + "\n" +
           "Explained Variance: " + str(scores[4]) + "\n" +
           "-------------------------------------------------------------" + "\n"
         )


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

def adjust_learning_rate(optimizer, epoch, s_lr, lr_d):
    """
    Description:        decays the current learning rate to 0.99 every lr_d epochs
    Arguments:
        optimizer     - PyTorch omptimizer object
        epoch         - current epoch
        s_lr          - starting learning rate
        lr_d          - learning rate decay
    Returns:            new learning rate
    """
    lr = s_lr * (0.99 ** (epoch // lr_d))
    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr

def train_model(dataset, n_l, n_h, n_in, n_out, s_lr, n_e, b_s, lr_d):
    """
    Description:  trains a neural network based on arguments passed
    Arguments:
        inputs  - input dataset
        targets - target dataset
        n_l     - number of layers
        n_h     - number of neurons in hidden layers
        n_in    - number of neurons in input layer
        n_out   - number of neurons in output layer
        s_lr    - starting learning rate
        n_e     - number of epochs
        b_s     - batch size
    Returns:      trained PyTorch model
    """
    # 1. prepare_data
    inputs, targets = prepare_data(dataset)

    # 2. build model
    model = make_model(n_l, n_h, n_in, n_out)

    # 3. Set learning criterion
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=s_lr)

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

            # Compute and print loss
            loss = criterion(pred, split_targets[mini_batch])

            # Zero the gradients
            optimizer.zero_grad()

            # perform backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()

            # adjust_learning_rate
            adjust_learning_rate(optimizer, epoch, s_lr, lr_d)

    # 5. return model
    return model


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
    scores = evaluate_architecture(predictions, dataset)

    # 4. print scores
    print_scores(scores)


if __name__ == "__main__":
    main()
