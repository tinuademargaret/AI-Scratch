import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_network.gradient_descent import batch_gradient_descent, stochastic_gradient_descent
from neural_network.network import NeuralNetwork


def get_config_data():
    """ Read the configurations for Neural Network
    Returns:
        sizes: Number of layers and number of neurons in the network
        learning_rate
        activation_function: list of activation functions from layer to another - Sigmoid,tanh,ReLU,LeakyReLU
        loss_function: MSE(Mean square error), BCE(Binary Cross Entropy), CE(Cross Entropy)
        gradient_type: FGD(full gradient descent) MB(mini-batch gradient descent)
        epochs: number of epochs
    """
    config_path = "./config.yml"

    with open(config_path) as f:
        configurations = yaml.safe_load(f)

    return configurations


def load_dataset(path):
    """ Function to read data from source
    Args:
        path: path to data source
    Returns:
        Dataframe of the breast cancer data
    """

    data = pd.read_csv(path)

    return data


def pre_process_data(data, train_split=0.70, test_split=0.30):
    """ Data Preprocessing for modelling
        - Data Normalising
        - Data Splitting into train and test
    """
    # drop unamed column
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

    X = data.drop('diagnosis', axis=1)

    y = data.diagnosis
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_split,
                                                        train_size=train_split,
                                                        random_state=30)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    return X_train, X_test, y_train, y_test


def predict(model, test_input, test_output):
    predictions = model.feed_forward(test_input)
    prediction_accuracy = accuracy_score((predictions > 0.5).astype(int), test_output)
    return prediction_accuracy


def main():
    config = get_config_data()
    sizes = config["sizes"]
    activation_functions = config["activation_function"]
    loss_function = config["loss_function"]
    gradient_type = config["gradient_type"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    data_path = config["data_path"]
    data = load_dataset(data_path)
    X_train, X_test, y_train, y_test = pre_process_data(data)
    layers = []
    for size, activation_function in zip(sizes, activation_functions):
        layers.append((size, activation_function))

    if gradient_type == "SGD":
        model = NeuralNetwork(X_train.shape[1],
                              layers,
                              learning_rate,
                              loss_function
                              ).create_network()
        print(f"Model architecture...{model.architecture}")
        model_output, model_accuracy = stochastic_gradient_descent(epochs, model, X_train, y_train, batch_size)
    else:
        model = NeuralNetwork(X_train.shape[1],
                              layers,
                              learning_rate,
                              loss_function
                              ).create_network()
        print(f"Model architecture...{model.architecture}")
        model_output, model_accuracy = batch_gradient_descent(epochs, model, X_train, y_train)
    print(f"Model accuracy score...:{model_accuracy}")
    print("Making prediction...")
    model_prediction_accuracy = predict(model, X_test, y_test)
    print(f"Model prediction accuracy score...:{model_prediction_accuracy}")


if __name__ == '__main__':
    main()
