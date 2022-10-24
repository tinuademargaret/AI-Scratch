import numpy as np
from sklearn.metrics import accuracy_score


def batch_gradient_descent(epochs,
                           model,
                           network_input,
                           output
                           ):

    losses = {'train': [], 'validation': []}
    network_output = 0
    for epoch in range(epochs):
        network_output, train_loss = model.train(network_input, output)

        print("\rProgress: {:2.1f}".format(100 * epoch / float(epochs))
              + "% ... Training loss: " + str(train_loss)

              )

        losses['train'].append(train_loss)

    accuracy = accuracy_score((network_output > 0.5).astype(int), output)

    return network_output, accuracy


def stochastic_gradient_descent(epochs,
                                model,
                                network_input,
                                output,
                                batch_size
                                ):
    losses = {'train': [], 'validation': []}
    network_output = 0
    for epoch in range(epochs):

        batch = np.random.choice(network_input.shape[0], size=batch_size)

        X = np.take(network_input, batch, axis=0)

        y = np.take(output, batch, axis=0)

        network_output, train_loss = model.train(X, y)

        print("\rProgress: {:2.1f}".format(100 * epoch / float(epochs))
              + "% ... Training loss: " + str(train_loss)

              )

        losses['train'].append(train_loss)

    accuracy = accuracy_score((network_output > 0.5).astype(int), y)

    return network_output, accuracy
