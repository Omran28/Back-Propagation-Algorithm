import numpy as np


def Y_Act_Function(activation_fun, a, b, y):
    if activation_fun == 'Sigmoid':
        return 1 / (1 + np.exp(-a * y))
    else:
        return a * np.tanh(b * y)


def forward_step(x, w, no_of_Layers, no_of_Neurons, activation_fun, bias, neurons, a=1, b=1):

    for i in range(no_of_Layers + 1):

        # First hidden layer
        if i == 0:
            for j in range(no_of_Neurons[i]):
                y = 0
                for k in range(len(x)):
                    y += x[k] * w[i][k][j]

                y = Y_Act_Function(activation_fun, a, b, y)
                neurons[i][j + bias] = y

        # Output layer
        elif i == no_of_Layers:
            for j in range(3):
                y = 0
                for k in range(no_of_Neurons[-1] + bias):
                    y += neurons[i - 1][k] * w[i][k][j]

                y = Y_Act_Function(activation_fun, a, b, y)
                neurons[i][j] = y
        else:
            # Hidden Layers
            for j in range(no_of_Neurons[i]):
                y = 0
                for k in range(no_of_Neurons[i-1] + bias):
                    y += neurons[i - 1][k] * w[i][k][j]

                y = Y_Act_Function(activation_fun, a, b, y)
                neurons[i][j + bias] = y

    return neurons


def back_step(activation_fun, t, neurons, no_of_Layers, no_of_Neurons, weight, bias, Sigma, a=1, b=1):
    newSigma = []
    Sum = 0
    counter = -1

    for i in range(no_of_Layers, -1, -1):
        counter += 1

        # Output layer
        if i == no_of_Layers:
            for j in range(3):
                if activation_fun == 'Sigmoid':
                    newSigma.append((int(t[j]) - neurons[-1][j]) * a * neurons[-1][j] * (1 - neurons[-1][j]))
                else:
                    newSigma.append((int(t[j]) - neurons[-1][j]) * (b / a) * (a - neurons[-1][j]) * (a + neurons[-1][j]))

                Sigma[counter][j] = newSigma[j]

        # Hidden layer
        else:
            previous_sigma = list(newSigma)
            newSigma.clear()

            for j in range(no_of_Neurons[i]):
                # Sum of previous sigmas
                for k in range(len(previous_sigma)):
                    Sum += (previous_sigma[k] * weight[i + 1][j + bias][k])

                if activation_fun == 'Sigmoid':
                    newSigma.append(a * neurons[i][j + bias] * (1 - neurons[i][j + bias]) * Sum)
                else:
                    newSigma.append((b / a) * (a - neurons[i][j + bias]) * (a + neurons[i][j + bias]) * Sum)

                Sigma[counter][j] = newSigma[j]

    return Sigma


def update_weight(w, eta, sigma, x, no_of_Layer, no_of_Neurons, neurons, bias):
    # W[layer #no][1st node][2nd node]
    # sigma,neuron[layer #no,column][neuron #no,row]

    counter = len(sigma) - 1
    for i in range(no_of_Layer + 1):
        # Input layer
        if i == 0:
            for k in range(no_of_Neurons[i]):
                for j in range(len(x)):
                    w[i][j][k] = w[i][j][k] + (eta * sigma[counter][k] * x[j])
            counter -= 1

        else:
            # Output layer
            if i == no_of_Layer:
                for k in range(3):
                    for j in range(no_of_Neurons[-1] + bias):
                        w[i][j][k] = w[i][j][k] + (eta * sigma[counter][k] * neurons[i][k])
            # Hidden layers
            else:
                for k in range(no_of_Neurons[i]):
                    for j in range(no_of_Neurons[i-1] + bias):
                        w[i][j][k] = w[i][j][k] + (eta * sigma[counter][k] * neurons[i][k])
            counter -= 1
    return w
