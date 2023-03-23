from Preprocessing import *
from Gui import *
from BackPropagation import *


def bias_or_not(Bias_bool, weights, training_dataset, testing_dataset):
    if Bias_bool:
        training_dataset.insert(loc=0, column='X0', value=1)
        testing_dataset.insert(loc=0, column='X0', value=1)
        for layer in weights:
            a = []
            for c in range(len(layer[0])):
                a.append(random.uniform(-1, 1))
            layer.insert(0, a)
        return training_dataset, testing_dataset, weights, 1
    else:
        return training_dataset, testing_dataset, weights, 0


def Initialization(Bias):
    Neurons = []
    Sigma = []
    Sigma.append([1, 1, 1])
    count = 0
    for i in range(noOfLayers + 1):
        count -=1
        Neurons.append([])
        if i == noOfLayers:
            for j in range(3):
                Neurons[i].append(1)
        else:
            Sigma.append([])
            for j in range(noOfNeurons[count]):
                Sigma[i + 1].append(1)
            for j in range(noOfNeurons[i] + Bias):
                Neurons[i].append(1)
    return Neurons, Sigma


def train(Weights, Epochs, training, target, No_Layers, No_Neurons, Act_Fun, Eta, Bias, Neurons, Sigma):
    for i in range(Epochs):
        for j in range(len(training)):
            x = pd.Series(training.iloc[j, :])
            Neurons = forward_step(x, Weights, No_Layers, No_Neurons, Act_Fun, Bias, Neurons)
            Sigma = back_step(Act_Fun, target[j], Neurons, No_Layers, No_Neurons, Weights, Bias, Sigma)
            Weights = update_weight(Weights, Eta, Sigma, x, No_Layers, No_Neurons, Neurons, Bias)
    return Weights


def test(Weights, testing, target, No_Layers, No_Neurons, Act_Fun, Bias, Neurons):
    matrix = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    c = 0
    TP1 = TN1 = FP1 = FN1 = 0
    TP2 = TN2 = FP2 = FN2 = 0
    TP3 = TN3 = FP3 = FN3 = 0

    for j in range(len(testing)):
        x = pd.Series(testing.iloc[j, :])
        Neurons = forward_step(x, Weights, No_Layers, No_Neurons, Act_Fun, Bias, Neurons)
        y1 = Neurons[-1][0]
        y2 = Neurons[-1][1]
        y3 = Neurons[-1][2]
        if y1 > y2 and y1 > y3:
            y = "100"
        elif y2 > y1 and y2 > y3:
            y = "010"
        else:
            y = "001"
        if target[j] != y:
            c += 1

        # Confusion matrix
        if target[j] == "100" and y == "100":
            matrix[0][0] += 1
        elif target[j] == "100" and y == "010":
            matrix[0][1] += 1
        elif target[j] == "100" and y == "001":
            matrix[0][2] += 1

        elif target[j] == "010" and y == "100":
            matrix[1][0] += 1
        elif target[j] == "010" and y == "010":
            matrix[1][1] += 1
        elif target[j] == "010" and y == "001":
            matrix[1][2] += 1

        elif target[j] == "001" and y == "100":
            matrix[2][0] += 1
        elif target[j] == "001" and y == "010":
            matrix[2][1] += 1
        elif target[j] == "001" and y == "001":
            matrix[2][2] += 1

    TP1 += matrix[0][0]
    FP1 += matrix[1][0] + matrix[2][0]
    TN1 += matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix[2][2]
    FN1 += matrix[0][1] + matrix[0][2]

    TP2 += matrix[1][1]
    FP2 += matrix[0][1] + matrix[2][1]
    TN2 += matrix[0][0] + matrix[0][2] + matrix[2][0] + matrix[2][2]
    FN2 += matrix[1][0] + matrix[1][2]

    TP3 += matrix[2][2]
    FP3 += matrix[0][2] + matrix[1][2]
    TN3 += matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
    FN3 += matrix[2][0] + matrix[2][1]

    print(matrix)
    acc = 100 - (c / len(target) * 100)
    print("Accuracy: " + str(acc))

    print('\n                         Predicted value')
    print('                             P    |    N    ')
    print('')
    print('                      T      ' + str(TP1) + '    |   ' + str(TN1))
    print('    Actual value -----------------------')
    print('                      F      ' + str(FP1) + '    |   ' + str(FN1))

    print('\n                         Predicted value')
    print('                             P    |    N    ')
    print('')
    print('                      T      ' + str(TP2) + '    |   ' + str(TN2))
    print('    Actual value -----------------------')
    print('                      F      ' + str(FP2) + '    |   ' + str(FN2))

    print('\n                         Predicted value')
    print('                             P    |    N    ')
    print('')
    print('                      T      ' + str(TP3) + '    |   ' + str(TN3))
    print('    Actual value -----------------------')
    print('                      F      ' + str(FP3) + '    |   ' + str(FN3))


# Reading the file
data = pd.read_csv("penguins.csv")

# Preprocessing filling the missing values randomly
data = fill_missing_value(data)

# Manual label Encoder
data['gender'] = lbl_encoder(data['gender'])

# Scaling
min = np.min(data['bill_length_mm'])
max = np.max(data['bill_length_mm'])
data['bill_length_mm'] = scaling(data['bill_length_mm'], min, max)
min = np.min(data['bill_depth_mm'])
max = np.max(data['bill_depth_mm'])
data['bill_depth_mm'] = scaling(data['bill_depth_mm'], min, max)
min = np.min(data['flipper_length_mm'])
max = np.max(data['flipper_length_mm'])
data['flipper_length_mm'] = scaling(data['flipper_length_mm'], min, max)
min = np.min(data['body_mass_g'])
max = np.max(data['body_mass_g'])
data['body_mass_g'] = scaling(data['body_mass_g'], min, max)

c1 = data[:50]
c2 = data[50:100]
c3 = data[100:]

# Shuffled Data
c1 = c1.sample(frac=1).reset_index()
c2 = c2.sample(frac=1).reset_index()
c3 = c3.sample(frac=1).reset_index()

# Inputs
noOfLayers = noOfLayersTxt.get()
noOfNeurons = noOfNeuronsTxt.get()
# for i in range(len(noOfNeurons)):
noOfNeurons = noOfNeurons.split(',')
for i in range(len(noOfNeurons)):
    noOfNeurons[i] = int(noOfNeurons[i])
eta = etaTxt.get()
epochs = epochsTxt.get()
bias_bool = biasTxt.get()
activationFun = fun.get()

# Train-Test split
training_dataset = pd.concat([c1.iloc[:30, :], c2.iloc[:30, :], c3.iloc[:30, :]])
testing_dataset = pd.concat([c1.iloc[30:, :], c2.iloc[30:, :], c3.iloc[30:, :]])
all_data = pd.concat([c1, c2, c3])

# Shuffling
training_dataset = training_dataset.sample(frac=1).reset_index()
# testing_dataset = testing_dataset.sample(frac=1).reset_index()

# Training Dataset
t_train = np.array(species_encoder(training_dataset['species']))
training_dataset = training_dataset.iloc[:, 3:]

# Testing Dataset
t_test = np.array(species_encoder(testing_dataset['species']))
testing_dataset = testing_dataset.iloc[:, 3:]

# Weights
weights = []
Neurons = []
for i in range(noOfLayers + 1):
    weights.append([])

    # Input layer
    if i == 0:
        for j in range(5):
            weights[i].append([])
            for k in range(noOfNeurons[i]):
                weights[i][j].append(random.uniform(-1, 1))
    else:
        for j in range(noOfNeurons[i - 1]):
            # Output layer
            if i == noOfLayers:
                weights[i].append([])

                for k in range(3):
                    weights[i][j].append(random.uniform(-1, 1))
            else:
                # Hidden layers
                # for o in range(noOfNeurons[]):
                weights[i].append([])
                for k in range(noOfNeurons[i]):
                    weights[i][j].append(random.uniform(-1, 1))

if bias_bool:
    neurons, sigma = Initialization(1)
else:
    neurons, sigma = Initialization(0)

# Adding bias
training_dataset, testing_dataset, weights, bias = bias_or_not(bias_bool, weights, training_dataset, testing_dataset)

# Train
weights = train(weights, epochs, training_dataset, t_train, noOfLayers, noOfNeurons, activationFun, eta, bias,
                neurons, sigma)

# Train data
test(weights, training_dataset, t_train, noOfLayers, noOfNeurons, activationFun, bias, neurons)

# Test data
test(weights, testing_dataset, t_test, noOfLayers, noOfNeurons, activationFun, bias, neurons)
