import numpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def sigmoid(x):
    sigm = 1 / (1 + numpy.exp(-x))
    return sigm

def derivative(x):
    return x * (1 - x)

# klasa warstwy sieci
class neuronLayer:
    def __init__(self, inputNodes, outputNodes):
        self.input = list([None] * inputNodes)
        self.output = list([None] * outputNodes)
	
	# inicjowanie wag oraz ewentualnego biasu losowymi wartościami z przedziału (-1,1)
        self.weights = (2 * numpy.random.rand(outputNodes, inputNodes).astype(numpy.float64) - 1) + 0.1
        if with_bias == 1:
            self.bias = (2 * numpy.random.rand(outputNodes, 1).astype(numpy.float64) - 1) + 0.1

	# błąd danej warstwy
        self.error = list([None] * (outputNodes))
        
	# gradient
        self.v = 0
        self.vb = 0   

class neuralNetwork:
    def __init__(self, inputNodes, layersDim, learningRate, momentuM):
        # inicjowanie warstw (neurony wejściowe również traktowane są jako warstwa, ale nie jest na niej dokonywana operacja aktywacji)
        self.layers = list([None] * (len(layersDim) + 1))
        self.layers[0] = neuronLayer(inputNodes, layersDim[0])
        for i in range(1, len(layersDim)):
            self.layers[i] = neuronLayer(layersDim[i-1], layersDim[i])
        self.layers[-1] = neuronLayer(layersDim[-1], layersDim[-1])

	# współczynniki nauki oraz momentum
        self.lr = learningRate
        self.mom = momentuM
		
	# funkcja aktywacji - sigmoida
        self.activation_function = sigmoid

	# błąd popełniony dla całej sieci
        self.global_error = 0

        self.toFile = saveToFile()
        
        pass

    # dla zbioru uczącego - tryb nauki
    def train(self, inputs_list, targets_list):
        # przekształcenie list w macierze (array)
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
		
        #propagacja wprzód
        self.layers[0].output = inputs
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i-1].weights @ self.layers[i-1].output
            if with_bias == 1:
                self.layers[i].output = self.activation_function(self.layers[i].input + self.layers[i-1].bias)
            else:
                self.layers[i].output = self.activation_function(self.layers[i].input)
        
        # wyznaczanie błędów i propagacja wstecz
        self.layers[-1].error = targets - self.layers[-1].output
        for i in reversed(range(len(self.layers) - 1)):
            if i == 0:
                break
            self.layers[i].error = numpy.transpose(self.layers[i].weights) @ numpy.diagflat(derivative(self.layers[i+1].output)) @ self.layers[i+1].error
        
        # aktualizacja wag i biasu
        self.layers[0].v = self.lr * (self.layers[1].error * derivative(self.layers[1].output)) @ numpy.transpose(self.layers[0].output) + self.layers[0].v * self.mom
        self.layers[0].vb = self.lr * (self.layers[1].error * derivative(self.layers[1].output)) + self.layers[0].vb * self.mom
        
        for i in range(1, len(self.layers)-1):
            self.layers[i].v = self.lr * (self.layers[i+1].error * derivative(self.layers[i+1].output)) @ numpy.transpose(self.layers[i].output) + self.layers[i].v * self.mom
            self.layers[i].vb = self.lr * (self.layers[i+1].error * derivative(self.layers[i+1].output)) + self.layers[i].vb * self.mom

        for i in range(len(self.layers)):
            self.layers[i].weights += self.layers[i].v
            self.layers[i].bias += self.layers[i].vb
            
        pass

    # dla zbioru testującego - tryb testowania
    def test(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        # propagacja wprzód
        self.layers[0].output = inputs
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i-1].weights @ self.layers[i-1].output
            if with_bias == 1:
                self.layers[i].output = self.activation_function(self.layers[i].input + self.layers[i-1].bias)
            else:
                self.layers[i].output = self.activation_function(self.layers[i].input)
        return self.layers[-1].output

    def errors(self, targets_list, outputs, final_outputs):
        targets = numpy.array(targets_list, ndmin=2)
        mistakes = 0
        # macierz pomyłek
        typeI_errors = list([0] * outputs)
        typeII_errors = list([0] * outputs)
        for i in range(len(targets)-1):
            guessed_class = numpy.where(final_outputs[i] == numpy.amax(final_outputs[i]))[0]
            correct_class = numpy.where(targets[i] == numpy.amax(targets[i]))[0]
            if guessed_class != correct_class:
                mistakes += 1
                typeI_errors[correct_class[0]] += 1
                typeII_errors[guessed_class[0]] +=1
        accuracy = 100 - (mistakes / len(targets) * 100)
        print("Liczba źle zakwalifikowanych elementów:")
        print(mistakes)
        print("Macierz pomyłek:")
        print(typeI_errors)
        print(typeII_errors)
        print("Dokładność[%]:")
        print(accuracy)
	
    # błąd dla zbioru testującego
    def test_error(self, outputs, targets):
        targets = numpy.array(targets, ndmin=2).T
        return numpy.sum(targets - outputs)
            
class data:
    def __init__(self, inputNode, targetNode):
        self.input = inputNode
        self.target = targetNode
        pass

###################################
###                             ###
###         PARAMETRY           ###
###                             ###
###################################

# wybór zestawu danych: 1 - iris, 2 - seeds
data_choice = 2

# jeżeli iris - 4, seeds - 7
input_nodes = 7
# ostatni element - liczba neuronów warstwy wyjściowej, pozostałe - liczba neuronów w odpowiedniej warstwie ukrytej (liczba warstw ukrytych dowolna)
dimLayers = [5,3]

#luczba epok
epochs = 1000
# oczekiwany błąd, po którym następuje zakończenie nauki
est_error = 0.00001

learning_rate = 0.2
momentum = 0.9

# preferencje dotyczące biasu i losowego trenowania sieci
with_bias = 1
train_randomly = 0

# skok błędu
error_jump = 1

# tworzenie nowej sieci
nn = neuralNetwork(input_nodes, dimLayers, learning_rate, momentum)

# irysy
if data_choice == 1:
    # zbiór uczący = 2/3 całego zbioru
    filename1 = 'iris_train.data'
    # zbiór testujący = 1/3 całego zbioru
    filename2 = 'iris_test.data'
    data_from_file = pd.read_csv(filename1, header=None)
    data_from_file_test = pd.read_csv(filename2, header=None)
    inputs_from_data = data_from_file.iloc[:, :-1].values
    targets_from_data = data_from_file.iloc[1:, [False, False, False, False, True]].values
    inputs_from_data_test = data_from_file_test.iloc[:, :-1].values
    targets_from_data_test = data_from_file_test.iloc[1:, [False, False, False, False, True]].values

    # przypisanie wartości liczbowych gatunkom irysów
    targets = []
    for i in range(len(targets_from_data)):
        if targets_from_data[i] == 'Iris-setosa':
            targets.append([1,0,0])
        elif targets_from_data[i] == 'Iris-versicolor':
            targets.append([0,1,0])
        elif targets_from_data[i] == 'Iris-virginica':
            targets.append([0,0,1])
    targets_test = []
    for i in range(len(targets_from_data_test)):
        if targets_from_data_test[i] == 'Iris-setosa':
            targets_test.append([1,0,0])
        elif targets_from_data_test[i] == 'Iris-versicolor':
            targets_test.append([0,1,0])
        elif targets_from_data_test[i] == 'Iris-virginica':
            targets_test.append([0,0,1])

# nasiona zbóż
if data_choice == 2:
    # wczytywanie z pliku wzorca nasion zbóż
    filename3 = 'seeds_dataset.txt'
    with open(filename3) as file:
        lines = []
        for line in file:
            lines.append(line.rstrip().split("\t"))

    seeds_inputs = [row[:-1] for row in lines]
    seeds_targets = [row[7:] for row in lines]

    # zmiana typu wartości wejściowych string -> float
    inputs_list = []
    for j in range(len(seeds_inputs)):
        floats_in = []
        for i in range(len(seeds_inputs[j])):
            try:
                floats_in.append(float(seeds_inputs[j][i]))
            except ValueError:
                pass
        inputs_list.append(floats_in)

    # przypisanie wartości liczbowych gatunkom nasion
    targets_list = []
    for i in range(len(seeds_targets)):
        if seeds_targets[i][0] == '1':
            targets_list.append([1,0,0])
        elif seeds_targets[i][0] == '2':
            targets_list.append([0,1,0])
        elif seeds_targets[i][0] == '3':
            targets_list.append([0,0,1])

    # wybór wzorców do nauki (5/7 całości) i do testu (2/7)
    inputs_from_data = []
    targets = []
    inputs_from_data_test = []
    targets_test = []
    for i in range(len(inputs_list)):
        if (i < 50) or (i > 69 and i < 120) or (i > 139 and i < 190):
            inputs_from_data.append(inputs_list[i])
            targets.append(targets_list[i])
        else:
            inputs_from_data_test.append(inputs_list[i])
            targets_test.append(targets_list[i])


# normalizacja danych do przedziału (0,1)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
inputs_from_data = sc.fit_transform(inputs_from_data)
inputs_from_data_test = sc.fit_transform(inputs_from_data_test)

# wpisanie do list obiektów klasy data zawierających dane wejściowe i oczekiwany wynik
d = []
for i in range(len(inputs_from_data)-1):
    d.append(data(inputs_from_data[i], targets[i]))
d_test = []
for i in range(len(inputs_from_data_test)-1):
    d_test.append(data(inputs_from_data_test[i], targets_test[i]))

# uczenie sieci
cost = []
ep = 0
for n in range(epochs):
    if train_randomly == 1:
        numpy.random.shuffle(d)
    for x in range(len(inputs_from_data)-1):
        nn.train(d[x].input, d[x].target)
    if n % error_jump == 0:
        ep += error_jump
        # błąd kwadratowy
        global_err = numpy.sum(nn.layers[-1].error * nn.layers[-1].error)
        cost.append(global_err)
        # jeżeli osiągnięto określony błąd zakończenie procesu nauki sieci
        if data_choice == 1:
            if global_err < est_error:
                print("Osiągnięty błąd:")
                print(global_err)
                break

# zapisanie sieci do pliku
pickle.dump(nn, open('network','wb'))

# wczytywanie nauczonej sieci z pliku
nn_test = pickle.load(open("network", "rb"))

# testowanie sieci na wcześniej nieużywanym zbiorze
final_outputs = []
global_output_error = 0
for i in range(len(inputs_from_data_test)-1):
    #print(d_test[i].target)
    output = nn_test.test(d_test[i].input)
    final_outputs.append(output)
    #print(output)
    # błędy dla poszczególnych wejść
    output_err = nn_test.test_error(output, d_test[i].target)
    # błąd popełniony dla całej sieci
    global_output_error += output_err * output_err
print("Błąd dla całej sieci:")
print(global_output_error)
nn.toFile.global_error = global_output_error
# macierz pomyłek
nn_test.errors(targets_test, dimLayers[-1], final_outputs)

# wykres błędu globalnego
plt.figure()
plt.scatter(numpy.arange(0, ep, error_jump),cost)
plt.title("Error")
plt.show()

nn.toFile.save('plik')
