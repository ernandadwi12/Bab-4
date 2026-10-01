#Exercise 4.1 - Neuron untuk logical OR

from numpy import exp, array, random, dot

#Set up parameters ==================================================
# Kolom ke-3 (nilai 1) adalah bias, supaya input [0, 0] bisa menghasilkan 0
inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
outputs = array([[0, 1, 1, 1]]).T      # Tabel kebenaran OR
random.seed(1)
weights = 2 * random.random((3, 1)) - 1

#Define a Single Neuron function ====================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

#Train the Neuron ===================================================
for iteration in range(50000):
    output = neuron(inputs, weights)
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

#Test the Neuron ====================================================
print("A  B  | OR (output neuron)  | dibulatkan")
for x in inputs:
    hasil = neuron(x, weights)[0]
    print(x[0], " ", x[1], " |", round(hasil, 4), "            |", round(hasil))