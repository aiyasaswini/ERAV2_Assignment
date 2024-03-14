## Understand Backpropagation through a neural network
Backpropagation, or backward propagation of errors, 
is an algorithm that is designed to test for errors 
working back from output nodes to input nodes. 
It's an important mathematical tool for improving 
the accuracy of predictions in data mining and machine learning.

<img src="https://github.com/aiyasaswini/ERAV2_Assignment/blob/main/Part1/NEURAL_NETWORK.PNG" width="500">

The initial weights are taken as
w1 = 0.15
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55


# Forward Propagation
Forward propagation is where input data is fed through 
a network, in a forward direction, to generate an 
output. The data is accepted by hidden layers and 
processed, as per the activation function, and 
moves to the successive layer.

From the image above h1 and h2 are the inputs 
h1 = w1 * i1 + w2 *i2 
h2 = w3 * i1 + w4 * i2

a_h1 = σ(h1)
a_h2 = σ(h2)


# Error Functions
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
# BackPropagation Calculation
For all weights we find the partial derivatives
∂E_total/∂w = ∂ (E1 +E2)/∂w
Learning rate = w - learning rate * ∂E_total/∂w
