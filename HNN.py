import random as rd
import numpy as np

def dSigmoid(x):
    """
    function to generate a error signal to correct the MLP on backpropagation
    @param X - Numpy array - vector to be transformed using sigmoid derivative
    """
    return x*(1-x)

def sigmoid(x):
    """
    threshold function to infer the activating of a hiperplane according 
    to sigmoid function.
    @param X - Numpy array - vector to be transformed using sigmoid function
    """
    return 1/(1+np.exp(-x))

# TRAINING EXAMPLES
# problem attribuites + BIAS
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Classificator attribuite
Y = np.array([[0],
			  [1],
			  [1],
			  [0]])

# NEURAL NETWORK INITIAL SETUP
numAtt = 3              # Number of problem attribuites + BIAS
numMid = 6              # Number of internal perceptrons
numOut = 6              # Number of out perceptrons
learnT = 60000          # Learning Time

#init synapses in NN
syn0 = np.zeros((numAtt, numMid))
syn1 = np.zeros((numMid, numOut))

rd.seed(1)
for i in xrange(numMid):
    syn0[i%(numAtt-1), i] = rd.uniform(-1, 1)
    syn0[numAtt-1, i] = rd.uniform(-1, 1)

for i in xrange(numOut):
    syn1[i%(numMid-1), i] = rd.uniform(-1, 1)
    syn1[numMid-1, i] = rd.uniform(-1, 1)

# TRAINING PROCESS
for j in xrange(learnT):

	# Propagation from input signals to layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(l0.dot(syn0))
    l2 = sigmoid(l1.dot(syn1))

    # BACKPROPAGATION
    # Calculates the error factor in the previous layer
    l2_error = Y - l2
    l2_delta = l2_error * dSigmoid(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * dSigmoid(l1)
    
    # Update all layers
    aux0 = l0.T.dot(l1_delta)
    aux1 = l1.T.dot(l2_delta)
    
    for i in xrange(numMid):
        syn0[i%(numAtt-1), i] += aux0[i%(numAtt-1), i]
        syn0[numAtt-1, i] += aux0[numAtt-1, i]
    
    for i in xrange(numOut):
        syn1[i%(numMid-1), i] += aux1[i%(numMid-1), i]
        syn1[numMid-1, i] += aux1[numMid-1, i]
    
    # Show the process of continue learning
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        print syn0
        print syn1