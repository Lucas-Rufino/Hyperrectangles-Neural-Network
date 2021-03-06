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
numMid = 4              # Number of internal perceptrons
numOut = 1              # Number of out perceptrons
learnT = 60000          # Learning Time

# Random process to init synapses in NN
np.random.seed(1)
syn0 = 2*np.random.random((numAtt, numMid)) - 1
syn1 = 2*np.random.random((numMid, numOut)) - 1

# TRAINING PROCESS
for j in xrange(learnT):

	# Propagation from input signals to layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(l0.dot(syn0))
    l2 = sigmoid(l1.dot(syn1))

    # BACKPROPAGATION
    # Calculates the error factor from output signal to the expected output
    l2_error = Y - l2
        
    # It calculates the output error impact in output layer
    l2_delta = l2_error * dSigmoid(l2)

    # Calculates the error factor from output layer to the middle layer
    l1_error = l2_delta.dot(syn1.T)
    
    # It calculates the middle error impact in middle layer
    l1_delta = l1_error * dSigmoid(l1)
    
    # Update all layers
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
    # Show the process of continue learning
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))