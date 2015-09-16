import math

#sigmoid as smooth approximation of a step function
def sigmoid(t):
    return 1 / (1 + math.exp(-t))

#dot product of vectors (weights and inputs here for consistancy)
def dot(weights, inputs):
    return sum([inputs[i]*v for i,v in enumerate(weights)])
    
#calculate the output of a neuron    
def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))
    
#feed forward network
def feed_forward(neural_network, input_vector):
    """takes in a neural network
    (represented as a list (one element per layer) of lists (neurons for layer) of lists (weights for neuron) of weights)
    and returns the output from forward-propagating the input"""

    outputs = []

    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1]              # add a bias input
        output = [neuron_output(neuron, input_with_bias)  # compute the output
                  for neuron in layer]                    # for each neuron
        outputs.append(output)                            # and remember it

        # then the input to the next layer is the output of this one
        input_vector = output

    return outputs

#Simple example of the xor network
def xor_network_example():
    xor_network = [# hidden layer
                   [[20, 20, -30],      # 'and' neuron
                    [20, 20, -10]],     # 'or'  neuron
                   # output layer
                   [[-60, 60, -30]]]    # '2nd input but not 1st input' neuron
    #make a truth table from all possible inputs
    for x in [0, 1]:
        for y in [0, 1]:
            print x, y, feed_forward(xor_network, [x, y])[-1]


def backpropagate(network, input_vector, targets):
    """
    1. Run feed_forward on input vector
    2. calculate the error (difference between output and target for each output neuron)
    3. compute the gradient of error as function of neuron's weights, and adjust weights in direction that most decreases error
    4. Propagate these outputs backwoard to infer errors in hidden layer
    5. Adjust weights in hidden layers in similar manner as (3)
    """

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, targets)]

    # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in output_layer])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input
    
               