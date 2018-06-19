import math

class Linear:
    def output(self, x):
        return x
    def derivative(self, x):
        return 1.0
class Tanh:
    def output(self, x):
        return math.tanh(x)
    def derivative(self, x):
        x = output(x)
        return 1.0-x*x

class Neuron:
    def __init__(self, transfer, inputs, output_ix, weight_ix):
        self.transfer = transfer
        self.inputs = inputs
        self.output_ix = output_ix
        self.weight_ix = weight_ix

    def forward(self, states, weights, preacts=None):
        v = self._preact(states, weights)
        if preacts is not None:
            preacts[self.output_ix] = v
        states[self.output_ix] = self.transfer.output(v)

    def backward(self):
        pass

    def _preact(self, states, weights):
        v = 0.0
        weight_ix = self.weight_ix
        for input_ix in self.inputs:
            v += weights[weight_ix]*states[input_ix]
            weight_ix += 1
        return v

def create(kind, inputs, output_ix, weight_ix):
    transfer = { "linear": Linear, "tanh": Tanh, }[kind]()
    return Neuron(transfer, inputs, output_ix, weight_ix)
