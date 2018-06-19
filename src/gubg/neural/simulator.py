from . import neuron

class Simulator:
    def clear(self):
        self.nr_states, self.nr_weights = 0, 0
        self._neurons = []

    def __init__(self):
        self.clear()
    def __str__(self):
        return "[simulator](states:{states})(weights:{weights})(neurons:{neurons})".format(states=self.nr_states, weights=self.nr_weights, neurons=len(self._neurons))

    def add_external(self, nr):
        ix = self.nr_states
        self.nr_states += nr
        return ix
    def add_neuron(self, transfer, inputs):
        output_ix, weight_ix = self.nr_states, self.nr_weights

        n = neuron.create(transfer, inputs, output_ix, weight_ix)
        self._neurons.append(n)

        self.nr_states += 1
        self.nr_weights += len(inputs)

        return (output_ix, weight_ix)

    def forward(self, states, weights, preacts=None):
        for n in self._neurons:
            n.forward(states, weights, preacts)
