from . import neuron

class Simulator:
    nr_states, nr_weights = 0, 0
    _neurons = []

    def add_external(self, nr):
        ix = self.nr_states
        self.nr_states += nr
        return ix

    def add_neuron(self, kind, inputs):
        output_ix, weight_ix = self.nr_states, self.nr_weights

        n = neuron.create(kind, inputs, output_ix, weight_ix)
        self._neurons.append(n)

        self.nr_states += 1
        self.nr_weights += len(inputs)

        return (output_ix, weight_ix)

    def forward(self, states, weights, preacts=None):
        for n in self._neurons:
            n.forward(states, weights, preacts)
