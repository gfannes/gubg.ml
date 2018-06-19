class Neuron:
    def __init__(self, input_ixs, weight_offset):
        self.input_ixs = input_ixs
        self.weight_offset = weight_offset

class Layer:
    def __init__(self, transfer, nr_inputs, nr_neurons):
        self.transfer = transfer
        self.nr_inputs = nr_inputs
        self.nr_neurons = nr_neurons
        self.neurons = []
    def __str__(self):
        return "[layer](inputs:{inp})(outputs:{outp})(transfer:{tf})".format(tf=self.transfer, inp=self.nr_inputs, outp=self.nr_neurons)

class MLP:
    def __init__(self, nr_inputs):
        self.nr_inputs = nr_inputs
        self.layers = []
    
    def add_layer(self, transfer, nr_neurons):
        nr_inputs = self.layers[-1].nr_neurons if self.layers else self.nr_inputs
        self.layers.append(Layer(transfer, nr_inputs, nr_neurons))
    def setup_simulator(self, simulator):
        simulator.clear()

        self.bias = simulator.add_external(1)
        self.input_offset = simulator.add_external(self.nr_inputs)

        input_ixs = [i+self.input_offset for i in range(self.nr_inputs)]+[self.bias]

        for layer in self.layers:
            new_input_ixs = []
            for i in range(layer.nr_neurons):
                output_ix, weight_offset = simulator.add_neuron(layer.transfer, input_ixs)
                neuron = Neuron(input_ixs, weight_offset)
                layer.neurons.append(neuron)
                new_input_ixs.append(output_ix)
            input_ixs = new_input_ixs+[self.bias]
