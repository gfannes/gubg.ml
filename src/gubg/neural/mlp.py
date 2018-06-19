class Layer:
    def __init__(self, transfer, nr_inputs, nr_outputs):
        self.transfer = transfer
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
    def __str__(self):
        return "[layer](inputs:{inp})(outputs:{outp})(transfer:{tf})".format(tf=self.transfer, inp=self.nr_inputs, outp=self.nr_outputs)

class MLP:
    def __init__(self, nr_inputs):
        self.nr_inputs = nr_inputs
        self.layers = []
    
    def add_layer(self, transfer, nr_neurons):
        nr_inputs = self.layers[-1].nr_outputs if self.layers else self.nr_inputs
        self.layers.append(Layer(transfer, nr_inputs, nr_neurons))
    def setup_simulator(self, simulator):
        simulator.clear()

        self.bias = simulator.add_external(1)
        self.input0 = simulator.add_external(self.nr_inputs)

        input_ixs = [self.bias]+[i+self.input0 for i in range(self.nr_inputs)]

        for layer in self.layers:
            new_input_ixs = [self.bias]
            for i in range(layer.nr_outputs):
                output_ix, weight_ix = simulator.add_neuron(layer.transfer, input_ixs)
                new_input_ixs.append(output_ix)
            input_ixs = new_input_ixs
