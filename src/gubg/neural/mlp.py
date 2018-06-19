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
