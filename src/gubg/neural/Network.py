from gubg.neural import Layer
import io

class Network:
	def __init__(self, nr_inputs):
		self.nr_inputs_ = nr_inputs
		self.layers = []

	def __str__(self):
		os = io.StringIO()
		for layer in self.layers:
			print(layer, file=os)
		return os.getvalue()

	def add_layer(self, nr_neurons, transfer):
		nr_inputs = self.layers[-1].nr_outputs() if self.layers else self.nr_inputs_
		layer = Layer(nr_inputs, nr_neurons, transfer)
		self.layers.append(layer)

	def forward(self, input):
		for layer in self.layers:
			print(f"layer {input}")
			layer.forward(input)
			input = layer.activations

	def reset_gradient(self):
		for layer in self.layers:
			layer.reset_gradient()

	def backward(self, output_error):
		for layer in reversed(self.layers):
			layer.backward(output_error)
			output_error = layer.input_error
