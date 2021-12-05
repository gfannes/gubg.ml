import numpy as np
import io

class Layer:
	def __init__(self, nr_inputs, nr_outputs, transfer):
		self.weights = np.zeros((nr_outputs, nr_inputs))
		self.weights_gradient = np.zeros((nr_outputs, nr_inputs))
		self.bias = np.zeros(nr_outputs)
		self.bias_gradient = np.zeros(nr_outputs)

		self.transfer_ = transfer

		self.input = np.zeros(nr_inputs)
		self.input_error = np.zeros(nr_inputs)
		self.pre_activations = np.zeros(nr_outputs)
		self.activations = np.zeros(nr_outputs)

	def __str__(self):
		os = io.StringIO()
		os.write(f"[Layer]{{\n")
		os.write(f"  [Weights]{{\n{self.weights}}}\n")
		os.write(f"  [Bias]{{\n{self.bias}}}\n")
		os.write(f"  [WeightsGradient]{{\n{self.weights_gradient}}}\n")
		os.write(f"  [BiasGradient]{{\n{self.bias_gradient}}}\n")
		os.write(f"}}")
		return os.getvalue()

	def nr_inputs(self):
		return self.weights.shape[1]
	def nr_outputs(self):
		return self.weights.shape[0]

	def forward(self, input):
		self.input = input
		self.pre_activations = self.weights.dot(input) + self.bias
		self.activations = self.transfer_.output(self.pre_activations)

	def reset_gradient(self):
		self.weights_gradient.fill(0)
		self.bias_gradient.fill(0)

	def backward(self, output_error):
		self.bias_gradient = output_error * self.transfer_.derivative(self.pre_activations)
		self.weights_gradient = self.bias_gradient.reshape(self.nr_outputs(), 1) @ self.input.reshape(1, self.nr_inputs())
		self.input_error = self.bias_gradient @ self.weights