import numpy as np

class Sine:
	def output(self, x):
		return np.sin(x)

	def derivative(self, x):
		return np.cos(x)