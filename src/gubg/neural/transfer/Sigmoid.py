import numpy as np

class Sigmoid:
	def output(self, x):
		return 1.0/(1.0+np.exp(-x))

	def derivative(self, x):
		v = self.output(x)
		return v*(1.0-v)