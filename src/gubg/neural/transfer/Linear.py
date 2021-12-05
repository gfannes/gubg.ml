import math
import numpy as np

class Linear:
	def output(self, x):
		return x

	def derivative(self, x):
		return np.ones(np.shape(x))