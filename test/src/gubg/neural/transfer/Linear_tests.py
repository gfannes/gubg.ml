from gubg.neural.transfer import Linear
import numpy as np
from pytest import approx

def test_ctor():
	linear = Linear()

	input = np.array([0,1,2])
	output = linear.output(input)

	assert(np.linalg.norm(output-input) == approx(0))

	derivative = linear.derivative(input)
	derivative_exp = np.ones(3)
	assert(np.shape(derivative) == derivative_exp.shape)
	assert(np.linalg.norm(derivative-derivative_exp) == approx(0))

	assert(linear.output(np.zeros(3)).shape == (3,))
	assert(linear.derivative(np.zeros(3)).shape == (3,))
