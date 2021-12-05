from gubg.neural.transfer import Sigmoid
from pytest import approx
import numpy as np

def test_sigmoid():
	sigmoid = Sigmoid()

	assert(sigmoid.output(-1000.0) == approx(0.0))
	assert(sigmoid.output(0) == approx(0.5))
	assert(sigmoid.output(1000.0) == approx(1.0))

	assert(sigmoid.output(0.1)-0.5 == approx(0.5-sigmoid.output(-0.1)))

	assert(sigmoid.derivative(-1000.0) == approx(0.0))
	assert(sigmoid.derivative(0) == approx(0.25))
	assert(sigmoid.derivative(1000.0) == approx(0.0))

	assert(sigmoid.output(np.zeros(3)).shape == (3,))
	assert(sigmoid.derivative(np.zeros(3)).shape == (3,))
	