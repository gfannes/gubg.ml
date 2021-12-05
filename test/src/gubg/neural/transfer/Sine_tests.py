from gubg.neural.transfer import Sine
from pytest import approx
import math
import numpy as np

def test_sigmoid():
	sine = Sine()

	assert(sine.output(-math.pi) == approx(0.0))
	assert(sine.output(-math.pi/2) == approx(-1.0))
	assert(sine.output(0) == approx(0.0))
	assert(sine.output(math.pi/2) == approx(1.0))
	assert(sine.output(math.pi) == approx(0.0))

	assert(sine.derivative(-math.pi) == approx(-1.0))
	assert(sine.derivative(-math.pi/2) == approx(0.0))
	assert(sine.derivative(0) == approx(1.0))
	assert(sine.derivative(math.pi/2) == approx(0.0))
	assert(sine.derivative(math.pi) == approx(-1.0))

	assert(sine.output(np.zeros(3)).shape == (3,))
	assert(sine.derivative(np.zeros(3)).shape == (3,))
	