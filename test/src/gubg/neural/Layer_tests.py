from gubg.neural.Layer import Layer
from gubg.neural.transfer import Sigmoid
import numpy as np
from pytest import approx

def test_ctor():
	layer = Layer(2, 3, Sigmoid())
	print(layer)

	assert(np.linalg.norm(layer.pre_activations - np.zeros(3)) == approx(0))
	assert(np.linalg.norm(layer.activations - np.zeros(3)) == approx(0))

	layer.forward([0, 0])
	assert(np.linalg.norm(layer.pre_activations - np.zeros(3)) == approx(0))
	assert(np.linalg.norm(layer.activations - np.array([0.5, 0.5, 0.5])) == approx(0))
