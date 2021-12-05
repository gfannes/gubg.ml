from gubg.neural import Network
from gubg.neural.transfer import Sigmoid, Linear
import numpy as np
from pytest import approx

def test_ctor():
	network = Network(2)
	network.add_layer(3, Sigmoid())
	network.add_layer(1, Linear())

	assert(np.linalg.norm(network.layers[0].activations - np.zeros(3)) == approx(0))
	assert(np.linalg.norm(network.layers[1].activations - np.zeros(1)) == approx(0))

	network.layers[1].bias = np.ones(1)
	network.forward(np.ones(2))
	assert(np.linalg.norm(network.layers[0].activations - np.ones(3)/2) == approx(0))
	assert(np.linalg.norm(network.layers[1].activations - np.ones(1)) == approx(0))

	network.reset_gradient()
	network.backward(np.ones(1))

	print(network)
