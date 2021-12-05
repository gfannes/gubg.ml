from gubg.model import QuadraticCost
import numpy as np
from pytest import approx

def test_quadratic_cost():
	cost = QuadraticCost(stddev=0.1)
	cost.target = [1, 2]
	assert(cost.output(cost.target) == approx(0.0))
	assert(cost.output([0, 0]) == approx(500.0))

	assert(np.linalg.norm(cost.gradient(cost.target) - np.array([0.0, 0.0]))   == approx(0.0))
	assert(np.linalg.norm(cost.gradient([0, 0]) - np.array([-200.0, -400.0])) == approx(0.0))

	cost.stddev = 1.0
	assert(cost.output(cost.target) == approx(0.0))
	assert(cost.output([0, 0]) == approx(5.0))

	assert(np.linalg.norm(cost.gradient(cost.target) - np.array([0.0, 0.0]))   == approx(0.0))
	assert(np.linalg.norm(cost.gradient([0, 0]) - np.array([-2.0, -4.0])) == approx(0.0))
