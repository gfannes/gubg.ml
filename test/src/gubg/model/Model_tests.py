from gubg.model import Model
from gubg.model import QuadraticCost
from gubg.data import Set
from pytest import approx
import numpy as np

class QuadraticPredictor:
	def __init__(self):
		self.w1 = 1.0
		self.w2 = 1.0
		self.reset_gradient()

	def forward(self, input):
		s = self.w1*input[0] + self.w2*input[1]
		self.input_ = input
		return [s*s]

	def reset_gradient(self):
		self.gradient = np.zeros(2)

	def backward(self, cost_gradient):
		self.gradient[0] = cost_gradient[0]*self.input_[0]
		self.gradient[1] = cost_gradient[0]*self.input_[1]


def test_ctor():
	predictor = QuadraticPredictor()
	cost = QuadraticCost()
	input_names = "input1 input2".split(" ")
	target_names = ["target"]
	data = Set.create(input_names+target_names, [(0,0 ,0.1), (1,0, 1.1), (1,1 ,4.1)])
	m = Model(predictor, cost, data, input_names, target_names)

	assert(m.cost() == approx(data.item_count()*0.1*0.1))

	m.compute_gradient()