import numpy as np

class QuadraticCost:
	def __init__(self, stddev=1.0):
		self.target_ = None
		self.stddev = stddev

	@property
	def stddev(self):
		return self.stddev_
	@stddev.setter
	def stddev(self, v):
		self.stddev_ = v
		self.stddev2_ = v*v

	@property
	def target(self):
		return self.target_
	@target.setter
	def target(self, v):
		self.target_ = v

	def output(self, prediction):
		def diff2(p, t):
			d = p-t
			return d*d
		return sum(diff2(p, t) for p, t in zip(prediction, self.target))/self.stddev2_

	def gradient(self, prediction):
		return np.array([2*(p-t) for p, t in zip(prediction, self.target)])/self.stddev2_
