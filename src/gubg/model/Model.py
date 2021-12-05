class Model:
	def __init__(self, predictor, cost, data, input_names, target_names):
		self.predictor_ = predictor
		self.cost_ = cost
		self.data_ = data
		self.input_ixs = data.indices(input_names)
		self.target_ixs = data.indices(target_names)

	def cost(self):
		total_cost = 0.0
		for item in self.data_.items:
			self.cost_.target = [item[ix] for ix in self.target_ixs]

			input  = [item[ix] for ix in self.input_ixs]
			total_cost += self.cost_.output(self.predictor_.forward(input))

		return total_cost

	def compute_gradient(self):
		self.predictor_.reset_gradient()
		for item in self.data_.items:
			self.cost_.target = [item[ix] for ix in self.target_ixs]

			input  = [item[ix] for ix in self.input_ixs]
			cost_gradient = self.cost_.gradient(self.predictor_.forward(input))
			self.predictor_.backward(cost_gradient)
