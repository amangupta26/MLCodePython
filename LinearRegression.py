import random
import numpy as np

class LinearRegression:
	def __init__(self, N):
		self.N  = N
		self.generate_target()
		self.generate_training_data()
		self.get_weights()


	def generate_target(self):
		"""Get a random line"""
		xA, yA, xB, yB = [random.uniform(-1, 1) for x in range(4)]
		self.target = np.array([xB * yA - xA * yB, yB - yA, xA - xB])


	def generate_training_data(self):
		training_data_arr_x = []
		training_data_arr_y = []
		for i in range(self.N):
			x, y = [random.uniform(-1, 1) for x in range(2)]
			Q = np.array([1, x, y])
			s = int(np.sign(self.target.T.dot(Q)))
			training_data_arr_x.append([1, x, y])
			training_data_arr_y.append(s)

		# print training_data_arr_x
		self.X = np.matrix(training_data_arr_x)
		self.Y = np.matrix(training_data_arr_y)


	def get_weights(self):
		pseudo_inverse = np.linalg.pinv(self.X)
		self.weights = pseudo_inverse * self.Y.T

	def get_ein(self):
		output = self.X * self.weights
		output = np.sign(output)
		output = output.astype(int)
		matrix = output == self.Y.T
		return 1 - np.sum(matrix) / float(self.N)






