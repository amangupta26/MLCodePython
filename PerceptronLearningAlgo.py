"""This is a class which implements Perceptron Learning Algorithm aka PLA
		

"""

import random
import numpy as np
import matplotlib.pyplot as plt

class PLA:
	def __init__(self, N, iterations):
		self.N = N
		self.iter = iterations
		self.generate_target()
		self.generate_training_data()

	def generate_target(self):
		"""Get a random line"""
		xA, yA, xB, yB = [random.uniform(-1, 1) for x in range(4)]
		self.target = np.array([xB * yA - xA * yB, yB - yA, xA - xB])

	def generate_training_data(self):
		self.training_data = []
		for i in range(self.N):
			x, y = [random.uniform(-1, 1) for x in range(2)]
			X = np.array([1, x, y])
			s = int(np.sign(self.target.T.dot(X)))
			self.training_data.append((X, s))


	def get_misclassified_count(self):
		count = 0
		for X,s in self.training_data:
			if (np.sign(self.weights.T.dot(X))) != s:
				count += 1
		return count


	def get_disagreement(self):
		count = 0
		for i in range(1000):
			x, y = [random.uniform(-1, 1) for x in range(2)]
			X = np.array([1, x, y])
			s1 = int(np.sign(self.target.T.dot(X)))
			s2 = int(np.sign(self.weights_g.T.dot(X)))
			if s1 != s2:
				count += 1
		return count / (1000.0)


	def get_misclassified_point(self):
		mis_pts = []
		for X,s in self.training_data:
			if (np.sign(self.weights.T.dot(X))) != s:
				mis_pts.append((X, s))
		if len(mis_pts) == 0:
			return [None, None]
		return mis_pts[random.randrange(0, len(mis_pts))]


	def run_pla(self):
		temp = self.N + 1
		self.mis_count = 0
		for i in range(1000):
			self.weights = np.zeros(3)
			for i in range(self.iter):
				X, s = self.get_misclassified_point()
				if X == None:
					break
				self.weights += s*X
			temp1 = self.get_misclassified_count()
			self.mis_count += temp1
			if temp1 < temp:
				self.weights_g = self.weights
				temp = temp1

		self.mis_count = self.mis_count / (1000.0)
		self.disagreement = self.get_disagreement()
		print self.mis_count, self.disagreement
					

	def traing_error(self):
		pass

	def test_error(self):
		pass


