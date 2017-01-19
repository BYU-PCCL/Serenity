import numpy
import random

class Q:

    def __init__(self, min_val=0, max_val=100):
	self.min_val = min_val
	self.max_val = max_val

    def randint(self, name):
	return random.randint(self.min_val, self.max_val)

