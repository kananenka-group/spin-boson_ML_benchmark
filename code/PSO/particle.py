import copy
import numpy as np
import random
class Particle:
  def __init__(self, p_error, ranges):
    # initialize position of the particle with 0.0 value
    dim = len(ranges)
    self.position = [ranges[i][0] + np.random.uniform(ranges[i][0], ranges[i][1]) for i in range(dim)]
    for i in range(dim):
    #  if i!=1:
      self.position[i] = int(self.position[i])

     # initialize velocity of the particle with 0.0 value
    self.velocity = [ranges[i][0] + np.random.uniform(ranges[i][0], ranges[i][1]/2) for i in range(dim)]

    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]

    # compute fitness of particle
    self.fitness = p_error(self.position, str(0) ) # curr fitness

    # initialize best position and fitness of this particle
    self.best_part_pos = [i for i in self.position]
    self.best_part_fitnessVal = self.fitness # best fitness
  def int_parameters(self):
    for i in range(len(self.position)):
      #if i!=1:
      self.position[i] = int(self.position[i])
      if self.position[i] == 0:
        self.position[i] += 1
