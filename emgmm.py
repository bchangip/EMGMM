import numpy as np
import matplotlib.pyplot as plt

CLUSTER_QUANTITY = 3

import random
from numpy import exp, matrix, transpose, subtract
from numpy.linalg import det, inv
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class EMGMM(object):
  def __init__(self, gaussiansQuantity, sourceFile, maxIterations):
    self.gaussiansQuantity = gaussiansQuantity
    self.samples = self.loadFile(sourceFile)

    # NSamples = 500.0

    # pi = np.array([0.2,0.5,0.3])
    # mus = ([.3,0.5],[2,4], [5,6])
    # sigmas = ([[.5,.3],[0,.6]], [[.8,-.3],[.5,.5]], [[.2,.2],[.2,.3]])

    # x1 = np.random.multivariate_normal(mus[0],sigmas[0],int(pi[0]*NSamples))
    # x2 = np.random.multivariate_normal(mus[1],sigmas[1],int(pi[1]*NSamples))
    # x3 = np.random.multivariate_normal(mus[2],sigmas[2],int(pi[2]*NSamples))

    # self.samples = np.vstack([x1,x2,x3])

    pis = np.ones(self.gaussiansQuantity) * 1.0 / self.gaussiansQuantity
    mus = self.samples[np.random.randint(0, len(self.samples), self.gaussiansQuantity)]
    sigma = np.array([[1, 0], [0, 1]]) * np.std(self.samples)
    self.gaussians = [
      {'pi': pis[i], 'sigma': sigma, 'mu': mus[i]}
      for i in range(len(pis))
    ]

  def loadFile(self, file):
    with open(file, 'r') as file:
      samples = file.readlines()
      # Remove brackets
      samples = [(sample).strip()[1:-1] for sample in samples]
      samples = [sample.split(', ') for sample in samples]
      return np.vstack([
        [float(sample[0]), float(sample[1])]
        for sample in samples
      ])
  
  def display(self):
    for sample in self.samples:
      plt.plot(sample[0], sample[1], 'k.')
    for gaussian in self.gaussians:
      plt.plot(gaussian['mu'][0], gaussian['mu'][1], 'ro')
    plt.show()

  def probability(self, sample, mu, sigma):
    probability = np.exp(-1*np.log(2*np.pi)) * np.power(np.linalg.det(sigma), -0.5) * np.exp(-0.5 * np.dot(np.dot((sample-mu).transpose(), np.linalg.inv(sigma)), (sample-mu)))
    if probability == 0:
      probability = 0.01
    return probability

  def expectation(self):
    probabilities = np.zeros((len(self.samples), self.gaussiansQuantity))
    for i in range(len(self.samples)):
      for j in range(self.gaussiansQuantity):
        probabilities[i, j] = self.gaussians[j]['pi'] * self.probability(self.samples[i], self.gaussians[j]['mu'], self.gaussians[j]['sigma'])
      probabilities[i] /= np.sum(probabilities[i])
    return probabilities

  def train(self):
    fig, ax = plt.subplots()
    plt.scatter(self.samples[:,0],self.samples[:,1],c='b',s=20,edgecolors='none')
    for l in range(self.gaussiansQuantity):
      plt.plot(self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1], 'ro')
      eigenvalues, eigenvectors = np.linalg.eig(self.gaussians[l]['sigma'])
      print('Gaussian', self.gaussians[l]['mu'])
      print('Angle', np.arctan2(np.linalg.norm(np.cross([1, 0], eigenvectors[0])), np.dot([1, 0], eigenvectors[0])))
      ax.add_artist(Ellipse((self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1]), 2*math.sqrt(eigenvalues[0]), 2*math.sqrt(eigenvalues[1]), np.degrees(np.arctan2(np.linalg.norm(np.cross([0, 1], eigenvectors[1])), np.dot([0, 1], eigenvectors[1]))), fill=False))
      ax.add_artist(Ellipse((self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1]), 4*math.sqrt(eigenvalues[0]), 4*math.sqrt(eigenvalues[1]), np.degrees(np.arctan2(np.linalg.norm(np.cross([0, 1], eigenvectors[1])), np.dot([0, 1], eigenvectors[1]))), fill=False))
    plt.show()

    for i in range(40):
      # print('Iteration', i)
      # for k in range(self.gaussiansQuantity):
      #   print('eigenvalues', np.linalg.eig(self.gaussians[k]['sigma']))
      probabilities = self.expectation()
      newPis = np.sum(probabilities, axis=0)/len(probabilities)
      newMus = zip(np.sum(probabilities*self.samples[:,0][:,np.newaxis],axis=0),np.sum(probabilities*self.samples[:,1][:,np.newaxis],axis=0))/(np.sum(probabilities,axis=0)[:,np.newaxis])
      tempSigmas = np.zeros([len(probabilities), self.gaussiansQuantity, 2, 2])
      for j in range(len(probabilities)):
        for k in range(self.gaussiansQuantity):
          tempSigmas[j, k] = probabilities[j, k] * np.dot((self.samples[j] - newMus[k])[:, np.newaxis], (self.samples[j] - newMus[k])[np.newaxis,:])
      newSigmas = np.sum(tempSigmas, axis=0) / np.sum(probabilities, axis=0)[:, np.newaxis, np.newaxis]
      self.gaussians = [
        {'pi': newPis[l], 'sigma': newSigmas[l], 'mu': newMus[l]}
        for l in range(len(newPis))
      ]
      print('Iteration', i)
      if (i % 5) == 0:
        fig, ax = plt.subplots()
        plt.scatter(self.samples[:,0],self.samples[:,1],c='b',s=20,edgecolors='none')
        for l in range(self.gaussiansQuantity):
          plt.plot(self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1], 'ro')
          eigenvalues, eigenvectors = np.linalg.eig(self.gaussians[l]['sigma'])
          # print('Gaussian', self.gaussians[l]['mu'])
          # print('Angle', np.arctan2(np.linalg.norm(np.cross([1, 0], eigenvectors[0])), np.dot([1, 0], eigenvectors[0])))
          ax.add_artist(Ellipse((self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1]), 2*math.sqrt(eigenvalues[0]), 2*math.sqrt(eigenvalues[1]), np.degrees(np.arctan2(np.linalg.norm(np.cross([0, 1], eigenvectors[1])), np.dot([0, 1], eigenvectors[1]))), fill=False))
          ax.add_artist(Ellipse((self.gaussians[l]['mu'][0], self.gaussians[l]['mu'][1]), 4*math.sqrt(eigenvalues[0]), 4*math.sqrt(eigenvalues[1]), np.degrees(np.arctan2(np.linalg.norm(np.cross([0, 1], eigenvectors[1])), np.dot([0, 1], eigenvectors[1]))), fill=False))
        plt.show()

emgmm = EMGMM(CLUSTER_QUANTITY, 'generatedPoints.txt', 10)
emgmm.train()

while True:
  inputPoint = input('Ingrese el punto a clasificar')
  inputPoint = [float(inputPoint[0]), float(inputPoint[1])]
  results = []
  for gaussian in emgmm.gaussians:
    results.append([gaussian['mu'], emgmm.probability(inputPoint, gaussian['mu'], gaussian['sigma'])])
  print('results', results)
