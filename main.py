# import pandas as pd
import numpy as np
import matplotlib
import time
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

from IPython import embed

Dimensions = 2

A = np.array([[1, 1], [0, 1]]) # Matrix that maps previous state vector to new state vector 
B = np.array([[0, 0], [0, 1]]) # Matrix that maps control signal to state vector changes


# This is a vector of how the random variable (acceleration) effects each of the state variables
effectOnP = Dimensions * [0.5]
effectOnV = Dimensions * [1.]
for entry in effectOnV:
    effectOnP.append(entry)

r = effectOnP
vertical = [[entry] for entry in r]
horizontal = [r]
R = np.matmul(vertical, horizontal) # The covariance matrix of the noise (wind). Variance is 1 and so is ignored



print("R: {}".format(R))
# timestep is 1, so its left out of all equations

class GroundTruth:
    def __init__(self):
        self.position = np.zeros(Dimensions)
        self.velocity = np.zeros(Dimensions)
        self.positionHistory = []
        self.velocityHistory = []
        self.noiseHistory = []

        self.noiseMean = 0.
        self.noiseDeviation = 1.

    def GenerateNoise(self):
        return np.array(np.random.normal(loc=self.noiseMean, scale=self.noiseDeviation, size=(Dimensions)))

    def update(self):
        previousPosition = self.position
        previousVelocity = self.velocity

        self.positionHistory.append(previousPosition)
        self.velocityHistory.append(previousVelocity)

        # First update the velocity from the acceleration, then update the position
        acceleration = self.GenerateNoise()
        # self.velocity = previousVelocity + acceleration
        # self.position = previousPosition + self.velocity

        x_t_1 = np.array([previousPosition, previousVelocity + 0.5 * acceleration])
        x_t = np.matmul(A,x_t_1)
        # embed()
        # time.sleep(1)

        self.position = x_t[0]
        self.velocity = x_t[1]

        # print(x_t)
        # print("{:2.2f}, {:2.2f}, {:2.2f}".format(self.position[0], self.velocity[0], acceleration[0]))

        self.noiseHistory.append(acceleration)


class KalmanFilter:
    def __init__(self):
        self.position = np.zeros(Dimensions)
        self.velocity = np.zeros(Dimensions)
        self.positionHistory = []
        self.velocityHistory = []
        self.ut = np.array([np.zeros(Dimensions), np.zeros(Dimensions)]) # control command (none)
        self.noiseMean = np.zeros(2 * Dimensions) # position and velocity for n-dimensions

    def update(self, z):
        x_t_1 = np.array([self.position, self.velocity])

        stateUpdate = np.matmul(A,x_t_1)
        controlUpdate = np.matmul(B, self.ut)
        epsilon = np.random.multivariate_normal(self.noiseMean, R) #TODO: what is R for dim > 1
        # transpose epsilon so shape makes sense for all dimensions
        # rows = 1 if (len(epsilon.shape) == 1) else epsilon.shape[1] # edge case for 1d arrays 
        # cols = epsilon.shape[0]
        epsilon = epsilon.reshape(2, Dimensions)


        x_t = stateUpdate + controlUpdate + epsilon
        embed()
        time.sleep(1)
        self.position = x_t[0]
        self.velocity = x_t[1]
        self.positionHistory.append(self.position)
        self.velocityHistory.append(self.velocity)
       
        # embed()
        # time.sleep(1)

        # print(x_t)
        
        # belief = np.linalg.det(np.pi*2*R)

class GPS:
    def __init__(self):
        self.variance = 1
        self.C = np.array([1, 0])
        self.Q = np.array([[self.variance, 0], [0, self.variance]])
        self.measurementHistory = []
        self.noiseMean = np.zeros(Dimensions) # position and velocity for n-dimensions

    def measure(self, groundTruth):
        p = np.array([groundTruth.position, groundTruth.velocity]) # Don't need velocity but it keeps the dimensions in line with the books notation
        
        Cp = np.matmul(self.C, p)
    
        delta = np.random.normal(self.noiseMean, self.variance) if Dimensions == 1 else np.random.multivariate_normal(self.noiseMean, self.Q) 
        z = Cp + delta 
        self.measurementHistory.append(z)

def main():
    print("Start...")
    gt = GroundTruth()
    kf = KalmanFilter()
    z = GPS()

    for i in range(10):
        gt.update()
        z.measure(gt)
        kf.update(z.measurementHistory[-1])

        print("diff: ", abs(z.measurementHistory[-1]-gt.positionHistory[-1]))

    x = [entry[0] for entry in gt.positionHistory]
    y = [entry[1] for entry in gt.positionHistory]
    embed()
    x1 = [entry[0] for entry in kf.positionHistory] 
    y1 = [entry[1] for entry in kf.positionHistory]
    # y = z.measurementHistory


    t = [i for i in range(len(x))]
    # y = [entry[1] for entry in self.positionHistory] if (Dimensions > 1) else [i for i in range(len(x))]
    plt.plot(x, y, x1, y1, '--')
    plt.legend("Ground Truth", "Kalman Filter")
    plt.show()


    # x0 = gt.positionHistory
    # t = [i for i in range(len(x0))]
    # x1 = gt.velocityHistory
    # x2 = gt.noiseHistory
    
    # fig, ax1 = plt.subplots()
    # color = 'tab:red'
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('position', color=color)
    # ax1.plot(t, x0, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('velocity', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, x1, t, x2)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

if __name__ == "__main__":
    main()