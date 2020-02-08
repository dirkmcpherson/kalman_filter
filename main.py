# import pandas as pd
import numpy as np
import time
import matplotlib
import os
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

from IPython import embed

DEBUG = False
Dimensions = 2
WindVariance = 1
WindMean = 0
GPSVariance = 0.1


if (Dimensions == 1):
    A = np.array([[1, 1], [0, 1]]) # Matrix that maps previous state vector to new state vector 
    B = np.array([[0, 0], [0, 1]]) # Matrix that maps control signal to state vector changes

elif (Dimensions == 2):
    A = np.array([[1,0,1,0],
                 [0,1,0,1],
                 [0,0,1,0],
                 [0,0,0,1]])
    B = np.array([[0,0,0,0],
                 [0,0,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])
    


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

# Wrapper for np.transpose that also converts a row vector to a column vector and vica versa
def customT(A):
    if len(A.shape) == 1:
        return A[:,None]
    else:
        return np.transpose(A)    

#utility function for A * B * A_transpose
def ABA_T(A, B):
    return np.matmul(np.matmul(A, B), customT(A))

class GroundTruth:
    def __init__(self):
        self.state = np.zeros(2*Dimensions)
        self.positionHistory = []
        self.velocityHistory = []
        self.noiseHistory = []

    def GenerateNoise(self):
        return np.array(np.random.normal(loc=WindMean, scale=np.sqrt(WindVariance), size=(Dimensions)))

    def position(self):
        return self.state[0] if Dimensions == 1 else self.state[0:2]

    def velocity(self):
        return self.state[1] if Dimensions == 1 else self.state[2:]

    def update(self):
        previousPosition = self.position()
        previousVelocity = self.velocity()

        self.positionHistory.append(previousPosition)
        self.velocityHistory.append(previousVelocity)

        # First update the velocity from the acceleration, then update the position
        acceleration = self.GenerateNoise()
        # self.velocity = previousVelocity + acceleration
        # self.position = previousPosition + self.velocity

        previousVelocity += 0.5 * acceleration

        if (Dimensions == 1):
            x_t_1 = np.array([previousPosition, previousVelocity])
        else:
            x_t_1 = np.array([previousPosition[0], previousPosition[1], previousVelocity[0], previousVelocity[1]])
        # x_t_1 = np.array([previousPosition, previousVelocity + 0.5 * acceleration])
        x_t = np.matmul(A,x_t_1)

        self.state = x_t

        print(x_t)
        # print("{:2.2f}, {:2.2f}, {:2.2f}".format(self.position[0], self.velocity[0], acceleration[0]))

        self.noiseHistory.append(acceleration)


class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(2*Dimensions)
        self.positionHistory = []
        self.velocityHistory = []
        self.ut = np.zeros(2*Dimensions) # control command (none)
        self.noiseMean = np.zeros(2 * Dimensions) # position and velocity for n-dimensions
        self.covariance = np.identity(2*Dimensions) # nxn where n is the number of state variables

    def position(self):
        return self.state[0] if Dimensions == 1 else self.state[0:2]

    def velocity(self):
        return self.state[1] if Dimensions == 1 else self.state[2:]

    def update(self, z):
        # from measurement
        z_t = z.measurementHistory[-1]
        C = z.C
        Q = z.Q

        x_t_1 = self.state

        stateUpdate = np.matmul(A,x_t_1)
        controlUpdate = np.matmul(B, self.ut)
        # epsilon = np.random.multivariate_normal(self.noiseMean, R) #TODO: what is R for dim > 1
        # transpose epsilon so shape makes sense for all dimensions
        # rows = 1 if (len(epsilon.shape) == 1) else epsilon.shape[1] # edge case for 1d arrays 
        # cols = epsilon.shape[0]
        # epsilon = epsilon.reshape(2, Dimensions)

        mean_bar = stateUpdate + controlUpdate
        covariance_bar = ABA_T(A, self.covariance) + R# epsilon # Maybe this should be a square matrix

        CEpC_T_Q = 1/(ABA_T(C, covariance_bar) + Q) if Dimensions == 1 else np.linalg.inv(ABA_T(C, covariance_bar) + Q)
        K = np.matmul(np.matmul(covariance_bar, customT(C)), CEpC_T_Q) # Kalman gain
        if (Dimensions == 1):
            K = K[:,None]
        print("Kalman Gain: ", K)


        mean = mean_bar + np.matmul(K, (z_t - np.matmul(C, mean_bar))) 
        KC = np.multiply(K,C) if Dimensions == 1 else np.matmul(K,C)
        I = np.identity(2*Dimensions)
        covariance = np.matmul((I - KC), covariance_bar)

        if (DEBUG):
            embed()
            time.sleep(1)

        self.covariance = covariance
        self.state = mean
        # x_t = stateUpdate + controlUpdate + epsilon

        print(mean)

        self.state = mean
        self.positionHistory.append(self.position())
        self.velocityHistory.append(self.velocity())
    
class GPS:
    def __init__(self):
        if (Dimensions == 1):
            self.C = np.array([1, 0])
        else:
            self.C = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.Q = np.array([GPSVariance]) if Dimensions == 1 else np.array([[GPSVariance, 0], [0, GPSVariance]]) # for 2d
        self.measurementHistory = []
        self.noiseMean = np.zeros(Dimensions) # position and velocity for n-dimensions

    def measure(self, groundTruth):
        p = groundTruth.state # Don't need velocity but it keeps the dimensions in line with the books notation
        
        # embed()
        # time.sleep(1)
        Cp = np.matmul(self.C, p)
    
        delta = np.random.normal(self.noiseMean, np.sqrt(GPSVariance)) if Dimensions == 1 else np.random.multivariate_normal(self.noiseMean, self.Q) 
        z = Cp + delta 
        self.measurementHistory.append(z)

def main():
    print("Start...")
    gt = GroundTruth()
    kf = KalmanFilter()
    z = GPS()

    for i in range(100):
        gt.update()
        z.measure(gt)
        kf.update(z)

        # print("diff: ", abs(z.measurementHistory[-1]-gt.positionHistory[-1]))


    if (Dimensions == 2):
        x = [entry[0] for entry in gt.positionHistory]
        x1 = [entry[0] for entry in kf.positionHistory] 
        y = [entry[1] for entry in gt.positionHistory]
        y1 = [entry[1] for entry in kf.positionHistory]
        x2 = [entry[0] for entry in z.measurementHistory]
        y2 = [entry[1] for entry in z.measurementHistory]
        plt.plot(x, y, x1, y1, '--', x2, y2, 'k+')
        plt.ylabel("Y position")
        plt.xlabel("X position")

    else:
        x = [entry for entry in gt.positionHistory]
        x1 = [entry for entry in kf.positionHistory] 
        x2 = [entry for entry in z.measurementHistory]
        t = [i for i in range(len(x))]
        plt.plot(t, x, t, x1, '--', t, x2, 'k+')
    # embed()
    # y = z.measurementHistory


    # y = [entry[1] for entry in self.positionHistory] if (Dimensions > 1) else [i for i in range(len(x))]
    plt.legend(["Ground Truth", "Kalman Filter", "Measurements"])
    plt.savefig('results.png', bbox_inches='tight')
    from PIL import Image
    with Image.open('results.png') as img:
        img.show()
    # plt.show() # this crashes everything on OSX


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
    import sys
    if (len(sys.argv) > 1):
        DEBUG = sys.argv[1]
    main()