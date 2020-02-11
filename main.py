# import pandas as pd
import numpy as np
import time
import matplotlib
import os
import random
from PIL import Image
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from IPython import embed

DEBUG = False
Dimensions = 1
WindVariance = 1
WindMean = 0
GPSVariance = 8
simulation_iterations = 100


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
R = WindVariance * np.matmul(vertical, horizontal) # The covariance matrix of the noise (wind). Variance is 1 and so is ignored



# print("R: {}".format(R))
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

        # print(x_t)
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
        self.Kalman_gains = []

    def position(self):
        return self.state[0] if Dimensions == 1 else self.state[0:2]

    def velocity(self):
        return self.state[1] if Dimensions == 1 else self.state[2:]

    # Calculate the state distribution in terms of (mean, covariance)
    def stateDistribution_noMeasurement(self):
        x_t_1 = self.state
        stateUpdate = np.matmul(A,x_t_1)
        controlUpdate = np.matmul(B, self.ut)
        mean = stateUpdate + controlUpdate + np.random.multivariate_normal(self.noiseMean, R)
        covariance = ABA_T(A, self.covariance) + R

        return mean, covariance

    def measurementUpdate_noState(self, mean_bar, covariance_bar, z):
        # from measurement
        z_t = z.measurementHistory[-1]
        C = z.C
        Q = z.Q

        CEpC_T_Q = 1/(ABA_T(C, covariance_bar) + Q) if Dimensions == 1 else np.linalg.inv(ABA_T(C, covariance_bar) + Q)
        K = np.matmul(np.matmul(covariance_bar, customT(C)), CEpC_T_Q) # Kalman gain
        if (Dimensions == 1):
            K = K[:,None] # transpose 1-D (should be a matrix but its not so do it this way)

        self.Kalman_gains.append(K)

        mean = mean_bar + np.matmul(K, (z_t - np.matmul(C, mean_bar))) 
        KC = np.multiply(K,C) if Dimensions == 1 else np.matmul(K,C)
        I = np.identity(2*Dimensions)
        covariance = np.matmul((I - KC), covariance_bar)

        if (DEBUG):
            embed()
            time.sleep(1)

        self.covariance = covariance
        self.state = mean
        
        return mean, covariance

    def update(self, z):
        # from measurement
        # z_t = z.measurementHistory[-1]
        # C = z.C
        # Q = z.Q

        # x_t_1 = self.state
        # stateUpdate = np.matmul(A,x_t_1)
        # controlUpdate = np.matmul(B, self.ut)
        # mean_bar = stateUpdate + controlUpdate
        # covariance_bar = ABA_T(A, self.covariance) + R

        mean_bar, covariance_bar = self.stateDistribution_noMeasurement()

        mean, covariance = self.measurementUpdate_noState(mean_bar, covariance_bar, z)

        # CEpC_T_Q = 1/(ABA_T(C, covariance_bar) + Q) if Dimensions == 1 else np.linalg.inv(ABA_T(C, covariance_bar) + Q)
        # K = np.matmul(np.matmul(covariance_bar, customT(C)), CEpC_T_Q) # Kalman gain
        # if (Dimensions == 1):
        #     K = K[:,None] # transpose 1-D (should be a matrix but its not so do it this way)

        # self.Kalman_gains.append(K)

        # mean = mean_bar + np.matmul(K, (z_t - np.matmul(C, mean_bar))) 
        # KC = np.multiply(K,C) if Dimensions == 1 else np.matmul(K,C)
        # I = np.identity(2*Dimensions)
        # covariance = np.matmul((I - KC), covariance_bar)

        # if (DEBUG):
        #     embed()
        #     time.sleep(1)

        # self.covariance = covariance
        # self.state = mean
        # # x_t = stateUpdate + controlUpdate + epsilon

        # print(mean)

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

def p1_3():
    n = 5
    kf = KalmanFilter()
    means = [[0., 0.]]
    covariances = [np.identity(2)]
    for i in range(n):
        m, cv = kf.stateDistribution_noMeasurement()

        # save state and covariance
        kf.state = m
        kf.covariance = cv
        means.append(m)
        covariances.append(cv)

    fig, ax = plt.subplots()
    x = [entry[0] for entry in means]
    y = [entry[1] for entry in means]
    
    ax.scatter(x,y)

    normalizingConstant = 1.
    for i in range(len(x)):
        center = (x[i],y[i])

        eigenVals, eigenVectors = np.linalg.eig(covariances[i])
        # embed()
        # normalize eigenvalues
        stdDevUncertainty = 2 * np.sqrt(2.77)
        eigenVals = [stdDevUncertainty * entry/sum(eigenVals) for entry in eigenVals]

        width = eigenVals[0]
        height = eigenVals[1]
        largestEigenVector = eigenVectors[np.argmax(eigenVals)]
        angle = np.arctan(largestEigenVector[1]/ largestEigenVector[0])
        # print("i: ", i)
        # print("    " + str(eigenVals))
        # print("    " + str(eigenVectors))
        print("angle ", angle)
        np.angle(angle)
        ax.add_patch(Ellipse(center, width, height, angle=np.angle(angle), facecolor="None", edgecolor="red"))
    

    extend = 2
    xlims = (min(x)-extend, max(x) + extend)
    ylims = (min(y)-extend, max(y) + extend)
    ax.set(xlim=xlims,ylim=ylims)


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('acceleration', color=color)  # we already handled the x-label with ax1
    # ax2.plot([i for i in range(len(gt))], data2, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.ylabel("velocity")
    plt.xlabel("position")
    plt.title("1D State Distribution")
    # embed()
    # y = z.measurementHistory


    # y = [entry[1] for entry in self.positionHistory] if (Dimensions > 1) else [i for i in range(len(x))]
    plt.savefig('results.png', bbox_inches='tight')
    with Image.open('results.png') as img:
        img.show()

def p2_2():
    beforeMeasure = 5
    gt = GroundTruth()
    kf = KalmanFilter()
    gps = GPS()

    means = []
    covariances = []
    for i in range(beforeMeasure):
        m, cv = kf.stateDistribution_noMeasurement()
        kf.state = m
        kf.covariance = cv
        kf.positionHistory.append(m)
        means.append(m)
        covariances.append(cv)

        # gps.measure(gt)

    gps.measurementHistory.append([10.])
    # gps.measure(gt)
    mean, covariance = kf.measurementUpdate_noState(kf.state, kf.covariance, gps)
    means.append(mean)
    covariances.append(covariance)
    kf.positionHistory.append(m)

    print("Mean changed from {} to {}".format(means[-2], means[-1]))
    print("covariance from {} to {}".format(covariances[-2], covariances[-1]))
    fig, ax = plt.subplots()
    x = [entry[0] for entry in means]
    y = [entry[1] for entry in means]

    ax.scatter(x,y)
    plt.ylabel("velocity")
    plt.xlabel("position")
    plt.title("1D State Distribution")
    # plt.show()

def p2_3():
    p_fail = [0.1, 0.5, 0.9]
    numTrials = 100
    xs = []
    ts = []
    for failRate in p_fail: 
        failureRate = failRate
        allError = [[] for n in range(numTrials)]    
        for n in range(numTrials):
            gt = GroundTruth()
            kf = KalmanFilter()
            gps = GPS()

            for i in range(20):
                gt.update()

                if (random.random() > failureRate):
                    gps.measure(gt)
                    kf.update(gps)
                else:
                    m, cv = kf.stateDistribution_noMeasurement()
                    kf.state = m
                    kf.covariance = cv
                    kf.positionHistory.append(m[0])

                error = abs(gt.position() - kf.position())
                print("error", error)
                allError[n].append(error[0])

        # embed()
            

        a = np.array(allError)

        x = a.mean(0)
        t = [i for i in range(len(x))]
        xs.append(x)
        ts.append(t)
    
    # embed()
    plt.plot(ts[0],xs[0],ts[1],xs[1],ts[2],xs[2])
    plt.legend(["0.1","0.5","0.9"])
    plt.ylabel("error (ground truth - estimate)")
    plt.xlabel("timesteps")
    plt.title("Error over time for different sensor failure rates.")
    plt.savefig('results.png', bbox_inches='tight')
    with Image.open('results.png') as img:
        img.show()
    



def plotResults_1D(gt, kf, z):
    x = [entry for entry in gt.positionHistory]
    x1 = [entry[0] for entry in kf.positionHistory] 
    x2 = [entry for entry in z.measurementHistory]
    # while len(x2) < len(x):
    #     x2.insert(None, 0)

    t = [i for i in range(len(x))]
    embed()
    plt.figure(0)
    plt.plot(t, x, t, x1, '--')#, t, x2, 'k+')
    plt.ylabel("position")
    plt.xlabel("timestep")
    plt.title("1D Kalman Filter")
    # embed()
    # y = z.measurementHistory


    # y = [entry[1] for entry in self.positionHistory] if (Dimensions > 1) else [i for i in range(len(x))]
    plt.legend(["Ground Truth", "Kalman Filter", "Measurements"])
    plt.savefig('results.png', bbox_inches='tight')
    # with Image.open('results.png') as img:
    #     img.show()

    # plotKalmanGain(kf.Kalman_gains)
    plt.show()

    # embed()

def main():
    print("Start...")
    gt = GroundTruth()
    kf = KalmanFilter()
    z = GPS()

    for i in range(simulation_iterations):
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
        plt.title("2D Kalman Filter")

    else:
        x = [entry for entry in gt.positionHistory]
        x1 = [entry for entry in kf.positionHistory] 
        x2 = [entry for entry in z.measurementHistory]
        t = [i for i in range(len(x))]
        # embed()
        plt.plot(t, x, t, x1, '--', t, x2, 'k+')
        plt.ylabel("position")
        plt.xlabel("timestep")
        plt.title("1D Kalman Filter")
    # embed()
    # y = z.measurementHistory


    # y = [entry[1] for entry in self.positionHistory] if (Dimensions > 1) else [i for i in range(len(x))]
    plt.legend(["Ground Truth", "Kalman Filter", "Measurements"])
    plt.savefig('results.png', bbox_inches='tight')
    with Image.open('results.png') as img:
        img.show()

    # plotKalmanGain(kf.Kalman_gains)
    # plt.show() # this crashes everything on OSX

def plotKalmanGain(K):
    numPlots = max(K[0].shape)
    for i in range(numPlots):
        if (i==0):
            plt.title("Kalman Gains")
        plotNumber = 100 * numPlots + 10 + i + 1
        print(plotNumber)
        plt.subplot(plotNumber)
        x = [entry[i] for entry in K]
        plt.plot(x)

    plt.savefig('Kalman_gain_results.png', bbox_inches='tight')
    with Image.open('Kalman_gain_results.png') as img2:
        img2.show()

if __name__ == "__main__":
    import sys
    if (len(sys.argv) > 1):
        DEBUG = sys.argv[1]
    main()
    # p1_3()
    # p2_2()
    # p2_3()
