import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize

def readInputData():
    inputdata = np.genfromtxt("input.data")
    X = inputdata[:, 0].reshape(-1, 1)
    Y = inputdata[:, 1].reshape(-1, 1)
    return X, Y
def kernelMethod(x, x_star, alpha, l):
    # rational quadratic kernel
    #                    |x-x_star|^2
    #    sigma^2 * (1 + ---------------)^(-alpha)
    #                    2 * alpha * l
    # where sigma is fixed to 1, x and x_star is vertor
    return (1+(distance.cdist(x, x_star, "sqeuclidean")/(2*alpha*(l**2))))**(-alpha) 

def marginalLogLikelihood(theta):
    C_theta = kernelMethod(X, X, theta[0], theta[1]) + (1/beta)*np.identity(X.shape[0])
    rlt = (-0.5)*np.log(np.linalg.det(C_theta))-0.5*Y.T.dot(np.linalg.inv(C_theta).dot(Y))-(X.shape[0]/2)*np.log(2*np.pi)
    return 0-rlt[0]

class GaussianProcess():
    def __init__(self, alpha = 1.0, l = 1.0, beta = 5.0):
        self.alpha = alpha
        self.l = l
        self.beta = beta
        pass
    def fit(self, X, Y, X_star):
        C = kernelMethod(X, X, self.alpha, self.l) + (1/beta)*np.identity(X.shape[0])
        k = kernelMethod(X, X_star, self.alpha, self.l)
        k_star = kernelMethod(X_star, X_star, self.alpha, self.l) + (1/beta)

        mean = k.T.dot(np.linalg.inv(C).dot(Y))
        var = k_star - k.T.dot(np.linalg.inv(C).dot(k))
        return mean, var
    def figResult(self, X, Y, X_star, mean, var):
        plt.xlim(-60, 60)
        plt.scatter(X, Y, color='blue')
        plt.plot(X_star, mean, 'r')
        upper = np.zeros(mean.shape[0])
        lower = np.zeros(mean.shape[0])
        for i in range(mean.shape[0]):
            upper[i] = mean[i] + 1.96 * var[i, i]
            lower[i] = mean[i] - 1.96 * var[i, i]
        plt.fill_between(X_star.ravel(), upper.ravel(), lower.ravel(), color = "red", alpha = 0.15)
        plt.show()
        return

if __name__ == "__main__":
    numPoints = 1000
    beta = 5
    X, Y = readInputData()
    X_star = np.linspace(-60, 60, numPoints).reshape(-1, 1) # let X_star is 1000 by 1
    t1 = time.time()
    gp = GaussianProcess(alpha=1.0, l = 1.0, beta=beta)
    mean_gp, var_gp = gp.fit(X, Y, X_star)
    elapsed = time.time() - t1
    print("The time to do first Gaussian process: ", elapsed)
    # plot the result
    gp.figResult(X, Y, X_star, mean_gp, var_gp)
    t2 = time.time()
    optTheta = minimize(marginalLogLikelihood, x0=[1.0, 1.0])
    optGP = GaussianProcess(optTheta.x[0], optTheta.x[1], beta=beta)
    mean_gp, var_gp = optGP.fit(X, Y, X_star)
    elapsed = time.time() - t2
    print("alpha = ", optTheta.x[0])
    print("    l = ", optTheta.x[1])
    print("The time to find optimal theta and do Gaussian process: ", elapsed)
    optGP.figResult(X, Y, X_star, mean_gp, var_gp)