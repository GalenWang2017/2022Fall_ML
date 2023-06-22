import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize
from libsvm.svmutil import *

def readInput():
    trainImg = np.genfromtxt("X_train.csv", delimiter=",")
    trainLabel = np.genfromtxt("Y_train.csv", delimiter=",")
    testImg = np.genfromtxt("X_test.csv", delimiter=",")
    testLabel = np.genfromtxt("Y_test.csv", delimiter=",")
    return trainImg, trainLabel, testImg, testLabel

def linearKernel(u, v):
    return u.dot(v.T)

def polynomialKernel(u, v, gamma = 1/(28*28), coef0 = 0, degree = 3):
    return (gamma * u.dot(v.T) + coef0)**degree

def RBFKernel(u, v, gamma = 1/(28*28)):
    return np.exp(-gamma*distance.cdist(u, v, "sqeuclidean"))

def linearANDrbf(u, v, gamma = 1/(28*28)):
    return linearKernel(u, v) + RBFKernel(u, v, gamma)

def linearANDpolynomial(u, v, gamma = 1/(28*28), coef0 = 0, degree = 3):
    return linearKernel(u, v) + polynomialKernel(u, v, gamma, coef0, degree)

def polynomialANDrbf(u, v, gamma1 = 1/(28*28), coef0 = 0, degree = 3, gamma2 = 1/(28*28)):
    return polynomialKernel(u, v, gamma1, coef0, degree) + RBFKernel(u, v, gamma2)

def SVM(kernelType):
    startTime = time.time()
    if kernelType == 0:
        print("Linear:")
        model = svm_train(trainLabel, trainImg, "-t 0 -q")
    elif kernelType == 1:
        print("Ploynomial:")
        model = svm_train(trainLabel, trainImg, "-t 1 -q")
    elif kernelType == 2:
        print("RBF:")
        model = svm_train(trainLabel, trainImg, "-t 2 -q")
    p_label, p_acc, p_vals = svm_predict(testLabel, testImg, model, "-q")
    endTime = time.time()
    print("Predict accuracy: %.2f " %p_acc[0], "%")
    print("Elapsed time: %.4f s" %(endTime-startTime))
    print("--------------------------------------")
    return

def SVM_gridSearch(kernelType):
    n = 5
    hyperParameters = ""
    startTime = time.time()
    if kernelType == 0:
        print("Linear:")
        print()
        optC = 1.0
        maxAcc = 0.0
        for C in [0.001, 0.01, 0.1, 0.5, 1, 5]:
            print("hyper-parameters: C = ", C)
            para = "-t 0 -c " + str(C) +" -v " + str(n) +" -q"
            p_acc = svm_train(trainLabel, trainImg, para)
            if p_acc > maxAcc:
                maxAcc = p_acc
                optC = C
            print()
        print()
        print("Best hyper-parameters: C = ", optC)
        print("Accuracy in training data: %.2f " %maxAcc, "%")
        hyperParameters = "-t 0 -c " + str(optC) + " -q"
    elif kernelType == 1:
        print("Polynomial:")
        print()
        optC = 1.0
        optCoef0 = 0
        optDegree = 3
        optGamma = 1/(28*28)
        maxAcc = 0.0
        for C in [0.1, 0.5, 1, 5]:
            for Degree in [1, 2, 3, 4, 5]:
                for Coef0 in [-2, -1, 0, 1, 2]:
                    for gamma in [1/(28*28), 0.01, 0.1, 1, 2]:
                        print("hyper-parameters: C = ", C, ", Degree = ", Degree, ", Coef0 = ", Coef0, ", Gamma = ", gamma)
                        para = "-t 1 -c " + str(C) + \
                            " -g " + str(gamma) +\
                            " -d " + str(Degree) + \
                            " -r "+ str(Coef0) + \
                            " -v " + str(n) +" -q"
                        p_acc = svm_train(trainLabel, trainImg, para)
                        if p_acc > maxAcc:
                            maxAcc = p_acc
                            optC = C
                            optCoef0 = Coef0
                            optDegree = Degree
                            optGamma = gamma
                        print()
        print()
        print("Best hyper-parameters: C = ", optC, ", Coef0 = ", optCoef0, ", Degree = ", optDegree, ", gamma = ", optGamma)
        print("Accuracy in training data: %.2f " %maxAcc, "%")
        hyperParameters = "-t 1 -c " + str(optC) + \
                            " -g " + str(optGamma) +\
                            " -d " + str(optDegree) + \
                            " -r "+ str(optCoef0) + \
                            " -q"
    elif kernelType == 2:
        print("RBF:")
        print()
        optC = 1.0
        optGamma = 1/(28*28)
        maxAcc = 0
        for C in [0.001, 0.01, 0.1, 0.5, 1, 5]:
            for gamma in [1/(28*28), 0.01, 0.1, 1, 2]:
                print("parameters: C = ", C, ", Gamma = ", gamma)
                para = "-t 2 -c " + str(C) +\
                    " -g " + str(gamma) +\
                    " -v " + str(n) + " -q"
                p_acc = svm_train(trainLabel, trainImg, para)
                if p_acc > maxAcc:
                    optC = C
                    optGamma = gamma
                    maxAcc = p_acc
                    hyperParameters = para
                print()
        print()
        print("Best hyper-parameters: C = ", optC, ", gamma = ", optGamma)
        print("Accuracy in training data: %.2f " %maxAcc, "%")
        hyperParameters = "-t 2 -c " + str(optC) +\
                    " -g " + str(optGamma) + " -q"
    endTime = time.time()
    print("Elapsed time of training mdel: %.4f s" %(endTime-startTime))
    model = svm_train(trainLabel, trainImg, hyperParameters)
    rlt= svm_predict(testLabel, testImg, model, "-q")
    print("Predict accuracy in testing data: ", rlt[1][0])
    print("--------------------------------------")
    return

def SVM_precomputed():
    startTime = time.time()
    optC = 1.0
    optGamma = 1/(28*28)
    maxAcc = 0
    print("Linear + RBF:")
    print()
    for g in [1/(28*28), 0.01, 0.1, 1, 2]:
        precomputedX = linearANDrbf(trainImg, trainImg, g)
        precomputedX = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputedX))
        for C in [0.001, 0.01, 0.1, 0.5, 1, 5]:
            print("hyper-parameters: C = ", C, ", gamma = ", g)
            para = svm_parameter("-t 4 -c " + str(C) +" -v 5 -q")
            problem = svm_problem(trainLabel, precomputedX, isKernel = True)
            acc = svm_train(problem, para)
            if acc > maxAcc:
                optC = C
                optGamma = g
                maxAcc = acc
            print()
    endTime = time.time()
    hyperParamter = svm_parameter("-t 4 -c " + str(optC) +" -q")
    precomputedX = linearANDrbf(trainImg, trainImg, optGamma)
    precomputedX = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputedX))
    X_star = linearANDrbf(testImg, trainImg, optGamma)
    X_star = np.hstack((np.arange(1, 2500+1).reshape(-1, 1), X_star))
    problem = svm_problem(trainLabel, precomputedX, isKernel = True)
    model = svm_train(problem, hyperParamter)
    rlt = svm_predict(testLabel, X_star, model, "-q")
    print("Best hyper-parameters: C = ", optC, ", gamma = ", optGamma)
    print("Accuracy in training data: %.2f " %maxAcc, "%")
    print("Elapsed time of training mdel: %.4f s" %(endTime-startTime))
    print("Predict accuracy in testing data: ", rlt[1][0])
    print("--------------------------------------")
    return

def usingLinearAndPoly():
    startTime = time.time()
    optC = 1.0
    optGamma = 1/(28*28)
    optCoef0 = 0
    optDegree = 3
    maxAcc = 0
    print("Linear + Polynomial:")
    print()
    for c in [0.1, 0.5, 1, 5]:
        for g in [1/(28*28), 0.01, 0.1, 1, 2]:
            for d in [1, 2, 3, 4, 5]:
                for c0 in [-2, -1, 0, 1, 2]:
                    precomputed_X = linearANDpolynomial(trainImg, trainImg, g, c0, d)
                    precomputed_X = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputed_X))
                    para = svm_parameter("-t 4 -c " + str(c) + " -v 5 -q")
                    problem = svm_problem(trainLabel, precomputed_X, isKernel = True)
                    acc = svm_train(problem, para)
                    if acc > maxAcc:
                        optC = c
                        optGamma = g
                        optDegree = d
                        optCoef0 = c0
                        maxAcc = acc
                    print()
    endTime = time.time()
    hyperParamter = svm_parameter("-t 4 -c " + str(optC) +" -q")
    precomputedX = linearANDpolynomial(trainImg, trainImg, optGamma)
    precomputedX = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputedX))
    X_star = linearANDpolynomial(testImg, trainImg, optGamma)
    X_star = np.hstack((np.arange(1, 2500+1).reshape(-1, 1), X_star))
    problem = svm_problem(trainLabel, precomputedX, isKernel = True)
    model = svm_train(problem, hyperParamter)
    rlt = svm_predict(testLabel, X_star, model, "-q")
    print("Best hyper-parameters: C = ", optC, ", gamma = ", optGamma, ", Coef0 = ", optCoef0, ", degree = ", optDegree)
    print("Accuracy in training data: %.2f " %maxAcc, "%")
    print("Elapsed time of training mdel: %.4f s" %(endTime-startTime))
    print("Predict accuracy in testing data: ", rlt[1][0])
    print("--------------------------------------")
    return

def usingPolyAndRBF():
    startTime = time.time()
    optC = 1.0
    optGamma1 = 1/(28*28)
    optGamma2 = 1/(28*28)
    optCoef0 = 0
    optDegree = 3
    maxAcc = 0
    print("Polynomial + RBF:")
    print()
    for c in [0.1, 0.5, 1, 5]:
        for g1 in [1/(28*28), 0.01, 0.1, 1]:
            for d in [1, 2, 3, 4, 5]:
                for c0 in [-1, 0, 1]:
                    for g2 in [1/(28*28), 0.01, 0.1, 1]:
                        precomputed_X = polynomialANDrbf(trainImg, trainImg, g1, c0, d, g2)
                        precomputed_X = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputed_X))
                        para = svm_parameter("-t 4 -c " + str(c) + " -v 5 -q")
                        problem = svm_problem(trainLabel, precomputed_X, isKernel = True)
                        acc = svm_train(problem, para)
                        if acc > maxAcc:
                            optC = c
                            optGamma1 = g1
                            optGamma2 = g2
                            optDegree = d
                            optCoef0 = c0
                            maxAcc = acc
                        print()
    endTime = time.time()
    hyperParamter = svm_parameter("-t 4 -c " + str(optC) +" -q")
    precomputedX = polynomialANDrbf(trainImg, trainImg, optGamma1, optCoef0, optDegree, optGamma2)
    precomputedX = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), precomputedX))
    X_star = polynomialANDrbf(testImg, trainImg, optGamma1, optCoef0, optDegree, optGamma2)
    X_star = np.hstack((np.arange(1, 2500+1).reshape(-1, 1), X_star))
    problem = svm_problem(trainLabel, precomputedX, isKernel = True)
    model = svm_train(problem, hyperParamter)
    rlt = svm_predict(testLabel, X_star, model, "-q")
    print("Best hyper-parameters: C = ", optC, ", gamma of ploy = ", optGamma1, ", Coef0 = ", optCoef0, ", degree = ", optDegree, ", gamma of rbf = ", optGamma2)
    print("Accuracy in training data: %.2f " %maxAcc, "%")
    print("Elapsed time of training mdel: %.4f s" %(endTime-startTime))
    print("Predict accuracy in testing data: ", rlt[1][0])
    print("--------------------------------------")
    return

if __name__ == "__main__":
    trainImg, trainLabel, testImg, testLabel = readInput()

    # part 1
    print("----------------Part 1----------------")
    SVM(0) # using linear kernel
    SVM(1) # using polynomial kernel
    SVM(2) # using rbf kernel
    print()
    # -----------------------
    # part 2
    print("----------------Part 2----------------")
    SVM_gridSearch(0) # grid search in linear kernel
    SVM_gridSearch(1) # grid search in polynomial kernel
    SVM_gridSearch(2) # grid search in rbf kernel
    print()
    # -----------------------
    # part 3
    print("----------------Part 3----------------")
    SVM_precomputed() # using linear and rbf kernel to create a new kernel function
    usingLinearAndPoly()
    usingPolyAndRBF()