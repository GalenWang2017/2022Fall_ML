import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.distance import *
import matplotlib.animation as animation
import math
import imageio.v2 as imageio
import time

K = 4
data_points = 100*100
img_size = 100
colors = [[0.85, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.0, 0.85], [0.9, 0.9, 0.0], [0.9, 0.5, 0]]
ccolors = [[240, 0, 0], [0, 240, 0], [0, 0, 240], [250, 250, 0], [250, 128, 0]]
kernelKmeansFile = "part3kernelKmeans3Img1.gif"
ratioFile = "part2ratioCut4Img1.gif"
normalFile = "part2normalizedCut4Img1.gif"


def showEigenSpace(eigen, alpha, k, cut):
    plt.clf()
    cluster = np.zeros(data_points, dtype = int)
    for i in range(data_points):
        for c in range(k):
            if alpha[i][c] == 1:
                cluster[i] = int(c)
    for i in range(len(eigen)):
            plt.scatter(eigen[i][0], eigen[i][1], c= ccolors[cluster[i]])
    if cut == "ratio":
        plt.savefig("eigenRatio.png")
    elif cut == "normalized":
        plt.savefig("eigenNormal.png")
    plt.show()
    return

def dist(p1, p2):
    return (p1-p2)**2

def generateImg(alpha, k):
    i = np.zeros((img_size, img_size, 3))
    for j in range(data_points):
        for c in range(k):
            if alpha[j][c] == 1:
                i[j//img_size][j%img_size] = colors[c]
    return i

def generateGif(imgs, fileName):
    plots = []
    it = 0
    for i in imgs:
        for f in range(8): # set number of frames for gif
            plt.imshow(i)
            plt.savefig(str(it)+str(f)+".png")
            plt.axis("off")
            plt.title("iteration "+str(it+1))
            plots.append(str(it)+str(f)+".png")
        it += 1
    with imageio.get_writer(fileName, mode='I') as writer:
        for i in plots:
            image = imageio.imread(i)
            writer.append_data(image)
    for i in plots:
        os.remove(i)
    return

def S():
    s = np.zeros((data_points, 2))
    for i in range(data_points):
        s[i][0] = i // 100
        s[i][1] = i % 100
    return s

def C(img):
    c = np.zeros((data_points, 3))
    for i in range(data_points):
        c[i] = img[i]*255
    return c

def kernelFunc(img, gammaS = 0.5, gammaC= 0.3):
    # return Gram matrix
    s = S()
    sd = cdist(s, s, 'sqeuclidean')
    c = C(img)
    cd = cdist(c, c, 'sqeuclidean')
    k = np.exp(-gammaS*sd) @ np.exp(-gammaC*cd)
    return k

# initial step in kernel K means
def initFunc(method = "rand", k = 2):
    # initial step in K means
    # alpha: 1, if this data point in cluster k; 0, otherwise
    # CenterK: the K centers for each k clusters
    # cK: # of data points in cluster k
    # muK: mu for each clusters
    alpha = np.zeros((data_points, k))
    if method == "rand":
        # randomly select k points as centers
        CenterK = np.random.randint(0, data_points, k)
        t = 0
        for i in CenterK:
            alpha[i][t] = 1
            t = t + 1
        cK = np.ones(k)
        muK = np.zeros(k, dtype=float)
        for c in range(k):
            muK[c] += alpha[CenterK[c]][c]*Kernel[CenterK[c]][CenterK[c]]
        for i in range(k):
            muK[i] = muK[i] / cK[i]
    elif method == "kernel++":
        CenterK = []
        CenterK.append(np.random.randint(data_points))
        for c in range(k-1):
            d = 0.0
            p = np.zeros((data_points), dtype = float)
            for i in range(data_points):
                t = np.inf
                for center in CenterK:
                    if math.sqrt((i//img_size - center//img_size)**2 + (i%img_size - center%img_size)**2) < t:
                        t = math.sqrt((i//img_size - center//img_size)**2 + (i%img_size - center%img_size)**2)
                d += t
                p[i] = t
            p /= d
            for i in range(1, data_points):
                p[i] += p[i-1]
            prob = np.random.rand()
            for i in range(1, data_points):
                if prob >= p[i-1] and prob < p[i]:
                    CenterK.append(i)
                    break
                t = 0
        for i in CenterK:
            alpha[i][t] = 1
            t = t + 1
        cK = np.ones(k)
        muK = np.zeros(k, dtype=float)
        for c in range(k):
            muK[c] += alpha[CenterK[c]][c]*Kernel[CenterK[c]][CenterK[c]]
        for i in range(k):
            muK[i] = muK[i] / cK[i]
    return CenterK, muK, cK, alpha

# kernel K means E step and M step
def kernelKMeans(img, muK, cK, alpha, k = K):
    numItr = 0
    old_alpha = alpha
    old_muK = muK
    old_cK = cK
    imgs = []
    print(old_cK)
    while True:
        # E step
        d = np.full((data_points, 2), np.inf)
        k_pq = np.zeros(k)
        for c in range(k):
            for p in range(data_points):
                for q in range(data_points):
                    k_pq[c] += (old_alpha[p][c]*old_alpha[q][c]*Kernel[p][q])
        for i in range(data_points):
            for c in range(k):
                t1 = 0.0
                for n in range(data_points):
                    t1 += (old_alpha[n][c]*Kernel[i][n])
                t = Kernel[i][i] - (2*t1/(old_cK[c]+1)) + (k_pq[c]/((old_cK[c]+1)**2))
                if t < d[i][1]:
                    d[i][1] = t
                    d[i][0] = c
        new_alpha = np.zeros((data_points, k))
        for i in range(data_points):
            new_alpha[i][int(d[i][0])] = 1
        
        # M step
        new_muK = np.zeros(k, dtype = float)
        new_cK = np.sum(new_alpha, axis = 0)
        for c in range(k):
            for i in range(data_points):
                new_muK[c] = new_muK[c] + alpha[i][c]*Kernel[i][i]
        for c in range(k):
            new_muK[c] = new_muK[c] / (new_cK[c]+1)
        imgs.append(generateImg(new_alpha, k))
        print(new_cK)
        if np.linalg.norm(new_muK-old_muK, 2) <= 0.01:
            break
        numItr += 1
        old_alpha = new_alpha
        old_cK = new_cK
        old_muK = new_muK
        
    generateGif(imgs, kernelKmeansFile)
    generateImg(old_alpha, k)
    print("Number of iterations: ", numItr+1)
    return

def Laplacian(W):
    D = np.zeros((data_points, data_points))
    for i in range(data_points):
        D[i][i] = np.sum(W[i])
    L = D-W
    return D, L

def normalizedLaplacian(D, L):
    sqrtD = np.sqrt(D)
    for i in range(data_points):
        sqrtD[i][i] = 1/sqrtD[i][i]
    Lsym = np.matmul(np.matmul(sqrtD, L), sqrtD)
    return Lsym

# spectral clustering
def spectralClustering(kernel, k, cut):
    # kernel: similarity matrix
    W = kernel
    D, L = Laplacian(W)
    if cut == "ratio":
        eigenVal, eigenVec = np.linalg.eig(L)
        # np.save("rev.npy", eigenVal)
        # np.save("revc.npy", eigenVec)
        eigenVal = np.load("rev.npy")
        eigenVec = np.load("revc.npy")
        idx = np.argsort(eigenVal)
        eigenVec = eigenVec[:][idx]
        U = eigenVec[:][0:k].T
        print(U.shape)
        imgs, alpha = Kmeans(U.real, k)
        generateGif(imgs, ratioFile)
        showEigenSpace(U.real, alpha, k, "ratio")

    elif cut == "normalized":
        Lsym = normalizedLaplacian(D, L)
        eigenVal, eigenVec = np.linalg.eig(Lsym)
        # np.save("nev.npy", eigenVal)
        # np.save("nevc.npy", eigenVec)
        eigenVal = np.load("nev.npy")
        eigenVec = np.load("nevc.npy")
        idx = np.argsort(eigenVal)
        eigenVec = eigenVec[:][idx]
        U = eigenVec[:][0:k].T
        T = np.copy(U.real)
        for i in range(data_points):
            for c in range(k):
                T[i][c] = T[i][c]/np.linalg.norm(U[i], 1)
        print(T.shape)
        imgs, alpha = Kmeans(T, k)
        generateGif(imgs, normalFile)
        showEigenSpace(U.real, alpha, k, "normalized")

    return 

def initKmeans(U, k, method = "rand"):
    r = np.zeros((data_points, k))
    if method == "rand":  
        mu = np.zeros(k)  
        C = np.ones(k)
        centers = np.random.randint(0, data_points, k)
        for c in range(k):
            mu[c] = U[centers[c]][c]
            r[centers[c]][c] = 1
    if method == "kernel++":
        centers = []
        centers.append(np.random.randint(data_points))
        for c in range(k-1):
            d = 0.0
            p = np.zeros((data_points), dtype = float)
            for i in range(data_points):
                t = np.inf
                for center in centers:
                    if math.sqrt((i//img_size - center//img_size)**2 + (i%img_size - center%img_size)**2) < t:
                        t = math.sqrt((i//img_size - center//img_size)**2 + (i%img_size - center%img_size)**2)
                d += t
                p[i] = t
            p /= d
            for i in range(1, data_points):
                p[i] += p[i-1]
            prob = np.random.rand()
            for i in range(1, data_points):
                if prob >= p[i-1] and prob < p[i]:
                    centers.append(i)
                    break
        t = 0
        for i in centers:
            r[i][t] = 1
            t = t + 1
        C = np.ones(k)
        mu = np.zeros(k, dtype=float)
        for c in range(k):
            mu[c] = U[centers[c]][c]
            r[centers[c]][c] = 1
    return C, mu, r

def Kmeans(U, k, method = "rand"):
    C, mu, r = initKmeans(U, k, method)
    new_r = np.zeros((data_points, k))
    imgs = []
    while True:
        # E step
        for i in range(data_points):
            temp_c = 0
            temp_d = np.inf
            for c in range(k):
                d = dist(U[i][c], mu[c])
                if d < temp_d:
                    temp_d = d
                    temp_c = c
            new_r[i][temp_c] = 1
        
        # M step
        new_C = np.sum(new_r, axis = 0)
        new_mu = np.zeros((k))
        for c in range(k):
            for i in range(data_points):
                new_mu[c] += new_r[i][c]*U[i][c]
        for c in range(k):
            new_mu[c] /= (new_C[c]+1)
        imgs.append(generateImg(new_r, k))
        
        C = new_C
        r = new_r
        if np.linalg.norm(new_mu-mu, 2) <= 0.01:
            break
        mu = new_mu
    return imgs, r

if __name__ == "__main__":
    
    # read which image file
    img = mpimg.imread("image1.png").reshape(-1, 3)
    # img = mpimg.imread("image2.png").reshape(-1, 3)

    t1 = time.time()
    Kernel = kernelFunc(img)

    # __CenterK__, __muK__, __cK__, __alpha__ = initFunc("kernel++", k = K)
    # kernelKMeans(img, __muK__, __cK__, __alpha__, K)
    # t2 = time.time()
    # print(t2-t1)

    t1 = time.time()
    spectralClustering(Kernel, K, "ratio")
    t2 = time.time()
    spectralClustering(Kernel, K, "normalized")
    t3 = time.time()
    print("ratio: ", t2-t1)
    print("normalized: ", t3-t2)
