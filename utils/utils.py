import numpy as np
import scipy.stats as st


def cutCube(X, center, shape, padd=0): #center is a 3d coord (zyx)
    center = center.astype(int)
    hlz = np.round(shape[0] / 2)
    hly = np.round(shape[1] / 2)
    hlx = np.round(shape[2] / 2)

    #add padding if out of bounds
    if ((center - np.array([hlz,hly,hlx])) < 0).any() or (
        (center + np.array([hlz,hly,hlx]) + 1) > np.array(X.shape)).any():  # if cropping is out of bounds, add padding
        Xn = np.ones(np.array(X.shape) + shape * 2) * padd
        Xn[shape[0]:(shape[0] + X.shape[0]), shape[1]:(shape[1] + X.shape[1]), shape[2]:(shape[2] + X.shape[2])] = X
        centern = center + shape
        cube = Xn[int(centern[0] - hlz):int(centern[0] - hlz + shape[0]),
               int(centern[1] - hly):int(centern[1] - hly + shape[1]),
               int(centern[2] - hlx):int(centern[2] - hlx + shape[2])]
        return np.copy(cube)
    else:
        cube = X[int(center[0] - hlz):int(center[0] - hlz + shape[0]), int(center[1] - hly):int(center[1] - hly + shape[1]),
               int(center[2] - hlx):int(center[2] - hlx + shape[2])]
        return np.copy(cube)


def pasteCube(X, cube, center, padd=0):  #center is a 3d coord (zyx)
    center = center.astype(int)
    hlz = np.round(cube.shape[0] / 2)
    hlx = np.round(cube.shape[1] / 2)
    hly = np.round(cube.shape[2] / 2)
    Xn = np.copy(X)

    #add padding if out of bounds
    if ((center - hlz) < 0).any() or ((center - hlx) < 0).any() or ((center - hly) < 0).any()  or ((center + hlz + 1) > np.array(X.shape)).any() or ((center + hlx + 1) > np.array(X.shape)).any() or ((center + hly + 1) > np.array(X.shape)).any():  # if cropping is out of bounds, add padding
        Xn = np.ones(np.array(X.shape) + np.max(cube.shape) * 2) * padd
        Xn[cube.shape[0]:(cube.shape[0] + X.shape[0]), cube.shape[1]:(cube.shape[1] + X.shape[1]), cube.shape[2]:(cube.shape[2] + X.shape[2])] = X
        center = center + np.array(cube.shape)
        Xn[int(center[0] - hlz):int(center[0] - hlz + cube.shape[0]), int(center[1] - hlx):int(center[1] - hlx + cube.shape[1]),
        int(center[2] - hly):int(center[2] - hly + cube.shape[2])] = cube
        Xn = Xn[cube.shape[0]:(Xn.shape[0] - cube.shape[0]), cube.shape[1]:(Xn.shape[1] - cube.shape[1]), cube.shape[2]:(Xn.shape[2] - cube.shape[2])]
    else:
        Xn[int(center[0] - hlz):int(center[0] - hlz + cube.shape[0]), int(center[1] - hlx):int(center[1] - hlx + cube.shape[1]),
        int(center[2] - hly):int(center[2] - hly + cube.shape[2])] = cube
    return Xn

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def kern01(kernlen,nsig):
    k = gkern(kernlen, nsig)
    return (k - np.min(k)) / (np.max(k) - np.min(k))

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm