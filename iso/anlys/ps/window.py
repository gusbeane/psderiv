import numpy as np
from numba import njit

@njit
def window(x, R0, R1):

    Rm = (R0+R1)/2
    L = (R1-R0)/2
    
    W = np.zeros(x.shape)
    
    for i in range(len(x)):
        if x[i] < (R0+1e-12) or x[i] > (R1-1e-12):
            W[i] = 0.0
        else:
            W[i] = np.exp(-1/(1-((x[i]-Rm)/L)**2))
    
    return W

@njit
def grad_window(x, R0, R1):
    
    Rm = (R0+R1)/2
    L = (R1-R0)/2
    
    dW = np.zeros(x.shape)
    
    for i in range(len(x)):
        if x[i] < (R0+1e-12) or x[i] > (R1-1e-12):
            dW[i] = 0.0
        else:
            dW[i] = np.exp(-1/(1-((x[i]-Rm)/L)**2))
            dW[i] *= -2*L**2*(x[i]-Rm)
            dW[i] /= (L**2-(x[i]-Rm)**2)**2
    
    # ans[np.logical_or(x<R0, x>R1)] = 0
    # ans[np.logical_or(x<(R0+1e-12), x>(R1-1e-12))] = 0
    
    return dW

@njit
def grad2_window(x, R0, R1):
    
    Rm = (R0+R1)/2
    L = (R1-R0)/2
    
    ddW = np.zeros(x.shape)
    
    for i in range(len(x)):
        if x[i] < (R0+1e-12) or x[i] > (R1-1e-12):
            ddW[i] = 0.0
        else:
            ddW[i] = np.exp(-1/(1-((x[i]-Rm)/L)**2))
            ddW[i] *= -2 * L**2 * (L**4 - 3*(x[i]-Rm)**4)
            ddW[i] /= (L**2-(x[i]-Rm)**2)**4
    
    return ddW
