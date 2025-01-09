import numpy as np
import math
import scipy as sp
from scipy import optimize

def Kalman(y, z, u):
    global _Q, _H, _Y, _Z, _X0, _P0, _A, _C, _U, _nZ, _nU, out_opt
    Y = y
    Z = z
    nZ = len(Z.T)
    param = np.zeros((6))
    # INITIAL STATES
    X0 = np.zeros((nZ, 1))
    # INITIAL STATE COVARIANCE
    P0 = np.zeros((nZ, nZ))  
    # TRANSITION MATRIX
    A = np.array([[1, 1], [0, 1]])  
    # STATE INNOVATION COVARIANCE MATRIX
    Q = np.zeros((nZ, nZ))    
    # SIGNAL COVARIANCE MATRIX
    H = np.zeros((1, 1))
    # CONTROL COEFFS  
    if np.isscalar(u) == 1:
        U = 0
        C = 0
        nU = 0
    else:
        U = u
        nU = len(U.T)
        C = np.zeros((nZ, nU))           
    # GLOBAL PARAMETERS
    _Q=Q; _H=H; _Y=Y; _Z=Z; _X0=X0; _P0=P0; _A=A; _nZ=nZ; _U=U; _C=C; _nU=nU     
    # OPTIMIZATION
    param_opt = optimize.fmin_bfgs(ml_Multi, param, gtol=0.001)  
    # ESTIMATION WITH OPTIMIZED PARAMETERS
    H = np.exp(param_opt[0])
    Q[0,0] = 0.5 * H
    Q[1,1] = 0.5 * Q[0,0]
    X0[0,0] = param_opt[3]
    P0[0,0] = np.exp(param_opt[4])
    P0[1,1] = np.exp(param_opt[5])
    (likely, XOUT, POUT) = kalman_filter(Y, Z, X0, P0, A, Q, H, U, C)
    return XOUT, POUT, param_opt


def ml_Multi(param):    
    global _Q, _H, _Y, _Z, _X0, _P0, _A, _nZ, _C, out_opt
    param = param.reshape(6)

    _H = np.exp(param[0])
    _Q[0, 0] = 0.5 * _H
    _Q[1, 1] = 0.5 * _Q[0, 0]
    _X0[0, 0] = param[3]
    _P0[0, 0] = np.exp(param[4])
    _P0[1, 1] = np.exp(param[5])
    (likely, XOUT, POUT) = kalman_filter(_Y, _Z, _X0, _P0, _A, _Q, _H, _U, _C)
    likely = np.reshape(likely, ((1)))
    return -likely


def kalman_filter(Y, Z, X0, P0, A, Q, H, U, C):
    nZ = len(Z.T)
    nT = len(Y)
    Xt = X0
    Pt = P0
    Zt = np.zeros((1, nZ))
    Xout = np.zeros((nT, nZ))
    Pout = np.zeros((nT, nZ, nZ))
    l1 = 0.0
    l2 = 0.0
    likely = 0.0
    #LOOP
    t = 0
    while t < nT:
        #estimation step
        if np.isscalar(U) == 1:
            Xte = A.dot(Xt)
        else:
            Xte = A.dot(Xt) + C.dot(U[t:t+1, :])
        Pte = A.dot(Pt).dot(A.T) + Q            
        # update
        Zt[:, :] = Z[t, :]
        vt = Y[t] - Zt.dot(Xte)
        Ft = Zt.dot(Pte).dot(Zt.T)+H
        Fti = np.linalg.inv(Ft)
        Xt = Xte + Pte.dot(Zt.T).dot(Fti).dot(vt)
        Pt = (np.eye(nZ) - Pte.dot(Zt.T).dot(Fti).dot(Zt)).dot(Pte)
        # storage
        Xout[t, :] = Xt.T
        Pout[t, :, :] = Pt
        l1 = l1 + np.log(np.linalg.det(Ft))
        l2 = l2 +  (vt.T).dot(Fti).dot(vt)
        t = t + 1
    likely = -0.5 * nT * np.log(2 * math.pi) - 0.5 * l1 - 0.5 * l2
    return likely/nT, Xout, Pout