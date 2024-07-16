from copy import deepcopy
from math import log, exp, sqrt
import sys
import warnings
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg

class KF: #칼만필터 클레스
    def __init__(self, n, m, k):
        self.n = n  # 추정하고자 하는 상태 갯수
        self.m = m  # 계측기 갯수
        self.k = k  # Time Step
        self.x_prev = np.zeros(n)  # 초기 상태값
        self.P_prev = 0.001 * np.eye(n)  # 초기 공분산

    def A_matrix(self):
        A = np.eye(self.n) # 대각행렬 생성
        A[0, 1] = self.k
        return A

    def H_matrix(self):
        # 측정 행렬 H를 정의합니다. 이는 상태를 측정값으로 변환합니다.
        H = np.zeros((self.m, self.n))
        H[0, 0] = 1
        return H

    def Estimation(self, z):
        kf = KalmanFilter(dim_x=self.n, dim_z=self.m)

        kf.x = self.x_prev  # 초기 상태
        kf.P = self.P_prev  # 초기 오차 공분산
        kf.A = self.A_matrix()  # 임의의 시스템 행렬
        kf.H = self.H_matrix()  # 임의의 측정 행렬
        kf.R = 0.01**2 * np.eye(self.m)  # 임의의 센서 노이즈 공분산
        kf.Q = 0.01**2 * np.eye(self.n)  # 임의의 시스템 노이즈 공분산

        kf.predict()
        kf.update(z)

        self.x_prev = kf.x
        self.P_prev = kf.P
        return kf.x
    
class KalmanFilter(object):
    r""" 
    References
    ----------

    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)

    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # 상태 행렬
        self.P = eye(dim_x)               # 오차 공분산 행렬
        self.Q = eye(dim_x)               # 시스템 노이즈
        self.B = None                     # 상태 전이(공간) 행렬
        self.F = eye(dim_x)               # 상태 전이(공간) 행렬
        self.H = zeros((dim_z, dim_x))    # Measurement function
        self.R = eye(dim_z)               # 센서 노이즈
        self._alpha_sq = 1.               # 패딩 메모리
        self.M = np.zeros((dim_z, dim_z)) 
        self.z = np.array([[None]*self.dim_z]).T


        self.K = np.zeros((dim_x, dim_z)) # 칼만 이득
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # 시스템 노이즈
        self.SI = np.zeros((dim_z, dim_z))

        # 행렬 정의
        self._I = np.eye(dim_x)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv


    def predict(self, u=None, B=None, F=None, Q=None):

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # 추정값 예측
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # 오차 공분산 예측
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


    def update(self, z, R=None, H=None):
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        z = self.reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H

        self.y = z - dot(H, self.x)

        PHT = dot(self.P, H.T)

        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # 칼만 이득 계산
        self.K = dot(PHT, self.SI)
        # 추정값 계산
        self.x = self.x + dot(self.K, self.y)

        # 오차 공분산 계산
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
    def reshape_z(self, z, dim_z, ndim):
        z = np.atleast_2d(z)
        if z.shape[1] == dim_z:
            z = z.T

        if z.shape != (dim_z, 1):
            raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

        if ndim == 1:
            z = z[:, 0]

        if ndim == 0:
            z = z[0, 0]
        return z