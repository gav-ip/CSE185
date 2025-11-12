from scipy import linalg
import numpy as np

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self, P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center

    def project(self, X):
        """ Project points in X (4*n array) and normalize coordinates. """
        # Project the points using dot product
        x = np.dot(self.P, X)

        # Normalize the projected points
        # Divide first two coordinates by the third coordinate (homogeneous normalization)
        x[0] /= x[2]
        x[1] /= x[2]
        x[2] /= x[2]  # This will be 1 after normalization

        return x

    def factor(self):
        """ Factorize the camera matrix into K, R, t. """
        # factor the first 3*3 part of P
        K, R = linalg.rq(self.P[:, :3])

        # create a diagonal matrix T to ensure the diagonal of K is positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)

        # compute translation t
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t


def rotation_matrix(a):
    """ Creates a 3D rotation matrix for rotation around the axis of the vector a. """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R

