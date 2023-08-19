import numpy as np
import math

MAX_RANGE = 0.8  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

class EKFSLAM:
    def __init__(self, Rt, Qt, landmarks):
        # State Vector [x y yaw]'
        self.xEst = np.zeros((STATE_SIZE, 1))
        self.PEst = np.eye(STATE_SIZE)
        self.initP = np.eye(2)
        self.lm_id = np.empty((0, 1))  # Landmark ID list
        self.Rt = Rt # Noise of robot model
        self.Qt = Qt # Noise of observation model
        self.landmarks = landmarks # Landmarks in world frame
        self.u = np.zeros((2, 1)) # Control input

    def predict(self, u, dt):
        """
        Performs the prediction step of EKF SLAM

        :param u:    2x1 control vector
        :returns:    predicted state vector, predicted covariance, jacobian of control vector, transition fx
        """
        S = STATE_SIZE
        self.u = u
        G, Fx = self.jacob_motion(self.xEst[0:S], u, dt)
        self.xEst[0:S] = self.motion_model(self.xEst[0:S], u, dt)
        # Fx is an an identity matrix of size (STATE_SIZE)
        # sigma = G*sigma*G.T + Noise
        self.PEst[0:S, 0:S] = G.T @ self.PEst[0:S, 0:S] @ G + Fx.T @ self.Rt @ Fx

        return self.xEst, self.PEst

    def update(self, measurement):
        """
        Performs the update step of EKF SLAM

        :param z:     the measurements read at new position
        :returns:     the updated state and covariance for the system
        """        
        
        z   = measurement[1:]
        id  = int(measurement[0] - 1) # ID start from 1
        nLM = self.calc_n_LM(self.xEst)

        if z[0] > MAX_RANGE:
            return self.PEst

        print("Num lm: ", nLM)
        if id == nLM: # Landmark is a NEW landmark
            print("New LM detected with ID: ", id)
            # Extend state and covariance matrix
            xAug = np.vstack((self.xEst, self.calc_LM_Pos(self.xEst, z)))
            PAug = np.vstack((np.hstack((self.PEst, np.zeros((len(self.xEst), LM_SIZE)))),
                            np.hstack((np.zeros((LM_SIZE, len(self.xEst))), self.initP))))
            self.xEst = xAug
            self.PEst = PAug

        lm = self.get_LM_Pos_from_state(self.xEst, id)
        y, S, H = self.calc_innovation(lm, z, id)

        K = (self.PEst @ H.T) @ np.linalg.inv(S) # Calculate Kalman Gain
        self.xEst = self.xEst + (K @ y)
        self.PEst = (np.eye(len(self.xEst)) - (K @ H)) @ self.PEst

        self.xEst[2] = self.pi_2_pi(self.xEst[2])
        print("Updated state: ", self.xEst)
        return self.PEst
    
    def motion_model(self, x, u, dt):
        """
        Computes the motion model based on current state and input function.

        :param x: 3x1 pose estimation
        :param u: 2x1 control input [v; w]
        :returns: the resulting state after the control function is applied
        """
        F = np.array([[1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0]])

        B = np.array([[dt * math.cos(x[2, 0]), 0],
                    [dt * math.sin(x[2, 0]), 0],
                    [0.0, dt]])

        x = (F @ x) + (B @ u)
        return x

    def calc_n_LM(self, x):
        """
        Calculates the number of landmarks currently tracked in the state
        :param x: the state
        :returns: the number of landmarks n
        """
        n = int((len(x) - STATE_SIZE) / LM_SIZE)
        return n

    def jacob_motion(self, x, u, dt):
        """
        Calculates the jacobian of motion model.

        :param x: The state, including the estimated position of the system
        :param u: The control function
        :returns: G:  Jacobian
                Fx: STATE_SIZE x (STATE_SIZE + 2 * num_landmarks) matrix where the left side is an identity matrix
        """

        # [eye(3) [0 x y; 0 x y; 0 x y]]
        Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
            (STATE_SIZE, LM_SIZE * self.calc_n_LM(x)))))

        jF = np.array([[0.0, 0.0, -dt * u[0] * math.sin(x[2, 0])],
                    [0.0, 0.0, dt * u[0] * math.cos(x[2, 0])],
                    [0.0, 0.0, 0.0]],dtype=object)

        G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx
        if self.calc_n_LM(x) > 0:
            print(Fx.shape)
        return G, Fx,
    
    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def jacobH(self, q, delta, x, i):
        """
        Calculates the jacobian of the measurement function

        :param q:     the range from the system pose to the landmark
        :param delta: the difference between a landmark position and the estimated system position
        :param x:     the state, including the estimated system position
        :param i:     landmark id + 1
        :returns:     the jacobian H
        """
        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                    [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

        G = G / q
        nLM = self.calc_n_LM(x)
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                        np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

        F = np.vstack((F1, F2))

        H = G @ F

        return H
    
    def calc_innovation(self, lm, z, LMid):
        """
        Calculates the innovation based on expected position and landmark position

        :param lm:   landmark position
        :param z:    read measurements
        :param LMid: landmark id
        :returns:    returns the innovation y, and the jacobian H, and S, used to calculate the Kalman Gain
        """
        delta = lm - self.xEst[0:2] # difference between landmark position and estimated position
        q = (delta.T @ delta)[0, 0]
        zangle = math.atan2(delta[1, 0], delta[0, 0]) - self.xEst[2, 0] # angle between landmark and estimated position
        zp = np.array([[math.sqrt(q), self.pi_2_pi(zangle)]])
        # zp is the expected measurement based on xEst and the expected landmark position
        y = (z - zp).T # y = innovation
        y[1] = self.pi_2_pi(y[1])

        H = self.jacobH(q, delta, self.xEst, LMid + 1)
        S = H @ self.PEst @ H.T + self.Qt[0:2, 0:2]

        return y, S, H

    def get_LM_Pos_from_state(self, x, ind):
        """
        Returns the position of a given landmark

        :param x:   The state containing all landmark positions
        :param ind: landmark id
        :returns:   The position of the landmark
        """
        lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :] # Get the landmark position from the state

        return lm

    def calc_LM_Pos(self, x, z):
        """
        Calculates the pose in the world coordinate frame of a landmark at the given measurement.

        :param x: [x; y; theta]
        :param z: [range; bearing]
        :returns: [x; y] for given measurement
        """
        zp = np.zeros((2, 1))

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1]) # x + r*cos(theta + phi)
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1]) # y + r*sin(theta + phi)

        return zp