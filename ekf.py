import numpy as np
from scipy.spatial.transform import Rotation as R

class EKF:
    def __init__(self, dt):
        # Time step
        self.dt = dt

        # Gravity
        self.g = np.array([0, 0, -9.81])

        # State: [position(3), velocity(3), quaternion(4), gyro bias(3), acc bias(3)]
        self.x = np.zeros(16)
        self.x[6] = 1.0  # quaternion w=1, x=y=z=0

        # Covariance
        self.P = np.eye(15) * 1e-3  # 15 because quaternion is handled separately

        # Process noise covariance
        self.Q = np.diag([
            0.015**2, 0.015**2, 0.015**2,  # gyro noise
            0.019**2, 0.019**2, 0.019**2,  # acc noise
            0.0001**2, 0.0001**2, 0.0001**2,  # gyro bias random walk
            0.0002**2, 0.0002**2, 0.0002**2   # acc bias random walk
        ])

    def predict(self, omega_meas, acc_meas):
        """
        EKF Prediction step.
        :param omega_meas: Gyroscope measurement (3x1)
        :param acc_meas: Accelerometer measurement (3x1)
        """

        # Extract states
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        bg = self.x[10:13]
        ba = self.x[13:16]

        # Corrected measurements
        omega = omega_meas - bg
        acc = acc_meas - ba

        # Quaternion update
        delta_q = self._small_angle_quaternion(omega * self.dt)
        q_new = self._quat_multiply(q, delta_q)
        q_new /= np.linalg.norm(q_new)

        # Rotation matrix from body to world
        Rwb = R.from_quat([q_new[1], q_new[2], q_new[3], q_new[0]]).as_matrix()

        # Position and velocity update
        a_world = Rwb @ acc + self.g
        v_new = v + a_world * self.dt
        p_new = p + v * self.dt + 0.5 * a_world * self.dt ** 2

        # Update state vector
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new

        # Bias random walk (already modeled in Q)

        # Jacobians for covariance propagation
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -Rwb @ self._skew(acc) * self.dt
        F[3:6, 12:15] = -Rwb * self.dt
        F[6:9, 9:12] = -np.eye(3) * self.dt

        # Process noise covariance for this step
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -Rwb
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        Qk = G @ self.Q @ G.T * self.dt**2

        # Covariance propagation
        self.P = F @ self.P @ F.T + Qk

    def update_camera(self, z, landmark_pos, K):
        """
        EKF Update step for camera measurements.
        :param z: 2D observed point (pixel coordinates)
        :param landmark_pos: 3D landmark position in world frame
        :param K: Camera intrinsic matrix
        """
        p = self.x[0:3]
        q = self.x[6:10]

        Rwb = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        # Project landmark into camera frame
        Pc = Rwb.T @ (landmark_pos - p)
        u = K @ Pc
        u = u[:2] / u[2]

        # Innovation
        y = z - u

        # Measurement Jacobian
        H = np.zeros((2, 15))
        X, Y, Z = Pc
        H[0, 0:3] = K[0, 0] / Z * np.array([-1, 0, X/Z])
        H[1, 0:3] = K[1, 1] / Z * np.array([0, -1, Y/Z])

        # Camera measurement noise
        R_cam = np.eye(2) * 1.0  # assume 1 pixel std dev

        # Kalman Gain
        S = H @ self.P @ H.T + R_cam
        K_gain = self.P @ H.T @ np.linalg.inv(S)

        # State correction
        dx = K_gain @ y
        self._inject_error(dx)

        # Covariance update
        I = np.eye(15)
        self.P = (I - K_gain @ H) @ self.P

    def _inject_error(self, dx):
        """Inject the error into the state"""
        self.x[0:3] += dx[0:3]
        self.x[3:6] += dx[3:6]

        dq = self._small_angle_quaternion(dx[6:9])
        q = self.x[6:10]
        q_new = self._quat_multiply(q, dq)
        self.x[6:10] = q_new / np.linalg.norm(q_new)

        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

    @staticmethod
    def _small_angle_quaternion(delta_theta):
        """Convert small angle to quaternion"""
        delta_q = np.zeros(4)
        delta_q[0] = 1.0
        delta_q[1:4] = 0.5 * delta_theta
        return delta_q

    @staticmethod
    def _quat_multiply(q1, q2):
        """Quaternion multiplication q = q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def _skew(v):
        """Return the skew-symmetric matrix of a vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
