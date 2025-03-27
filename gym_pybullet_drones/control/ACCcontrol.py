import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class AccelerationControl(BaseControl):
    """
    Acceleration-based control class for Crazyflies.
    Replaces position PID with externally provided target acceleration (e.g., from GNN).
    Outputs 4 motor RPMs based on desired acceleration and yaw.
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] AccelerationControl only supports CF2X or CF2P")
            exit()

        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([0., 0., 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                [-.5, -.5, -1],
                [-.5,  .5,  1],
                [ .5,  .5, -1],
                [ .5, -.5,  1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [ 0, -1, -1],
                [+1,  0, +1],
                [ 0, +1, -1],
                [-1,  0, +1]
            ])

        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_acc,
                       target_rpy=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """
        Core interface. Accepts target acceleration and yaw, outputs motor RPMs.
        """
        self.control_counter += 1
        thrust, computed_target_rpy = self._accelDrivenPositionControl(
            target_acc=target_acc,
            target_yaw=target_rpy[2],
            cur_quat=cur_quat
        )

        rpm = self._dslPIDAttitudeControl(
            control_timestep,
            thrust,
            cur_quat,
            computed_target_rpy,
            target_rpy_rates
        )

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, computed_target_rpy[2] - cur_rpy[2]

    def _accelDrivenPositionControl(self, target_acc, target_yaw, cur_quat):
        """
        Converts target acceleration and yaw into thrust magnitude and target attitude.
        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        target_thrust = target_acc + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        z_b = target_thrust / np.linalg.norm(target_thrust)
        x_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        R_target = np.vstack([x_b, y_b, z_b]).T
        target_euler = Rotation.from_matrix(R_target).as_euler('XYZ', degrees=False)

        return thrust, target_euler

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates):
        """
        PID controller for attitude tracking.
        """
        cur_rot = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_rot = Rotation.from_euler('XYZ', target_euler, degrees=False).as_matrix()

        rot_matrix_e = target_rot.T @ cur_rot - cur_rot.T @ target_rot
        rot_e = np.array([rot_matrix_e[2,1], rot_matrix_e[0,2], rot_matrix_e[1,0]])

        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy

        self.integral_rpy_e -= rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        target_torques = -np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)

        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm
