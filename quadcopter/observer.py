import pymoskito as pm
import numpy as np
import sympy as sp
from scipy.linalg import norm

from extended_kalman_filter import ExtendedKalmanFilterObserver
from quadcopter.model import QuadcopterObserverModel


def gen_traj_funcs(t, x, y, z, phi, theta, psi):
    xp = sp.lambdify(t, x)
    yp = sp.lambdify(t, y)
    zp = sp.lambdify(t, z)
    vx = sp.diff(x, t)
    vy = sp.diff(y, t)
    vz = sp.diff(z, t)
    ax = sp.diff(vx, t)
    ay = sp.diff(vy, t)
    az = sp.diff(vz, t)
    wx = sp.diff(phi, t) - sp.diff(psi, t) * sp.sin(theta)
    wy = sp.diff(theta, t) * sp.cos(phi) + sp.diff(psi, t) * sp.cos(theta) * sp.sin(phi)
    wz = -sp.diff(theta, t) * sp.sin(phi) + sp.diff(psi, t) * sp.cos(theta) * sp.cos(phi)

    def c(x_):
        return sp.cos(x_)

    def s(x_):
        return sp.sin(x_)

    R = sp.Matrix([[c(theta) * c(psi), c(theta) * s(psi), -s(theta)],
                   [s(phi) * s(theta) * c(psi) - c(phi) * s(psi), s(phi) * s(theta) * s(psi) + c(phi) * c(psi),
                    s(phi) * c(theta)],
                   [c(phi) * s(theta) * c(psi) + s(phi) * s(psi), c(phi) * s(theta) * s(psi) - s(phi) * c(psi),
                    c(phi) * c(theta)]])

    f = R * (sp.Matrix([[ax, ay, az]]).T + sp.Matrix([[0, 0, 9.81]]).T)

    phip = sp.lambdify(t, phi)
    thetap = sp.lambdify(t, theta)
    psip = sp.lambdify(t, psi)

    return xp, yp, zp, phip, thetap, psip, sp.lambdify(t, wx), sp.lambdify(t, wy), sp.lambdify(t,
                                                                                               wz), sp.lambdify(
        t, f[0]), sp.lambdify(t, f[1]), sp.lambdify(t, f[2])


def lagrange_polynomial(x, points):
    n = len(points)
    expr = sp.S(0)

    for i, ipoint in enumerate(points):
        numerator = sp.S(1)
        for j, jpoint in enumerate(points):
            if i != j:
                numerator *= x - jpoint[0]

        denominator = numerator.subs(x, ipoint[0])
        expr += numerator / denominator * ipoint[1]

    return sp.expand(expr)


class QuadcopterEKF(ExtendedKalmanFilterObserver):
    def __init__(self, settings):
        t = sp.symbols('t')
        x = sp.S(0)
        y = sp.S(0)
        z = sp.S(0)
        psi = lagrange_polynomial(t, [(i-1, y) for i,y in enumerate([0, 0, 3.14, 0, 0, 0, 0])])
        theta = lagrange_polynomial(t, [(i-1, y) for i,y in enumerate([0, 0, 0, 1.5, 0, 0, 0])])
        phi = lagrange_polynomial(t, [(i-1, y) for i,y in enumerate([0, 0, 0, 0, 1.5, 0, 0])])

        self.x, self.y, self.z, self.phi, self.theta, self.psi, self.wx, self.wy, self.wz, self.fx, self.fy, self.fz = gen_traj_funcs(t, x, y, z, psi, theta, phi)

        super().__init__(settings, QuadcopterObserverModel())

    def _observe(self, t, system_input, system_output):
        pi = np.pi
        ref_pos = np.array([self.x(t), self.y(t), self.z(t)])
        ref_angles = np.array([((self.phi(t) + pi) % (2*pi))-pi,
                               ((self.theta(t) + pi / 2) % pi)-pi/2,
                               ((self.psi(t) + pi) % (2 * pi)) - pi])
        ref_angular_velocities = np.array([self.wx(t), self.wy(t), self.wz(t)])
        ref_specific_force = np.array([self.fx(t), self.fy(t), self.fz(t)])

        gyro_measurement = ref_angular_velocities
        accelerometer_measurement = ref_specific_force
        system_measurement = np.concatenate((accelerometer_measurement, np.array([ref_angles[2]])))

        observer_out = super()._observe(t, system_input=gyro_measurement, system_output=system_measurement)
        a = observer_out[0]
        b = observer_out[1]
        c = observer_out[2]
        d = observer_out[3]

        obs_phi = np.arctan2(2*(c*d+a*b), a**2-b**2-c**2+d**2)
        obs_theta = np.arcsin(2*(a*c-b*d))
        obs_psi = np.arctan2(2*(b*c+a*d), a*+2+b**2-c**2-d**2)

        output = np.concatenate((ref_pos, ref_angles, observer_out, np.array([obs_phi, obs_theta, obs_psi]), np.array([norm(observer_out[:4])])))
        return output


pm.register_simulation_module(pm.Observer, QuadcopterEKF)
