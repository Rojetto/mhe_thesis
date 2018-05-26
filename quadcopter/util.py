import math

import numpy as np
import sympy as sp


def get_trajectory(traj_name):
    import os
    import pathlib
    import pickle
    script_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    traj_dir = script_directory / 'trajectories'
    with open(traj_dir / f"{traj_name}.dat", 'rb') as f:
        traj_tuple = pickle.load(f)

    return traj_tuple


def time_to_index(t, ts):
    t_min = min(ts)
    t_max = max(ts)
    index = math.floor((t - t_min) / (t_max - t_min) * ts.size)
    index = max(min(index, ts.size - 1), 0)
    return index


def euler_difference(ref, meas):
    pi = np.pi
    err_phi = (((meas[0] - ref[0]) + pi) % (2 * pi)) - pi
    err_theta = (((meas[1] - ref[1]) + pi / 2) % pi) - pi / 2
    err_psi = (((meas[2] - ref[2]) + pi) % (2 * pi)) - pi

    return np.array([err_phi, err_theta, err_psi])


def q_to_euler(q):
    a, b, c, d = q

    obs_phi = np.arctan2(2 * (c * d + a * b), a ** 2 - b ** 2 - c ** 2 + d ** 2)
    obs_theta = np.arcsin(2 * (a * c - b * d))
    obs_psi = np.arctan2(2 * (b * c + a * d), a ** 2 + b ** 2 - c ** 2 - d ** 2)

    return np.array([obs_phi, obs_theta, obs_psi])


def euler_to_q(euler):
    phi2, theta2, psi2 = np.array(euler)*0.5
    s = np.sin
    c = np.cos

    q1 = c(phi2)*c(theta2)*c(psi2)+s(phi2)*s(theta2)*s(psi2)
    q2 = s(phi2)*c(theta2)*c(psi2)-c(phi2)*s(theta2)*s(psi2)
    q3 = c(phi2)*s(theta2)*c(psi2)+s(phi2)*c(theta2)*s(psi2)
    q4 = c(phi2)*c(theta2)*s(psi2)+s(phi2)*s(theta2)*c(psi2)

    return np.array([q1, q2, q3, q4])


def generate_trajectory_files():
    """To be called by hand"""

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

    def sim_trajectory(t, x, y, z, phi, theta, psi, gyro_covariance, accelerometer_covariance, end_time, step_width):
        x, y, z, phi, theta, psi, wx, wy, wz, fx, fy, fz = gen_traj_funcs(t, x, y, z, psi, theta, phi)
        pi = np.pi
        np.random.seed(10)
        ts = np.arange(0, end_time, step_width)
        ref_positions = np.array([[x(t), y(t), z(t)] for t in ts])
        ref_orientations = np.array([[((phi(t) + pi) % (2 * pi)) - pi,
                                      ((theta(t) + pi / 2) % pi) - pi / 2,
                                      ((psi(t) + pi) % (2 * pi)) - pi] for t in ts])
        meas_gyro = np.array([np.random.normal([wx(t), wy(t), wz(t)], np.sqrt(gyro_covariance)) for t in ts])
        meas_accelerometer = np.array(
            [np.random.normal([fx(t), fy(t), fz(t)], np.sqrt(accelerometer_covariance)) for t in ts])

        return ts, ref_positions, ref_orientations, meas_gyro, meas_accelerometer

    def write_tuple(traj_name, traj_tuple):
        import os
        import pathlib
        import pickle
        script_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
        traj_dir = script_directory / 'trajectories'
        traj_dir.mkdir(exist_ok=True)
        with open(traj_dir / f"{traj_name}.dat", 'wb') as f:
            pickle.dump(traj_tuple, f)

    t = sp.symbols('t')
    x = sp.S(0)
    y = sp.S(0)
    z = sp.S(0)
    psi = lagrange_polynomial(t, [(i - 1, y) for i, y in enumerate([0, 0, 3, 0, 0, 0, 0])])
    theta = lagrange_polynomial(t, [(i - 1, y) for i, y in enumerate([0, 0, 0, 1.5, 0, 0, 0])])
    phi = lagrange_polynomial(t, [(i - 1, y) for i, y in enumerate([0, 0, 0, 0, 1.5, 0, 0])])
    traj_tuple = sim_trajectory(t, x, y, z, phi, theta, psi, 0.1, 0.1, 4, 0.005)
    write_tuple('sim_1_stationary', traj_tuple)