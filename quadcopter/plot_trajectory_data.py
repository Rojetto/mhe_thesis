from quadcopter.util import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi

rcParams['figure.dpi'] = 96*2

imu_ts, acc, gyro, mocap_ts, pos, ori = get_trajectory(TrajectoryData.UAV_3)

print(f"Initial orientation: {list(ori[0])}")
print(f"Initial state: {list(euler_to_q(ori[0])) + [0., 0., 0.]}")

figure('3D Trajectory')
ax = subplot(111, projection='3d')
ax.set_aspect('equal')
plot(pos[:,0], pos[:,1], pos[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

figure('Reference positions')
plot(mocap_ts, pos[:, 0])
plot(mocap_ts, pos[:, 1])
plot(mocap_ts, pos[:, 2])
grid()

figure('Reference orientations')
plot(mocap_ts, rad_to_deg(ori[:, 0]))
plot(mocap_ts, rad_to_deg(ori[:, 1]))
plot(mocap_ts, rad_to_deg(ori[:, 2]))
grid()

figure('IMU accelerometer')
plot(imu_ts, acc[:, 0])
plot(imu_ts, acc[:, 1])
plot(imu_ts, acc[:, 2])
grid()

figure('IMU gyro')
plot(imu_ts, rad_to_deg(gyro[:, 0]))
plot(imu_ts, rad_to_deg(gyro[:, 1]))
plot(imu_ts, rad_to_deg(gyro[:, 2]))
grid()


show()