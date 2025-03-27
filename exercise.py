import numpy as np

H = .1
H_STEP = .05
R = .3
num_drones = 3
INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

#### Initialize a circular trajectory ######################
PERIOD = 15
INIT_POS_A = INIT_XYZS[0, :]
INIT_POS_B = INIT_XYZS[1, :]
control_freq_hz = 4
NUM_WP = control_freq_hz * PERIOD  # 轨迹点数量
TARGET_POS_A = np.linspace(INIT_POS_A, INIT_POS_B, NUM_WP)  # A -> B 轨迹
TARGET_POS_B = np.linspace(INIT_POS_B, INIT_POS_A, NUM_WP)  # B -> A 轨迹
TARGET_POS = np.vstack([TARGET_POS_A, TARGET_POS_B])  # 存储轨迹
wp_counters = np.array([0, NUM_WP // 2])
print('')