# L = size of net, m                                                    %
# lmesh = size of mesh, m                                               %
# z0 = initial z-coordinate, m                                          %
# n = number of nodes in a line, -                                      %
# N = number of nodes in the net, -                                     %
# N_I = number of nodes inside a single thread                          %
# N_k = number of knots on a side                                       %
# N_ct = number of nodes on a corner thread                             %
# d_knot = distance between knots                                       %
# d_knot_node = distance between two nodes in the net                   %
# mbullet = mass of a corner bullet, kg                                 %
# mnet = mass of net, kg                                                %
# E = Young's modulus, Pa                                               %
# r = radius of braid, m                                                %
# A = cross-section of braid, m^2                                       %
# X(i) = x-coordinate of node i, m                                      %
# Y(i) = y-coordinate of node i, m                                      %
# Conn_table = matrice con: indice nodo1, indice nodo2 , -              %
# lrest = length of threads at rest, m                                  %
# M(i) = mass of i-th knot, kg                                          %
# Nlinks = number of links between nodes, -                             %
# mhalf = mass of half a segment, kg                                    %
# F = force matrix, N*3                                                 %
# s = state vector (positions, velocities of nodes, power (m,m/s, W))   %
# Ek = kinetic energy, J (matrix with Ek of each mass at each time)     %
# Ug = gravitational potential energy, J (matrix with U ..."...    )    %
# Ue = elastic potential energy, J (vector with Ue at each time)        %

# from numpy import *
import jax.numpy as jnp
from jax import config as jax_config
from scipy.spatial.transform import Rotation as R
# from math import sqrt
jax_config.update("jax_enable_x64", True)


# targetPresent, closingPresent, controlPresent, chaserPresent = 1, 1, 1, 1 # For PID
# targetPresent, closingPresent, controlPresent, chaserPresent = 0,0,0,0 # For passive, flight only
targetPresent, closingPresent, controlPresent, chaserPresent = (
    1,
    1,
    1,
    1,
)  # For passive, flight and capture
# targetPresent, closingPresent, controlPresent, chaserPresent = (
#     1,
#     0,
#     0,
#     0,
# )  # For passive, flight and capture

# Net Properties, fixed ################################################################################
lknot = 0
N_I = 0
L = 22
alpha = 1.5 / L
mknot = 0.0
g = 0.0
# l_ct = jnp.sqrt(2) * lmesh  # for zenit cap
# l_ct = 1.4142
# lmesh = 2.0

# Net Properties, vary between simulation ################################################################################
# N_k = 15  #
# lmesh = L / (N_k - 1)
mbullet = 2.5  # kg/m3
rho_al = 2700  #      kg/m3
# rho_net = 1440#     kg/m3  Technora (Mortensen06)
rho_net = 1390  #     kg/m3  for zenit cap
E = 70000000000  # Pa
r = 0.0011  # m, for zenit cap
r_corner = 0.0007  # m, for zenit cap
csi = 0.106  # damping ratio
beta = 5.808404002974731 * (10 ** (-4))  # damping constant

# Contact Properties################################################################################
E_net = E / 10.0
# because E is for a fiber. It is too high for the spheres with which we model the contact
G_ground = 25.0 * 10 ** (9)  # Al
G_net = 2.9 * 10 ** (9)  # Kevlar
ni_net = 0.36  # kevlar guida DuPont
ni_ground = 0.33  # Al
ni_bullet = 0.33  # Al
E_ground = 69 * (10 ** (9))  # Al
E_bullet = 69 * (10 ** (9))  # Al
alpha_contact_CMCM = 0.25
alpha_contact_net = 0.05
n_contact = 1.5
# data for bristle friction (no friction model for node-node impact)  % mu_s_NetNet = 0.2;
vt_lim = 10 ** (-3)
mu_s_NetAl = 0.19  # Davis97
mu_s_AlAl = 0.42  # Davis97
mu_k_NetAl = mu_s_NetAl / 1.25
mu_k_AlAl = mu_s_AlAl / 1.25
k_b = 1.0485 * 10 ** (8)
c_b = 206.4873
alpha_contact_CMCM = 0.25
alpha_contact_net = 0.05
n_contact = 1.5

# Contact Properties################################################################################
# Gstar_SC_Net = c.G_ground / (2 - c.ni_ground) + c.G_net / (2 - c.ni_net)
h_net = (1 - ni_net**2) * (1 / jnp.pi) * (1 / E)
h_ground = (1 - ni_ground**2) * (1 / jnp.pi) * (1 / E_ground)
h_bullet = (1 - ni_bullet**2) * (1 / jnp.pi) * (1 / E_bullet)

# Net IC #################################################################################
ve = 0.0  # for PID
# ve = 2.5;
# theta = 35.0;


# PID #################################################################################
K1, K2, K3 = 10.0, 6.0, 6.0
# tFlight = 25.0
tFlight = 20.0
# tclose = tFlight + 0.0 # For PID
# tclose = 21.0  # For passive
tclose = 130.0  # For passive
tOn = 15.0  # For RL OL
# tOn = 0.0  # For PID

# target inputs to modify between simulations, PID #################################################################################
# xCylinder = 8.9 # range: -9 to 9 m
# yCylinder = 6.2000 # range: -9 to 9 m
# zCylinder = -44.0 # range: -60 to -40 m
# thetaTargetX_rate = 0.0 #  range: 1 to 10 deg/s
# thetaTargetY_rate = 5.0 #  range: 5 to 30 deg/s
# thetaTargetZ_rate = 0.0 #  range: 1 to 10 deg/s

# target inputs for passive net #################################################################################
xCylinder = 0.0  # range: -5 to 5 m
yCylinder = 0.0  # range: -5 to 5 m
zCylinder = -15.0  # range: -10 to -20 m
thetaTargetX_rate = 0.0  #  range: 1 to 10 deg/s
thetaTargetY_rate = 5.0  #  range: 5 to 30 deg/s
thetaTargetZ_rate = 0.0  #  range: 1 to 10 deg/s

# target info #################################################################################
targetType = 0 # Simple Sat
# targetType = 1 # Zenit-2
# targetType = 2 # Apollo
config_cy = 2
# xCylinder = 8.9
# yCylinder = 6.2000
# zCylinder = -44.0
# shift to center with net
# xCylinder = L*alpha/2.0 + xCylinder
# yCylinder = L*alpha/2.0 + yCylinder
sphereRadius = 3.0
apolloBodyCylRadius = 1.9
apolloBodyCylHeight = 4.8
apolloEngineRadius = 1.289
apolloEngineHeight = 4.946
apolloBigNoseSphere = 1.8
apolloSmallNoseSphere = 0.8
cylinderRadius = 1.95
smallCylinderRadius = 0.08
mainEngineRadius = 1.0
smallEngineRadius = 0.3
cylinderHeight = 10.2
mainEngineHeight = 0.8
smallEngineHeight = 0.8
l_s = 4.0 # side lengths of cubic debris
rect_Ls = 4.0 # rectangular side length on the inertial x-axis (body length)
rect_Hs = 4.0 # rectangular side length on the inertial z-axis (body length)
rect_Ws = 4.0 # rectangular side length on the inertial y-axis (body length)
rect2_Ls = 4.0
rect2_Hs = 2.0
rect2_Ws = 2.0
rect3_Ls = rect2_Ls
rect3_Hs = 2.0
rect3_Ws = 2.0 
# Initial angle at t=0 in deg
thetaTargetX_t0 = 90.0
thetaTargetY_t0 = 0.0
thetaTargetZ_t0 = 0.0
# constant angular rate in deg/s, body frame
# thetaTargetX_rate = 0.0
# thetaTargetY_rate = 5.0
# thetaTargetZ_rate = 0.0
# constant angular rate in deg/s, inertial frame
# targetIAngRateX = 0.0
# targetIAngRateY = 0.0
# targetIAngRateZ = 0.0
targetSideLengthX = cylinderRadius * 2
targetSideLengthY = cylinderRadius * 2
targetSideLengthZ = cylinderHeight
targetJX = 94880
targetJY = 94880
targetJZ = 46295.5
targetM = 9000
# cylinderHeight = 2.0; # Original

# Chaser #############################################################################################
chaserMass = 1600
chaserSideLength = 1.5
chaserDistanceFromNet = 0.85
chaserJX = 266.6667
chaserJY = 266.6667
chaserJZ = 266.6667
chaserIAngRateX = 0.0
chaserIAngRateY = 0.0
chaserIAngRateZ = 0.0
chaserIQuatS = 0.576627
chaserIQuatX = 0.107787
chaserIQuatY = 0.555270
chaserIQuatZ = 0.589542

# Chaser and Target IC #####################################################################################
# For Chaser #####################################################################################
# chaser properties
chaserI = jnp.array([[chaserJX, 0, 0], [0, chaserJY, 0], [0, 0, chaserJZ]])
# kg m**2
pos0chaser = jnp.array(
    [[L * alpha / 2.0], [L * alpha / 2.0], [chaserDistanceFromNet]]
)
# m
vel0chaser = jnp.array([[0.0], [0.0], [0.0]])
# m/s
q_C = jnp.array([[1.0], [0.0], [0.0], [0.0]])
rotMat_C_A_I = jnp.array(
    [
        [
            q_C[0, 0] ** 2 + q_C[1, 0] ** 2 - q_C[2, 0] ** 2 - q_C[3, 0] ** 2,
            2 * (q_C[1, 0] * q_C[2, 0] - q_C[0, 0] * q_C[3, 0]),
            2 * (q_C[1, 0] * q_C[3, 0] + q_C[0, 0] * q_C[2, 0]),
        ],
        [
            2 * (q_C[1, 0] * q_C[2, 0] + q_C[0, 0] * q_C[3, 0]),
            q_C[0, 0] ** 2 - q_C[1, 0] ** 2 + q_C[2, 0] ** 2 - q_C[3, 0] ** 2,
            2 * (q_C[3, 0] * q_C[2, 0] - q_C[0, 0] * q_C[1, 0]),
        ],
        [
            2 * (q_C[1, 0] * q_C[3, 0] - q_C[2, 0] * q_C[0, 0]),
            2 * (q_C[2, 0] * q_C[3, 0] + q_C[0, 0] * q_C[1, 0]),
            q_C[0, 0] ** 2 - q_C[1, 0] ** 2 - q_C[2, 0] ** 2 + q_C[3, 0] ** 2,
        ],
    ]
).T
omega0chaserIner = jnp.array(
    [[chaserIAngRateX], [chaserIAngRateY], [chaserIAngRateZ]]
)
# exp in inertial frame
omega0chaser = rotMat_C_A_I @ omega0chaserIner
# transform to exp in body frame
# intial states of the chaser
# states0_chaser = [pos0chaser; vel0chaser; q_C; omega0chaser];
# For Target #####################################################################################
# target properties
targetI = jnp.array([[targetJX, 0, 0], [0, targetJY, 0], [0, 0, targetJZ]])
# kg m**2
pos0target = jnp.array([[xCylinder], [yCylinder], [zCylinder]])
# m
vel0target = jnp.array([[0.0], [0.0], [0.0]])
# m/s
# rotMat_D_A_I = (
#     calc_contactF_3_MO.rotMatX(thetaTargetX_t0)
#     @ calc_contactF_3_MO.rotMatY(thetaTargetY_t0)
#     @ calc_contactF_3_MO.rotMatZ(thetaTargetZ_t0)
# ).T
r = R.from_euler(
    "zyx", [thetaTargetX_t0, thetaTargetY_t0, thetaTargetZ_t0], degrees=True
)  # 321 EA, from I to D
# rTrans = r.as_matrix().transpose()
# r.inv().as_quat();# # r.as_euler('zyx', degrees=True);# r.as_matrix()
q_D = jnp.flip(r.as_quat())
omega0target = (jnp.pi / 180) * jnp.array(
    [[thetaTargetX_rate], [thetaTargetY_rate], [thetaTargetZ_rate]]
)
# exp in body frame

# Main tether #################################################################################
l0_mt = (
    jnp.sqrt(xCylinder**2 + yCylinder**2 + zCylinder**2)
    - 1.0 * cylinderRadius
    + 1.0 * chaserDistanceFromNet
)
# l0_mt = jnp.sqrt(xCylinder**2 + yCylinder**2 + zCylinder**2) - 0.0
# l0_mt = 20
# K_mt = 10000
EA_mt = 47040.0
K_mt = EA_mt / l0_mt
c_mt = K_mt / 1000.0
r_mt = 0.002
chaserTimeOnT = tFlight + 100.0
# Closing mech #################################################################################
# k_close = 20
# c_close = 2
# k_close = 2000.0
k_close = 1000.0
c_close = k_close / 10000.0
closeLength = 0.1
closingDuration = 5.0
countClose = 0
winch_rate = -2.25

winchRadius = 0.05
winchHeight = 0.02
winchMass = 0.1
winchMMI = 0.5*winchMass*winchRadius*winchRadius

toll = 10 ** (-3)

# Integrator Options #################################################################################
t0 = 0.0
# t1 = tFlight + 0.0
# t1 = tFlight + 15.0s
# t1 = 25.0  # With Capture
# t1_AC = 15.0 # With Capture

t1 = 25.0  # With Capture
t1_AC = 10.0 # With Capture

# t1 = 20.0 # Flight only
dt = 0.1
# t1 = 14.0 + dt# for valid
# t1_AC = 1.0 + dt# for valid
# t1 = 14.0 # for valid
# t1_AC = 14.0 # for valid
# t1 = 13.5 # for valid
# t1_AC = 14.5 # for valid

## Additional Vars
