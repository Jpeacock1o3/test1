from scipy.spatial.transform import Rotation as R
import numpy as np
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# import matplotlib.pyplot as plt  # plotting
import jax.numpy as jnp
import diffrax
import config as c

def ODEfunc(
    t,
    y,
    args,
):
        
    q_D = jnp.array([[y[0]], [y[1]], [y[2]], [y[3]]])
    omegaTarget = jnp.array([[y[4]], [y[5]], [y[6]]])

    quarKinMat_D = jnp.array(
        [
            [-q_D[1, 0], -q_D[2, 0], -q_D[3, 0]],
            [q_D[0, 0], -q_D[3, 0], q_D[2, 0]],
            [q_D[3, 0], q_D[0, 0], -q_D[1, 0]],
            [-q_D[2, 0], q_D[1, 0], q_D[0, 0]],
        ]
    )
    quar_DDot = 0.5 * (quarKinMat_D @ omegaTarget)

    ds_18 = quar_DDot.flatten()
    # ds_19 = jnp.array([[0.0], [0.0], [0.0]]).flatten()

    inv_targetI = jnp.linalg.inv(c.targetI)
    omegaDotTarget = inv_targetI @ (
        jnp.array([[0.0], [0.0], [0.0]])
        - jnp.cross(
            omegaTarget.flatten(), (c.targetI @ omegaTarget).flatten()
        ).reshape((3, 1))
    )
    ds_19 = omegaDotTarget.flatten()

    # jax.debug.print("{x}", x=ds_19[0])
    # jax.debug.print("{x}", x=ds_19[0])
    # jax.debug.print("{x}", x=ds_19[0])

    ds = jnp.hstack(
        [
            ds_18,
            ds_19,
        ]
    )

    return ds

def solveODEpose(y0, args, tFinal ):
    term = diffrax.ODETerm(
        ODEfunc
    )
    solver = diffrax.Bosh3()  # RK23
    # solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, tFinal, int((tFinal - 0.0) / 0.1) + 1))
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.295,
        icoeff=0.295,
        dcoeff=0,
        rtol=1e-5,
        atol=1e-5,
    )  # compile and ran 2 for Bosh3() # RK23

    sol = diffrax.diffeqsolve(
        term,
        solver,
        0.0,
        tFinal,
        0.1,
        y0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=40000000,
    )

    return sol

# @jax.jit
def solveODE(q_D, omega0target, tFinal):

    s0 = jnp.concatenate(
        (q_D, omega0target), axis=None
    )  # for target

    # y0 = tuple(s0)
    y0 = jnp.asarray(s0)
    args = 0;
    #######################  Sim 
    sol_BC = solveODEpose(y0, args, tFinal)
    s_mod_code2_BC = sol_BC.ys
    timeOut_BC = sol_BC.ts

    sim_results = s_mod_code2_BC

    return sim_results

    ####################### outputs ends ############################

# Sim Inputs ####################################################################################
def calc_TargetPoseAtTime(thetaTargetX_t0, thetaTargetY_t0, thetaTargetZ_t0, thetaTargetX_rate, thetaTargetY_rate, thetaTargetZ_rate, tFinal):

    omega0target = (jnp.pi / 180.0) * jnp.array(
        [[thetaTargetX_rate], [thetaTargetY_rate], [thetaTargetZ_rate]]
    )

    target321EA = R.from_euler(
        "zyx", [thetaTargetX_t0, thetaTargetY_t0, thetaTargetZ_t0], degrees=True
    )  # 321 EA, from I to D
    q_D = jnp.flip(target321EA.as_quat())

    sim_out = solveODE(q_D, omega0target, tFinal)

    q1_target = sim_out[:, 0]
    q2_target = sim_out[:, 1]
    q3_target = sim_out[:, 2]
    q4_target = sim_out[:, 3]
    omegaX_target = sim_out[:, 4]
    omegaY_target = sim_out[:, 5]
    omegaZ_target = sim_out[:, 6]

    omegaCheck = ( 180.0/ jnp.pi ) * jnp.array(
        [[omegaX_target[-1]], [omegaY_target[-1]], [omegaZ_target[-1]]]
    )

    finalQuat = R.from_quat(
        [q4_target[-1], q3_target[-1], q2_target[-1], q1_target[-1]], 
    ) 
    finalEA = finalQuat.as_euler('zxy', degrees=True)

    outputs = np.concatenate(
        (finalEA, omegaCheck), axis=None
    )

    return outputs

# # Initial angle at t=0 in deg
thetaTargetX_t0 = 0.0
thetaTargetY_t0 = 0.0
thetaTargetZ_t0 = 0.0

# thetaTargetX_rate = 0.0  #  range: 1 to 10 deg/s
# thetaTargetY_rate = 0.0  #  range: 5 to 30 deg/s
# thetaTargetZ_rate = 10.0  #  range: 1 to 10 deg/s

thetaTargetX_rate = 5.0  #  range: 1 to 10 deg/s
thetaTargetY_rate = 5.0  #  range: 5 to 30 deg/s
thetaTargetZ_rate = 5.0  #  range: 1 to 10 deg/s

# Initial angle at t=0 in deg
# thetaTargetX_t0 = 60.0
# thetaTargetY_t0 = 40.0
# thetaTargetZ_t0 = 0.0

tFinal = 14.1

calc_TargetPoseAtTime(thetaTargetX_t0, thetaTargetY_t0, thetaTargetZ_t0, thetaTargetX_rate, thetaTargetY_rate, thetaTargetZ_rate, tFinal)