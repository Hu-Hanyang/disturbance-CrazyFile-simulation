import heterocl as hcl
import numpy as np
import math
import imp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.integrate
import random

from odp.computeGraphs.CustomGraphFunctions import my_abs
from odp.Grid import Grid  # Utility functions to initialize the problem
from odp.Shapes import *
from odp.Plots import PlotOptions # Plot options
from odp.solver import HJSolver  # Solver core
from odp.solver import TTRSolver
from scipy.integrate import solve_ivp
import argparse


class UAV6D:
    def __init__(self, 
                x=[0, 0, 0, 0, 0, 0], 
                uMin=np.array([-5.3*10**-3, -5.3*10**-3, -1.43*10**-4]), 
                uMax=np.array([5.3*10**-3,  5.3*10**-3,  1.43*10**-4]),
                dims=6, 
                uMode="min", 
                dMode="max"):

        # mode        
        self.uMode = uMode
        self.dMode = dMode

        # state
        self.x = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax
    
        # Disturbance bounds
        self.dMin = 2*uMin
        self.dMax = 2*uMax

        # Dimension 
        self.dims = dims

        # Constants in the equations
        self.Ixx = 1.65*10**-5  # moment of inertia from ETH page 57
        self.Iyy = 1.65*10**-5  # moment of inertia
        self.Izz = 2.92*10**-5  # moment of inertia

    def opt_ctrl(self, t, state, spat_deriv):
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")
        uOpt3 = hcl.scalar(0, "uOpt3")
        # Just create and pass back, even though they're not used
        in4 = hcl.scalar(0, "in4")


        if (self.uMode == "min"):
            with hcl.if_(spat_deriv[3] > 0):
                uOpt1[0] = self.uMin[0]
            with hcl.elif_(spat_deriv[3] < 0):
                uOpt1[0] = self.uMax[0]

            with hcl.if_(spat_deriv[4] > 0):
                uOpt2[0] = self.uMin[1]
            with hcl.elif_(spat_deriv[4] < 0):
                uOpt2[0] = self.uMax[1]
            
            with hcl.if_(spat_deriv[5] > 0):
                uOpt3[0] = self.uMin[2]
            with hcl.elif_(spat_deriv[5] < 0):
                uOpt3[0] = self.uMax[2]

        elif (self.uMode == "max"):
            with hcl.if_(spat_deriv[3] > 0):
                uOpt1[0] = self.uMax[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                uOpt1[0] = self.uMin[0]

            with hcl.if_(spat_deriv[4] > 0):
                uOpt2[0] = self.uMax[1] 
            with hcl.elif_(spat_deriv[4] < 0):
                uOpt2[0] = self.uMin[1] 

            with hcl.if_(spat_deriv[5] > 0):
                uOpt3[0] = self.uMax[2]
            with hcl.elif_(spat_deriv[5] < 0):
                uOpt3[0] = self.uMin[2] 
        
        else:
            raise ValueError("undefined uMode ...")

        return (uOpt1[0], uOpt2[0], uOpt3[0], in4)
    
    
    def opt_dstb(self, t, state, spat_deriv):
        # Graph takes in 4 possible inputs, by default, for now
        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")
        dOpt3 = hcl.scalar(0, "dOpt3")
        # Just create and pass back, even though they're not used
        d4 = hcl.scalar(0, "d4")

        if (self.dMode == "min"):
            with hcl.if_(spat_deriv[3] > 0):
                dOpt1[0] = self.dMax[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt1[0] = self.dMin[0]

            with hcl.if_(spat_deriv[4] > 0):
                dOpt2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[4] < 0):
                dOpt2[0] = self.dMin[1] 
            
            with hcl.if_(spat_deriv[5] > 0):
                dOpt3[0] = self.dMax[2]
            with hcl.elif_(spat_deriv[5] < 0):
                dOpt3[0] = self.dMin[2] 

        elif (self.dMode == "max"):
            with hcl.if_(spat_deriv[3] > 0):
                dOpt1[0] = self.dMin[0] 
            with hcl.elif_(spat_deriv[3] < 0):
                dOpt1[0] = self.dMax[0] 

            with hcl.if_(spat_deriv[4] > 0):
                dOpt2[0] = self.dMin[1] 
            with hcl.elif_(spat_deriv[4] < 0):
                dOpt2[0] = self.dMax[1]

            with hcl.if_(spat_deriv[5] > 0):
                dOpt3[0] = self.dMin[2] 
            with hcl.elif_(spat_deriv[5] < 0):
                dOpt3[0] = self.dMax[2] 

        return (dOpt1[0], dOpt2[0], dOpt3[0], d4)
        
    """
    :: Dynamics of 6D full quadrotor, refer to https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
    """
    def dynamics(self, t, state, uOpt, dOpt):

        phi_dot = hcl.scalar(0, "phi_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        psi_dot = hcl.scalar(0, "psi_dot")
        p_dot = hcl.scalar(0, "p_dot")
        q_dot = hcl.scalar(0, "q_dot")
        r_dot = hcl.scalar(0, "r_dot")

        phi = state[0] # roll
        theta = state[1] # pitch 
        psi = state[2] # yaw
        p = state[3]
        q = state[4]
        r = state[5]
        tau_x = uOpt[0]
        tau_y = uOpt[1]
        tau_z = uOpt[2]

        # some constants
        I_xx = self.Ixx
        I_yy = self.Iyy
        I_zz = self.Izz
        tau_wx = dOpt[0]
        tau_wy = dOpt[1]
        tau_wz = dOpt[2]

        # state dynamics equation
        phi_dot[0] = p + r * hcl.cos(phi) * (hcl.sin(theta)/hcl.cos(theta)) + q * hcl.sin(phi) * (hcl.sin(theta)/hcl.cos(theta))
        theta_dot[0] = q * hcl.cos(phi) - r * hcl.sin(phi)
        psi_dot[0] = r * hcl.cos(phi)/hcl.cos(theta) + q * hcl.sin(phi)/hcl.cos(theta)

        p_dot[0] = ((I_yy - I_zz)/I_xx) * r * q + ((tau_x+tau_wx)/I_xx)
        q_dot[0] = ((I_zz - I_xx)/I_yy) * p * r + ((tau_y+tau_wy)/I_yy)
        r_dot[0] = ((I_xx - I_yy)/I_zz) * p * q + ((tau_z+tau_wz)/I_zz)

        return (phi_dot[0], theta_dot[0], psi_dot[0], p_dot[0], q_dot[0], r_dot[0])
    
    def Hamiltonian(self, t_deriv, spatial_deriv):
        return t_deriv[0] * spatial_deriv[0] + t_deriv[1] * spatial_deriv[1] + t_deriv[2] * spatial_deriv[2] \
            + t_deriv[3] * spatial_deriv[3] + t_deriv[4] * spatial_deriv[4] + t_deriv[5] * spatial_deriv[5]


    def opt_ctrl_non_hcl(self, t, state, spat_deriv):

        uOpt1, uOpt2, uOpt3 = self.uMax[0], self.uMax[1], self.uMax[2]
        
       
        if self.uMode == "min":
                if spat_deriv[3] > 0:
                    uOpt1 = self.uMin[0]

                if spat_deriv[4] > 0:
                    uOpt2 = self.uMin[1]
                
                if spat_deriv[5] > 0:
                    uOpt3 = self.uMin[2]

        elif (self.uMode == "max"):
                if spat_deriv[3] < 0:
                    uOpt1 = self.uMin[0]

                if spat_deriv[4] < 0:
                    uOpt2 = self.uMin[1]

                if spat_deriv[5] < 0:
                    uOpt3 = self.uMin[2]
            
        else:
                raise ValueError("undefined uMode ...")
        
        #print (spat_deriv[3],spat_deriv[4],spat_deriv[5])
        
        return (uOpt1, uOpt2, uOpt3)


    def opt_dstb_non_hcl(self, t, state, spat_deriv):

        dOpt1,dOpt2,dOpt3 = self.dMax[0], self.dMax[1], self.dMax[2] 
        # add predefined values to avoid "none" type input when spat-deriv = 0

        if (self.dMode == "min"):

                if spat_deriv[3] < 0:
                    dOpt1 = self.dMin[0]

                if spat_deriv[4] < 0:
                    dOpt2 = self.dMin[1]
                
                if spat_deriv[5] < 0:
                    dOpt3 = self.dMin[2]

        elif (self.dMode == "max"):

                if spat_deriv[3] > 0:
                    dOpt1 = self.dMin[0]
                if spat_deriv[4] > 0:
                    dOpt2 = self.dMin[1]
                if spat_deriv[5] > 0:
                    dOpt3 = self.dMin[2]


        return (dOpt1, dOpt2, dOpt3)

    def dynamics_non_hcl(self, t, state, uOpt, dOpt):
        phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot = None, None, None, None, None, None


        phi = state[0] # roll
        theta = state[1] # pitch 
        psi = state[2] # yaw
        p = state[3]
        q = state[4]
        r = state[5]
        tau_x = uOpt[0]
        tau_y = uOpt[1]
        tau_z = uOpt[2]

        # some constants
        I_xx = self.Ixx
        I_yy = self.Iyy
        I_zz = self.Izz
        tau_wx = dOpt[0]
        tau_wy = dOpt[1]
        tau_wz = dOpt[2]

        # state dynamics equation
        phi_dot = p + r * np.cos(phi) * np.tan(theta) + q * np.sin(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = r * np.cos(phi)/np.cos(theta) + q * np.sin(phi)/np.cos(theta)

        p_dot = ((I_yy - I_zz)/I_xx) * r * q + ((tau_x + tau_wx)/I_xx)
        q_dot = ((I_zz - I_xx)/I_yy) * p * r + ((tau_y + tau_wy)/I_yy)
        r_dot = ((I_xx - I_yy)/I_zz) * p * q + ((tau_z + tau_wz)/I_zz)

        return (phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot)




class UAVSolution(object):
    def __init__(self):
        # warning: grid bound for each dimension should be close, not be too different. 
        self.grid_num_1 = 15
        self.grid_num_2 = 15
        self.grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([self.grid_num_2,self.grid_num_2,self.grid_num_2,self.grid_num_1,self.grid_num_1,self.grid_num_1]), [0,1,2])
        self.dyn = UAV6D(uMode="min", dMode="max")  # reaching problem
        self.lookback_length = 2  # Look-back length and time step of computation
        self.t_step = 0.001

        self.result = None

    def FasTrackTarget(self, grid, ignore_dims, center):
        """
        Customized definition of FasTrack Target, it is similar to Cylinder, but with no radius
        """
        data = np.zeros(grid.pts_each_dim)
        for i in range(grid.dims):
            if i not in ignore_dims:
                # This works because of broadcasting
                data = data + np.power(grid.vs[i] - center[i], 2)
        #data = np.sqrt(data)
        return data

    def get_fastrack(self):
        self.targ = self.FasTrackTarget(self.grid, [2], np.zeros(6)) 
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        compMethods = { "TargetSetMode": "maxVWithV0"}  # In this example, we compute based on FasTrack 
        slice = int((self.grid_num_1-1)/2)
        self.po = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[int((self.grid_num_1-1)/2),int((self.grid_num_1-1)/2),int((self.grid_num_1-1)/2)])
        self.result = HJSolver(self.dyn, self.grid, self.targ, tau, compMethods, self.po, saveAllTimeSteps=False)
        np.save("./fastrack_{}x{}.npy".format(self.grid_num_2, self.grid_num_1), self.result)
        print("saving the result ..., done!")

        return self.result, self.grid, slice
    
    def spa_deriv(self,index, V, g, periodic_dims):
        '''
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        index:
        V:
        g:
        periodic_dims:

    Returns:
        List of left and right spatial derivatives for each dimension

        '''
        spa_derivatives = []

        for dim, idx in enumerate(index):
            if dim == 0:
                left_index = []
            else:
                left_index = list(index[:dim])

            if dim == len(index) - 1:
                right_index = []
            else:
                right_index = list(index[dim + 1:])

            next_index = tuple(
                left_index + [index[dim] + 1] + right_index
            )
            prev_index = tuple(
            left_index + [index[dim] - 1] + right_index
            )
            if idx == 0:
                if dim in periodic_dims:
                    left_periodic_boundary_index = tuple(
                        left_index + [V.shape[dim] - 1] + right_index
                    )
                    left_boundary = V[left_periodic_boundary_index]
                else:
                    left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
                left_deriv = (V[index] - left_boundary) / g.dx[dim]
                right_deriv = (V[next_index] - V[index]) / g.dx[dim]
            elif idx == V.shape[dim] - 1:
                if dim in periodic_dims:
                    right_periodic_boundary_index = tuple(
                        left_index + [0] + right_index
                    )
                    right_boundary = V[right_periodic_boundary_index]
                else:
                    right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
                left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
                right_deriv = (right_boundary - V[index]) / g.dx[dim]
            else:
                left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
                right_deriv = (V[next_index] - V[index]) / g.dx[dim]

            spa_derivatives.append((left_deriv + right_deriv) / 2)

        return np.array(spa_derivatives)



    def next_state(self,dyn_sys, u, d, delta_t):
        """
        Simulate apply control to dynamical systems for delta_t time

        Args:
            dyn_sys: dynamic system
            u: control
            d: disturbance
            delta_t: duration of control

        Returns:

        """
        init_state = dyn_sys.x
        t_span = [0, delta_t]
        solution = solve_ivp(dyn_sys.dynamics_non_hcl, t_span, init_state, args=[u, d], dense_output=True)
        next_state = solution.y[:, -1]
        if next_state[2] < -np.pi:
            next_state[2] += 2 * np.pi
        elif next_state[2] > np.pi:
            next_state[2] -= 2 * np.pi
        return next_state

    def update_state(self,dyn_sys, next_state):
        dyn_sys.x = next_state


    def compute_opt_traj(self,grid: Grid, V, tau, dyn_sys, initial, subsamples=1, arriveAfter = None, obstVal = None): # default subsample in helperOC is 4
        """
    Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

    Args:
        grid:
        V:
        tau:
        dyn_sys:
        subsamples: Number of times opt_u and opt_d are calculated within dt

    Returns:
        traj: State of dyn_sys from time tau[-1] to tau[0]
        opt_u: Optimal control at each time
        opt_d: Optimal disturbance at each time

        """
        

        dt = (tau[1] - tau[0]) / subsamples
        dyn_sys.x = initial

        # first entry is dyn_sys at time tau[-1]
        # second entry is dyn_sys at time tau[-2]...
        traj = np.empty((len(tau), len(dyn_sys.x)))
        # Here it is chcking roll angle. traj[0] is all the roll states
        traj[0] = dyn_sys.x

        # flip the value with respect to the index
        #V = np.flip(V, grid.dims)

        opt_u = []
        opt_d = []
        t = []
        v_log= []
        t_earliest =0 # In helperOC, t_earliest starts at 1


        for iter in range(0,len(tau)):
 
            if iter < t_earliest:
                traj[iter] = np.array(dyn_sys.x)
                t.append(tau[iter])
                v_log.append(grid.get_value(V, dyn_sys.x))
                continue

            # Update trajectory, calculate gradient 
            gradient = self.spa_deriv(grid.get_index(dyn_sys.x), V, grid, periodic_dims=[0,1,2])
            for _ in range(subsamples):
                u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
                d = dyn_sys.opt_dstb_non_hcl(_, dyn_sys.x, gradient)
                #dNone = [0,0,0]
                bestU = u
                bestD = d
                nextState = self.next_state(dyn_sys, bestU, bestD, dt)
                self.update_state(dyn_sys, nextState)
                opt_u.append(u)
                opt_d.append(d)
            
            v_log.append(grid.get_value(V, dyn_sys.x))
            #the agent has entered the target
            if t_earliest == V.shape[-1]:
                traj[iter:] = np.array(dyn_sys.x)
                break

            #if iter != V.shape[-1]:
            traj[iter] = np.array(dyn_sys.x)
            
        return traj, opt_u, opt_d, t, v_log


    def getopt_traj(self,g,V,initial):
        small_number = 1e-5
        tau = np.arange(start=0, stop=self.lookback_length + small_number, step=self.t_step)
        [opt_traj, opt_u, opt_d, t,v_log] = self.compute_opt_traj (g,V,tau,self.dyn,initial)

        return opt_traj, opt_u, opt_d, t, v_log



def plot_2d(grid, V, index, slicecut):

    dims_plot = index
    dim1, dim2= dims_plot[0], dims_plot[1]
    V_2D = V[:,:,slicecut,slicecut,slicecut,slicecut]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y= np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]

    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
    ))
    fig.show()


def plot_optimal(opt_traj,v_log,opt_u,opt_d,initial):


    # plotting functions for traj 
    opt_traj_ = np.empty((6,len(opt_traj)))
    opt_u_d = np.empty((6,len(opt_traj)))
    
    for index in range(0,len(opt_traj)): 
        opt_traj_[0][index]=opt_traj[index][0]
        opt_traj_[1][index]=opt_traj[index][1]
        opt_traj_[2][index]=opt_traj[index][2]
        opt_traj_[3][index]=opt_traj[index][3]
        opt_traj_[4][index]=opt_traj[index][4]
        opt_traj_[5][index]=opt_traj[index][5]

        opt_u_d[0][index]=opt_u[index][0]
        opt_u_d[1][index]=opt_u[index][1]
        opt_u_d[2][index]=opt_u[index][2]
        opt_u_d[3][index]=opt_d[index][0]
        opt_u_d[4][index]=opt_d[index][1]
        opt_u_d[5][index]=opt_d[index][2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[0],name="Roll Angle",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[1],name="Pitch Angle",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[2],name="Yaw Angle",mode = "lines"))

    fig.update_layout(
    title="Angle (States) Over Timestep with disturbance with initial value{}".format(initial), 
    xaxis_title="Time Step", yaxis_title="Degrees"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[3],name="Roll Rate",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[4],name="Pitch Rate",mode = "lines"))
    fig.add_trace(go.Scatter(y=180/np.pi*opt_traj_[5],name="Yaw Rate",mode = "lines"))

    fig.update_layout(
    title="Angle Rate (States) Over Timestep (0.001s step, 2s time) with disturbance", xaxis_title="Time Step", yaxis_title="Degrees/sec"
    )
    fig.show()


    fig = go.Figure(data=go.Scatter(
            y=v_log, line=dict(color="crimson")),
            layout_title_text="Value Function with initial value{}".format(initial),
            layout_yaxis_range=[0,2]

        )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=opt_u_d[0],name="tau_x",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[1],name="tau_y",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[2],name="tau_z",mode = "lines"))

    fig.update_layout(
    title="Optimal Control Over Timestep (0.001s step, 2s time) with disturbance", xaxis_title="Time Step", yaxis_title="Torques"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=opt_u_d[3],name="tau_wx",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[4],name="tau_wy",mode = "lines"))
    fig.add_trace(go.Scatter(y=opt_u_d[5],name="tau_wz",mode = "lines"))

    fig.update_layout(
    title="Optimal Disturbance Over Timestep (0.001s step, 2s time)", xaxis_title="Time Step", yaxis_title="Torques"
    )
    fig.show()

def maxDiff(a):
    vmin = a[0]
    dmax = 0
    for i in range(len(a)):
        if (a[i] < vmin):
            vmin = a[i]
        elif (a[i] - vmin > dmax):
            dmax = a[i] - vmin
    return dmax


def evaluation (iteration, threshold):


    roll_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    pitch_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    roll_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]
    pitch_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]

    initial = np.empty((iteration,6))

    for i in range(iteration):
        initial[i] = [random.choice(roll_range),random.choice(pitch_range),0,random.choice(roll_rate),random.choice(pitch_rate),0]
    initial_nodu=np.unique(initial,axis=0)

    for i in range(len(initial_nodu)):
        [opt_traj, opt_u, opt_d, t, v_log] = uavsol.getopt_traj(grid,V,initial_nodu[i])
        # plot optimal control
        if maxDiff(v_log) >threshold:
            print ("These initial values lead to an unstable system under disturbance")
            print (initial_nodu[i])
            plot_optimal(opt_traj,v_log,opt_u,opt_d,initial_nodu[i])
        else: 
            print (initial_nodu[i],"gives a stable system")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description ='Calculate Value Function using HJ')
    
    parser.add_argument('--load_data', type=bool, default=True, help="load precollected data")
    parser.add_argument('--evaluate', type=bool, default=False, help= "plot evaluation")
    args = parser.parse_args()
    
    uavsol = UAVSolution()
    if args.load_data is True: 
        slicecut = 7  #for 15*15
        V = np.load('/localhome/xza213/quadrotor-stabilization/adversarial_generation/FasTrack_data/fastrack_1.5_15x15.npy')
        grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])
        
    else: 
        [V,grid,slicecut] = uavsol.get_fastrack()

    ## if you want to see the 2D value function plot with prelaoding please uncomment the code below
    # you should comment out the line where we calculated the [V,grid,slicecut] if you want to plot pre-calculated data

    '''    
    slicecut = 7  #for 15*15
    V = np.load('fastrack_0.5_15x15.npy')
    grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])
    '''

    plot_2d(grid, V, [1,2], slicecut) # we are plotting first two dimensions
    
    if args.evaluate is True:
    ## if you want to evaluate your calculated control/disturbance, please uncomment the following code
    # The code randomly sampled states from the pool, and check whether the control/disturbance gives a stable system
    # It will generate the plots for unstable systems in your broswer
    # The creteria is "threshold" -> The difference between maxvalue and minvalue in value functions for all time step

        iteration = 50 # how many samples you want to generate
        threshold = 5 # max diff of values
        evaluation (iteration, threshold) 
    




