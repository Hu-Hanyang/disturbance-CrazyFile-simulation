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


def distur_gener(states, disturbance):


    def opt_ctrl_non_hcl(uMax, spat_deriv):
        
        uOpt1, uOpt2, uOpt3 = uMax[0],uMax[1], uMax[2]
        uMin = -uMax
   
        if spat_deriv[3] > 0:
            uOpt1 = uMin[0]

        if spat_deriv[4] > 0:
            uOpt2 = uMin[1]
                    
        if spat_deriv[5] > 0:
            uOpt3 = uMin[2]


            
        return (uOpt1, uOpt2, uOpt3)
        
    def spa_deriv(index, V, g, periodic_dims):
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

            return  spa_derivatives  # np.array(spa_derivatives)  # Hanyang: change the type of the return




            dyn_sys.x = next_state

    def opt_dstb_non_hcl(dMax, spat_deriv):

        dOpt1,dOpt2,dOpt3 = dMax[0],dMax[1],dMax[2]
        dMin = -dMax
        # Joe:
        if spat_deriv[3] > 0:
            dOpt1 = dMin[0]
        if spat_deriv[4] > 0:
            dOpt2 = dMin[1]
        if spat_deriv[5] > 0:
            dOpt3 = dMin[2]
        # Hanyang: try different calculation
        # if spat_deriv[3] < 0:
        #     dOpt1 = dMin[0]
        # if spat_deriv[4] < 0:
        #     dOpt2 = dMin[1]
        # if spat_deriv[5] < 0:
        #     dOpt3 = dMin[2]

        return (dOpt1, dOpt2, dOpt3)

    def compute_opt_traj(grid: Grid, V, states, umax, dmax): 
            """
        Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

        Args:
            grid:
            V:
            current states:
            maximum control
            maximum disturbance


        Returns:
            opt_u: Optimal control at current time step
            opt_d: Optimal disturbance at current time step

            """
            
            gradient = spa_deriv(grid.get_index(states), V, grid, periodic_dims=[0,1,2])
            u = opt_ctrl_non_hcl(umax, gradient)
            d = opt_dstb_non_hcl(dmax, gradient)
                
            return u,d

    
    umax=np.array([5.3*10**-3,  5.3*10**-3,  1.43*10**-4])
    # dmax = 0*umax
    if disturbance < 2:
        V = np.load(f'./adversarial_generation/FasTrack_data/fastrack_{disturbance}_15x15.npy')
    else: 
        V = np.load('./adversarial_generation/FasTrack_data/fastrack_15x15.npy')
    dmax = disturbance * umax
    # if disturbance == 0: 
    #     V = np.load('fastrack_0_15x15.npy')
    #     dmax = 0*umax
    # elif disturbance == 0.5: 
    #     V = np.load('fastrack_0.5_15x15.npy')
    #     dmax = 0.5*umax
    # elif disturbance == 1: 
    #     V = np.load('fastrack_1_15x15.npy')
    #     dmax = 1*umax
    # elif disturbance == 1.5: 
    #     V = np.load('fastrack_1.5_15x15.npy')
    #     dmax = 1.5*umax
    # elif disturbance == 2: 
    #     V = np.load('fastrack_15x15.npy')
    #     dmax = 2*umax
    # else: 
    #     print ("Sorry, we could not find the corresponding value function for your input disturbance, please calculate a new value function using file name /Fastrack/ ")

    grid = Grid(np.array([-math.pi/2.4, -math.pi/2.4, -math.pi/2.4, -math.pi, -math.pi, -math.pi]), np.array([math.pi/2.4, math.pi/2.4, math.pi/2.4, math.pi, math.pi, math.pi]), 6, np.array([15,15,15,15,15,15]), [0,1,2])

    [opt_u, opt_d] = compute_opt_traj(grid,V,states,umax,dmax)

    return opt_u, opt_d
    

def quat2euler(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

if __name__ == "__main__":

    roll_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    pitch_range = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    roll_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]
    pitch_rate = [-2.5,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5]

    disturbance = 1.5

    col = 15
    initial = np.empty((col,6))

    for i in range(col):
        initial[i] = [random.choice(roll_range),random.choice(pitch_range),0,random.choice(roll_rate),random.choice(pitch_rate),0]

    initial_nodu=np.unique(initial,axis=0)

    for i in range(len(initial_nodu)):

        initial_point = initial_nodu[i]
    
        [u,d] = distur_gener(initial_point,disturbance)
        print (d)




