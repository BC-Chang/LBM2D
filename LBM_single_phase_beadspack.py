"""
D2Q9 BGK LBM simulation of single phase gravity driven flow through
a 2D cross section of a glass bead pack

Author: Bernard Chang, The University of Texas at Austin
Adopted from: Rui Xu, The University of Texas at Austin
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tifffile
from time import perf_counter_ns

# Initialization
tau = 1.0  # Relaxation time
g = 0.00001  # Gravity or other force
density = 1
tf = 10  # Maximum number of iteration steps
precision = 1.E-5  # Convergence criterion
vold = 1000

check_convergence = 100  # Check convergence every [check_convergence] time steps


# Read in and create geometry
data = tifffile.imread("beads_dry.slice150.sub.tif")  # Data labels are 0 and 255
data = data / 255  # Make Data 0 and 1s

# Solid nodes are labeled 1, fluid nodes are labeled 0
is_solid_node = data

nx, ny = data.shape

# Initialize distribution functions
f = np.array([4./9.,
              1./9.,
              1./9.,
              1./9.,
              1./9.,
              1./36.,
              1./36.,
              1./36.,
              1./36.]) * density
# Broadcast to 3D array with each slice corresponding to
f = np.tile(f[:, np.newaxis, np.newaxis], (9, nx, ny))
# Allocate memory to equilibrium functions
feq = np.empty_like(f)
ftmp = feq.copy()

# Define lattice velocity vectors. Transpose to make it a column vector
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]).T
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1]).T

# Each point has x-component ex, and y-component ey
u_x = np.empty((nx, ny), dtype=np.double)
u_y = np.empty((nx, ny), dtype=np.double)

# Node density
rho = np.zeros((nx, ny), dtype=np.double)

# # Begin time loop
tic = perf_counter_ns()
for ts in range(tf):
    print(f"{ts = }")  # Print timestep

    # Compute macroscopic density, rho and velocity.

    for j in range(ny):
        for i in range(nx):
            u_x[j, i] = 0.
            u_y[j, i] = 0.
            rho[j, i] = 0.

            if is_solid_node[j, i] == 0:
                for a in range(9):
                    rho[j, i] = rho[j, i] + f[a, j, i]
                    u_x[j, i] = u_x[j, i] + ex[a] * f[a, j, i]
                    u_y[j, i] = u_y[j, i] + ey[a] * f[a, j, i]


                u_x[j, i] = u_x[j, i] / rho[j, i]
                u_y[j, i] = u_y[j, i] / rho[j, i]

    # Compute equilibrium distribution function, feq
    f1 = 3.
    f2 = 9./2.
    f3 = 3./2.

    for j in range(ny):
        for i in range(nx):
            if is_solid_node[j, i] == 0:
                rt0 = 4./9. * rho[j, i]
                rt1 = 1./9. * rho[j, i]
                rt2 = 1./36. * rho[j, i]
                # Add forcing to velocity
                ueqxij = u_x[j, i] + tau * g
                ueqyij = u_y[j, i] + tau * g
                uxsq = ueqxij ** 2.
                uysq = ueqyij ** 2.
                uxuy5 = ueqxij + ueqyij
                uxuy6 = -ueqxij + ueqyij
                uxuy7 = -ueqxij - ueqyij
                uxuy8 = ueqxij - ueqyij
                usq = uxsq + uysq

                feq[0, j, i] = rt0 * (1 - f3*usq)
                feq[1, j, i] = rt1 * (1 + f1 * ueqxij + f2 * uxsq - f3 * usq)
                feq[2, j, i] = rt1 * (1 + f1 * ueqyij + f2 * uysq - f3 * usq)
                feq[3, j, i] = rt1 * (1 - f1 * ueqxij + f2 * uxsq - f3 * usq)
                feq[4, j, i] = rt1 * (1 - f1 * ueqyij + f2 * uysq - f3 * usq)
                feq[5, j, i] = rt2 * (1 + f1 * uxuy5 + f2 * uxuy5**2 - f3 * usq)
                feq[6, j, i] = rt2 * (1 + f1 * uxuy6 + f2 * uxuy6**2 - f3 * usq)
                feq[7, j, i] = rt2 * (1 + f1 * uxuy7 + f2 * uxuy7**2 - f3 * usq)
                feq[8, j, i] = rt2 * (1 + f1 * uxuy8 + f2 * uxuy8 ** 2 - f3 * usq)

    # Collision Step

    for j in range(ny):
        for i in range(nx):
            if is_solid_node[j, i] == 1:
                # Standard Bounceback
                # Node 1 & 3
                tmp = f[1, j, i]
                f[1, j, i] = f[3, j, i]
                f[3, j, i] = tmp

                # Node 2 & 4
                tmp = f[2, j, i]
                f[2, j, i] = f[4, j, i]
                f[4, j, i] = tmp

                # Node 5 & 7
                tmp = f[5, j, i];
                f[5, j, i] = f[7, j, i];
                f[7, j, i] = tmp

                # Node 6 & 8
                tmp = f[6, j, i];
                f[6, j, i] = f[8, j, i];
                f[8, j, i] = tmp

            else:
                # Regular collision away from solid boundary
                for a in range(9):
                    f[a, j, i] = f[a, j, i] - (f[a, j, i] - feq[a, j, i]) / tau




toc = perf_counter_ns()
print(f"Elapsed Time: {(toc - tic)*1E-9}s")


# f = np.apply_along_axis(f *slicewise_initial_distribution, axis=0)










