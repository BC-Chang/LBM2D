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
f = np.broadcast_to(f[:, np.newaxis, np.newaxis], (9, nx, ny))
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
toc = perf_counter_ns()
print(f"Elapsed Time: {(toc - tic)*1E-9}s")


# f = np.apply_along_axis(f *slicewise_initial_distribution, axis=0)










