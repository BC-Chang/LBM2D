"""
D2Q9 BGK LBM simulation of single phase gravity driven flow through
a 2D cross section of a glass bead pack

Author: Bernard Chang, The University of Texas at Austin
Adopted from: Rui Xu, The University of Texas at Austin
"""

# Import packages
import numpy as np
# import jax as np
import matplotlib.pyplot as plt
from plotting_utils import plot_profile, plot_quiver, plot_streamlines
import scipy as sc
import tifffile
from time import perf_counter_ns
import tqdm


def run_lbm(data):
    
    # Initialization
    tau = 1.0  # Relaxation time
    g = 0.00001  # Gravity or other force
    density = 1.
    tf = 10001  # Maximum number of iteration steps
    precision = 1.E-5  # Convergence criterion
    vold = 1000
    eps = 1E-6
    
    check_convergence = 30  # Check convergence every [check_convergence] time steps
    
    
    # Indices of fluid nodes
    fluid_id = np.argwhere(data == 0)
    fx = fluid_id[:, 0]
    fy = fluid_id[:, 1]
    
    # Indices of solid nodes
    solid_id = np.argwhere(data == 1)
    sx = solid_id[:, 0]
    sy = solid_id[:, 1]
    
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
                  1./36.], dtype=np.double) * density
    
    # Broadcast to 3D array with each slice corresponding to
    f = np.tile(f[:, np.newaxis, np.newaxis], (nx, ny)).astype(np.single)
    
    # Allocate memory to equilibrium functions
    feq = np.empty_like(f, dtype=np.double)
    
    # Define lattice velocity vectors. Transpose to make it a column vector
    ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.double)
    ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.double)
    
    # Each point has x-component ex, and y-component ey
    u_x = np.empty((nx, ny), dtype=np.double)
    u_y = np.empty((nx, ny), dtype=np.double)
    
    # Node density
    rho = np.zeros((nx, ny), dtype=np.double)
    
    # For equilibrium distribution function (feq) calculation 
    f1 = 3.
    f2 = 9./2.
    f3 = 3./2.
    
    # # Begin time loop
    tic = perf_counter_ns()
    for ts in tqdm(range(tf)):
        # print(f"{ts = }")  # Print timestep
    
        # Compute macroscopic density, rho and velocity.
        u_x = np.zeros((nx, ny), dtype=np.double)
        u_y = np.zeros((nx, ny), dtype=np.double)
        rho = np.zeros((nx, ny), dtype=np.double)
        
        rho[fx, fy] += np.sum(f[:, fx, fy], axis=0)
        u_x[fx, fy] += np.sum(ex[:, None] * f[:, fx, fy], axis=0)
        u_y[fx, fy] += np.sum(ey[:, None] * f[:, fx, fy], axis=0)
        
        u_x[fx, fy] = u_x[fx, fy] / rho[fx, fy]
        u_y[fx, fy] = u_y[fx, fy] / rho[fx, fy]
        
    
    
        # Add forcing to velocity
        ueqxij = u_x[fx, fy] + tau * g
        ueqyij = u_y[fx, fy]
        uxsq = ueqxij ** 2
        uysq = ueqyij ** 2
        uxuy5 = ueqxij + ueqyij
        uxuy6 = -ueqxij + ueqyij
        uxuy7 = -ueqxij + (-ueqyij)
        uxuy8 = ueqxij + (-ueqyij)
        usq = uxsq + uysq
        
        # Compute equilibrium distribution function, feq
        rt0 = 4./9. * rho[fx, fy]
        rt1 = 1./9. * rho[fx, fy]
        rt2 = 1./36. * rho[fx, fy]
        
        feq[0, fx, fy] = rt0*(1. - f3 * usq)
        feq[1, fx, fy] = rt1*(1. + f1 * ueqxij + f2 * uxsq - f3 * usq)
        feq[2, fx, fy] = rt1*(1. + f1 * ueqyij + f2 * uysq - f3 * usq)
        feq[3, fx, fy] = rt1*(1. - f1 * ueqxij + f2 * uxsq - f3 * usq)
        feq[4, fx, fy] = rt1*(1. - f1 * ueqyij + f2 * uysq - f3 * usq)
        feq[5, fx, fy] = rt2*(1. + f1 * uxuy5  + f2 * uxuy5 * uxuy5 - f3 * usq)
        feq[6, fx, fy] = rt2*(1. + f1 * uxuy6  + f2 * uxuy6 * uxuy6 - f3 * usq)
        feq[7, fx, fy] = rt2*(1. + f1 * uxuy7  + f2 * uxuy7 * uxuy7 - f3 * usq)
        feq[8, fx, fy] = rt2*(1. + f1 * uxuy8  + f2 * uxuy8 * uxuy8 - f3 * usq)
    
        # Collision Step
        # Standard Bounceback for Solid Nodes
        # Left-Right
        tmp = f[1, sx, sy]
        f[1, sx, sy] = f[3, sx, sy]
        f[3, sx, sy] = tmp
        
        # Up-Down
        tmp = f[2, sx, sy]
        f[2, sx, sy] = f[4, sx, sy]
        f[4, sx, sy] =  tmp
        
        # Top Right - Bottom Left
        tmp = f[5, sx, sy]
        f[5, sx, sy] = f[7, sx, sy]
        f[7, sx, sy] =  tmp
        
        # Top Left - Bottom Right
        tmp = f[6, sx, sy]
        f[6, sx, sy] = f[8, sx, sy]
        f[8, sx, sy] =  tmp
        
        # Regular collision in fluid nodes
        f[:, fx, fy] -= (f[:, fx, fy] - feq[:, fx, fy]) / tau
    
    
        # Streaming Step
        f[1] = np.roll(f[1], 1, axis=1)
        f[2] = np.roll(f[2], 1, axis=0)
        f[3] = np.roll(f[3], -1, axis=1)
        f[4] = np.roll(f[4], -1, axis=0)
        
        f[5] = np.roll(f[5], (1, 1), axis=(0,1))
        f[6] = np.roll(f[6], (-1, 1), axis=(1,0))
        f[7] = np.roll(f[7], (-1, -1), axis=(0,1))
        f[8] = np.roll(f[8], (1, -1), axis=(1,0))
        
        u = np.sqrt(u_x**2 + u_y**2)
        # Plot the time step and check convergence every check_convergence time step
        if ts % check_convergence == 0:
            
            vnew = np.mean(u)
            error = np.abs(vold - vnew) / (vold+eps)
            vold = vnew
    
            if error < precision:
                print(f'Simulation has converged in {ts} time steps')
                break
    
        if ts == tf:
            print('Reached maximum iterations')
        
    toc = perf_counter_ns()
    print(f"Elapsed Time: {(toc - tic)*1E-9}s")
    
    return u_x, u_y, u


if __name__ == '__main__':

    # Read in and create geometry
    data = tifffile.imread("beads_dry.slice150.sub.tif")  # Data labels are 0 and 255
    data = data / 255  # Make Data 0 and 1s
    # data = data.astype(np.uint8)
    # data = data[0:100, :]
    u_x, u_y, u = run_lbm(data)
    # profile_fig = plot_profile(u, cmap='jet')
    # quiver_fig = plot_quiver(u_x, u_y)
    streamline_fig = plot_streamlines(u_x, u_y, arrowsize=0, density=6)
    
    # profile_fig.savefig("output_figures/velocity_profile.png")
    # quiver_fig.savefig("output_figures/velocity_field.png")
    streamline_fig.savefig("output_figures/streamlines.png")
    
    # plt.figure(dpi=300)
    # plt.imshow(u, cmap='jet')
    # plt.tight_layout()
    # plt.colorbar()
    # plt.title('Speed Profile')
    # plt.show()


