# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:04:27 2022

@author: bchan
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_profile(v: np.ndarray, title_text: str="Speed Profile", **kwargs) -> plt.Figure:
    fig = plt.figure(dpi=300)
    plt.imshow(v, **kwargs)
    plt.colorbar()
    plt.title(title_text)
    
    return fig


def plot_quiver(vx: np.ndarray, vy: np.ndarray, title_text: str="Velocity Field", skip: int=10, **kwargs) -> plt.Figure:
    nx, ny = vx.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    [x1, y1] = np.meshgrid(x, y)
    
    # v = vx != 0
    # vx[v == 0] = np.nan
    # vy[v == 0] = np.nan
    vx_copy = vx.copy()
    vy_copy = vy.copy()
    fig = plt.figure(dpi=300)
    binary, fig = plot_binary(vx, cmap="Pastel1")
    vx_copy[binary == 0] = np.nan
    vy_copy[binary == 0] = np.nan
    # plt.imshow(v, cmap='Pastel1')
    plt.quiver(x1[::skip, ::skip], y1[::skip, ::skip], vx_copy[::skip, ::skip], vy_copy[::skip, ::skip], **kwargs)
    
    plt.title(title_text)

    return fig


def plot_streamlines(v_x: np.ndarray, v_y: np.ndarray, title_text: str="Streamlines", **kwargs) -> plt.Figure:
    nx, ny = v_x.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    [x1, y1] = np.meshgrid(x, y)
    
    start_y = np.arange(0, ny, 10)
    start_x = np.zeros_like(start_y)
    start_point_array = np.array([start_x, start_y]).T
    
    fig = plt.figure(dpi=300)
    # plt.imshow(uv, cmap='Pastel1')
    _, fig = plot_binary(v_x, cmap="Pastel1")
    plt.streamplot(x1, y1, v_x, v_y, start_points=start_point_array, **kwargs)
    
    plt.title(title_text)
    
    return fig


def plot_binary(vel_data: np.ndarray, **kwargs) -> plt.Figure():
    binary = vel_data != 0
    fig = plt.figure(dpi=300)
    plt.imshow(binary, **kwargs)
    
    return binary, fig
