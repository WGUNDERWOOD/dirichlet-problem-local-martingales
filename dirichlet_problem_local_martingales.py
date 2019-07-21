import numpy as np
import matplotlib.pyplot as plt

# draw a region

def get_circle(center, radius, num_samples):

    thetas = np.linspace(0, 2*np.pi, num=num_samples, endpoint=True)

    xs = radius*np.cos(thetas) + center[0]
    ys = radius*np.sin(thetas) + center[1]

    circle_coords = np.array([xs,ys]).T

    return circle_coords


def get_offset_circles(center1, center2, radius1, radius2, num_samples):

    circle1_coords = get_circle(center1, radius1, num_samples)
    circle2_coords = get_circle(center2, radius2, num_samples)

    circle2_coords = np.flipud(circle2_coords)

    offset_circles_coords = np.concatenate((circle1_coords, circle2_coords), axis=0)

    return offset_circles_coords
   

def apply_to_coords(coords, func):

    n_rows = np.shape(coords)[0]
    zs = np.apply_along_axis(func, 1, coords).reshape(n_rows,1)
    xyzs = np.concatenate((coords,zs), axis=1).reshape(n_rows,3)

    return xyzs

# simulate one-dimensional BM

def sim_bm(T, mesh_size):

    num_samples = int(T/mesh_size)
    sd = np.sqrt(mesh_size)

    normal_vec = np.random.normal(loc=0, scale=sd, size=num_samples)
    bm = np.cumsum(normal_vec)

    return bm


# simulate two-dimensional BM

def sim_2d_bm(T, mesh_size):
    
    x_bm = sim_bm(T, mesh_size)
    y_bm = sim_bm(T, mesh_size)
    
    bm_2 = np.array([x_bm, y_bm])

    return bm_2
