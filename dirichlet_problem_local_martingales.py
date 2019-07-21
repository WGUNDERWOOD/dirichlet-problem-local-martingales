import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def data_to_polygon(data, ref_height, color, alpha):

    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]

    v = []
    for k in range(0, len(xs) - 1):
        x = [xs[k], xs[k+1], xs[k+1], xs[k]]
        y = [ys[k], ys[k+1], ys[k+1], ys[k]]
        z = [zs[k], zs[k+1], ref_height, ref_height]
        v.append(list(zip(x, y, z))) 
    poly3dCollection = Poly3DCollection(v, color=color, alpha=alpha)
    
    return poly3dCollection

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
