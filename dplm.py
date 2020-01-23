import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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


def get_region_boundary(num_samples):

    offset_circles_coords = get_offset_circles((1,0),(0,0),5,2,num_samples)

    return offset_circles_coords


def apply_to_coords(coords, func):

    zs = np.apply_along_axis(func, 1, coords).reshape(-1,1)

    return zs


def phi(xy):

    x = xy[0]
    y = xy[1]

    return x**2 + y + 10


def data_to_polygon(boundary_coords, boundary_values, ref_height, color, alpha):

    xs = boundary_coords[:,0]
    ys = boundary_coords[:,1]
    zs = boundary_values

    v = []
    for k in range(0, len(xs) - 1):
        x = [xs[k], xs[k+1], xs[k+1], xs[k]]
        y = [ys[k], ys[k+1], ys[k+1], ys[k]]
        z = [zs[k], zs[k+1], ref_height, ref_height]
        v.append(list(zip(x, y, z)))

    poly3dCollection = Poly3DCollection(v)
    poly3dCollection.set_alpha(alpha)
    poly3dCollection.set_facecolor(color)

    return poly3dCollection




# simulate BM

def sim_bm(x, T, num_samples):

    mesh_size = T/num_samples
    sd = np.sqrt(mesh_size)

    normal_vec = np.random.normal(loc=0, scale=sd, size=num_samples)
    normal_vec[0] = normal_vec[0] + x
    bm = np.cumsum(normal_vec)

    return bm


def sim_2d_bm(xy, T, num_samples):

    x_bm = sim_bm(xy[0], T, num_samples)
    y_bm = sim_bm(xy[1], T, num_samples)

    bm_2 = np.array([x_bm, y_bm]).T

    return bm_2


def inside_U(xy):

    x = xy[0]
    y = xy[1]

    if (x-1)**2 + y**2 >= 25:
        return False

    elif x**2 + y**2 <= 4:
        return False

    else:
        return True


def terminal_value(bm_2d):

    terminal_point = bm_2d[-1]
    terminal_val = phi(terminal_point)

    return terminal_val


def up_to_escape(bm_2d):

    still_in_U = np.apply_along_axis(inside_U, 1, bm_2d)

    last_time_before_escape = list(still_in_U).index(False)
    bm_up_to_escape = bm_2d[0:(last_time_before_escape + 1)]

    return bm_up_to_escape


def simulate_many_bms(xy, M, T, num_samples):

    values = M*[None]

    for m in range(M):
        bm = sim_2d_bm(xy, T, num_samples)
        escape_value = terminal_value(up_to_escape(bm))
        values[m] = escape_value

    mean_value = np.mean(values)

    return mean_value


def make_final_surface(M, T, num_samples, fidelity):

    x_scope = np.arange(-4,6, fidelity)
    y_scope = np.arange(-5,5, fidelity)
    mesh = product(x_scope, y_scope)

    surface = []

    n_iters = len(x_scope) * len(y_scope)
    i = 1

    for xy in mesh:
        print("make_final_surface: {}/{}".format(i, n_iters))
        i = i+1
        if inside_U(xy):
            f_xy = simulate_many_bms(xy, M, T, num_samples)
            surface.append([xy[0], xy[1], f_xy])

    return surface





# plots

def plot_region():

    # get data
    num_samples = 50
    boundary_coords = get_region_boundary(num_samples)

    # set up plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # region
    plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], linewidth=0, color='lightsteelblue')

    # boundary
    ax.plot(boundary_coords[0:num_samples,0], boundary_coords[0:num_samples,1], color='slateblue', linewidth=2, zorder=2)
    ax.plot(boundary_coords[num_samples:,0], boundary_coords[num_samples:,1], color='slateblue', linewidth=2, zorder=2)

    # text
    ax.text(x=3.8, y=0, s='$U$', fontsize=20, zorder=6)
    ax.text(x=2, y=-5.7, s='$\partial U$', fontsize=20, zorder=6)

    # save
    plt.axis('off')
    plt.savefig("./graphics/plot_region.png")


def plot_region_and_boundary_condition():

    num_samples = 50
    boundary_coords = get_region_boundary(num_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)

    # set up plot
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0))

    # vertical shading
    ax.add_collection3d(data_to_polygon(boundary_coords[0:num_samples,:], boundary_values[0:num_samples], 0, 'r', 0.5))
    ax.add_collection3d(data_to_polygon(boundary_coords[num_samples:,:], boundary_values[num_samples:,:], 0, 'r', 0.5))

    # region boundary
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=0, color='slateblue', linewidth=2, zorder=4)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=0, color='slateblue', linewidth=2, zorder=4)

    # phi values
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=boundary_values[0:num_samples,0], color='r', linewidth=2, zorder=5)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=boundary_values[num_samples:,0], color='r', linewidth=2, zorder=5)

    # text
    ax.text(x=3.8, y=0, z=0, s='$U$', fontsize=20, zorder=6)
    ax.text(x=2, y=-5.7, z=0, s='$\partial U$', fontsize=20, zorder=6)
    ax.text(x=-3, y=7.5, z=0, s='$\phi(\partial U)$', fontsize=20, zorder=6)

    # axis limits
    ax.set_xlim([-4,6])
    ax.set_ylim([-5,5])
    ax.set_zlim([0,40])
    plt.axis('off')

    # viewpoint
    ax.view_init(elev=60, azim=250)

    plt.savefig("./graphics/plot_region_and_boundary_condition.png")

    return


def plot_few_bm_paths():

    # get data
    num_samples = 10000
    boundary_coords = get_region_boundary(num_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)
    x = (3,0)
    T = 20

    # set up plot
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0))

    # vertical shading
    ax.add_collection3d(data_to_polygon(boundary_coords[0:num_samples,:], boundary_values[0:num_samples], 0, 'r', 0.5))
    ax.add_collection3d(data_to_polygon(boundary_coords[num_samples:,:], boundary_values[num_samples:,:], 0, 'r', 0.5))

    # region boundary
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=0, color='slateblue', linewidth=2, zorder=4)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=0, color='slateblue', linewidth=2, zorder=4)

    # phi values
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=boundary_values[0:num_samples,0], color='r', linewidth=2, zorder=5)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=boundary_values[num_samples:,0], color='r', linewidth=2, zorder=7)

    # start value
    ax.plot([x[0]], [x[1]], 'ko', zorder=8, markersize=3)

    # BMs
    np.random.seed(seed=2)
    for i in range(3):
        col = ['brown','red','green'][i]
        bm = up_to_escape(sim_2d_bm(x, T, num_samples))
        xs = 2 * [bm[-1,0]]
        ys = 2 * [bm[-1,1]]
        zs = [0, terminal_value(bm)]
        ax.plot(xs, ys, zs, '-o', markersize=3, zorder=8, linewidth=1, color=col)
        ax.plot(bm[:,0], bm[:,1], zorder=6, linewidth=0.5, color=col)

    # text
    ax.text(x=3.8, y=0, z=0, s='$U$', fontsize=20, zorder=6)
    ax.text(x=2, y=-5.7, z=0, s='$\partial U$', fontsize=20, zorder=6)
    ax.text(x=-3, y=7.5, z=0, s='$\phi(\partial U)$', fontsize=20, zorder=6)

    # axis limits
    ax.set_xlim([-4,6])
    ax.set_ylim([-5,5])
    ax.set_zlim([0,40])
    plt.axis('off')

    # viewpoint
    ax.view_init(elev=60, azim=250)

    plt.savefig("./graphics/plot_few_bm_paths.png", dpi=1000)

    return


def plot_final_surface():

    M = 10
    T = 50
    num_samples = 100
    fidelity = 0.2

    np.random.seed(seed=2)

    surface = make_final_surface(M, T, num_samples, fidelity)
    surface = np.array(surface)

    boundary_coords = get_region_boundary(num_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)

    # set up plot
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0))

    # vertical shading
    #ax.add_collection3d(data_to_polygon(boundary_coords[0:num_samples,:], boundary_values[0:num_samples], 0, 'r', 0.5))
    #ax.add_collection3d(data_to_polygon(boundary_coords[num_samples:,:], boundary_values[num_samples:,:], 0, 'r', 0.5))

    # region boundary
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=0, color='slateblue', linewidth=2, zorder=4)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=0, color='slateblue', linewidth=2, zorder=4)

    # phi values
    ax.plot(xs=boundary_coords[0:num_samples,0], ys=boundary_coords[0:num_samples,1], zs=boundary_values[0:num_samples,0], color='r', linewidth=2, zorder=5)
    ax.plot(xs=boundary_coords[num_samples:,0], ys=boundary_coords[num_samples:,1], zs=boundary_values[num_samples:,0], color='r', linewidth=2, zorder=5)

    # surface
    cmap = cm.autumn
    norm = Normalize(vmin=min(surface[:,2]), vmax=max(surface[:,2]))
    cols = cmap(norm(surface[:,2]))
    ax.scatter(surface[:,0], surface[:,1], surface[:,2], linewidths=0, zorder=7, color=cols, s=2)

    # text
    ax.text(x=3.8, y=0, z=0, s='$U$', fontsize=20, zorder=6)
    ax.text(x=2, y=-5.7, z=0, s='$\partial U$', fontsize=20, zorder=6)
    ax.text(x=-3, y=7.5, z=0, s='$\phi(\partial U)$', fontsize=20, zorder=6)

    # axis limits
    ax.set_xlim([-4,6])
    ax.set_ylim([-5,5])
    ax.set_zlim([0,40])
    plt.axis('off')

    # viewpoint
    ax.view_init(elev=60, azim=250)

    plt.savefig("./graphics/plot_final_surface.png", dpi=1000)
