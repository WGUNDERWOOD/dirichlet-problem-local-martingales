import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# drawing
# -----------------------------------------------------------------------------------

def get_circle(center, radius, n_draw_samples):

    '''get array of points forming a circle'''

    thetas = np.linspace(0, 2*np.pi, num=n_draw_samples, endpoint=True)

    xs = radius*np.cos(thetas) + center[0]
    ys = radius*np.sin(thetas) + center[1]

    circle_coords = np.array([xs,ys]).T

    return circle_coords


def get_offset_circles(center1, center2, radius1, radius2, n_draw_samples):

    '''get array of points forming two offset circles'''

    circle1_coords = get_circle(center1, radius1, n_draw_samples)
    circle2_coords = get_circle(center2, radius2, n_draw_samples)

    circle2_coords = np.flipud(circle2_coords)

    offset_circles_coords = np.concatenate((circle1_coords, circle2_coords), axis=0)

    return offset_circles_coords


def get_region_boundary(n_draw_samples):

    '''get array of points for specific region U'''

    offset_circles_coords = get_offset_circles((1,0), (0,0), 5, 2, n_draw_samples)

    return offset_circles_coords


def apply_to_coords(coords, func):

    '''apply a function pointwise to a list of coordinates'''

    zs = np.apply_along_axis(func, 1, coords).reshape(-1,1)

    return zs


def phi(xy):

    '''the boundary condition'''

    x = xy[0]
    y = xy[1]

    return x**2 + y + 10


def data_to_polygon(boundary_coords, boundary_values, ref_height, color, alpha):

    '''convert array of points to polygon for plotting'''

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




# bm
# -----------------------------------------------------------------------------------

def first_false(logical_values):

    '''get the first false entry in a logical list and return zero if there are none'''

    logical_list = list(logical_values)

    if False in logical_list:
        return logical_list.index(False)

    else:
        return 0


def stop_at_time(points, stop_time):

    '''set all points after stop_time equal to the point at stop_time'''

    time = int(stop_time)
    stopped_points = np.zeros_like(points)
    stopped_points[0:time] = points[0:time]
    stopped_points[time:] = points[max(time-1,0)]

    return stopped_points


def inside_region(xy):

    '''check if a point is inside the region'''

    x = xy[0]
    y = xy[1]

    if (x-1)**2 + y**2 >= 25:
        return False

    elif x**2 + y**2 <= 4:
        return False

    else:
        return True


def sim_bms(xys, timestep, total_time):

    '''simulate bms started from each point in xys, run for total_time at time-granularity of timestep'''

    # dim 0: which starting point
    # dim 1: which time step
    # dim 2: x or y

    n_starts = len(xys)
    n_steps = int(total_time / timestep)
    sd = np.sqrt(timestep)

    normal_array = np.random.normal(loc=0, scale=sd, size=(n_starts, n_steps, 2))
    normal_array[:,0,:] = xys
    bms = np.cumsum(normal_array, axis=1)

    return bms


def stop_bms(bms):

    '''stop a list of bms at their escape times'''

    xs = bms[:,:,0]
    ys = bms[:,:,1]

    check_1 = ((xs-1)**2 + ys**2 < 25)
    check_2 = (xs**2 + ys**2 > 4)
    points_inside = np.logical_and(check_1, check_2)

    escape_times = np.apply_along_axis(first_false, 1, points_inside).reshape(-1,1,1)
    reshaped_times = np.concatenate((escape_times, escape_times), axis=2)
    bms_and_times = np.concatenate((bms, reshaped_times), axis=1)

    stopped_bms_x = np.apply_along_axis(lambda x: stop_at_time(x[:-1], x[-1]), 1, bms_and_times[:,:,0])
    stopped_bms_y = np.apply_along_axis(lambda x: stop_at_time(x[:-1], x[-1]), 1, bms_and_times[:,:,1])

    stopped_bms_x = stopped_bms_x.reshape((len(xs),-1,1))
    stopped_bms_y = stopped_bms_y.reshape((len(xs),-1,1))
    stopped_bms = np.concatenate((stopped_bms_x, stopped_bms_y), 2)

    return stopped_bms


def terminal_values_stopped_bms(stopped_bms):

    '''get the terminal values of a list of stopped bms'''

    term_vals = apply_to_coords(stopped_bms[:,-1,:], phi)

    return term_vals


def make_final_surface(n_monte_carlo, timestep, fidelity, total_time):

    '''run sim bms started on a mesh, and get their expected terminal values'''

    x_scope = np.arange(-4,6, fidelity)
    y_scope = np.arange(-5,5, fidelity)
    xys = n_monte_carlo * list(product(x_scope, y_scope))
    xys = [item for item in xys if inside_region(item)]

    bms = sim_bms(xys, timestep, total_time)
    stopped_bms = stop_bms(bms)
    terminal_values = terminal_values_stopped_bms(stopped_bms)

    n_iters = len(xys)
    surface_raw = np.zeros(shape=(n_iters,3))
    surface_raw[:,0:2] = xys
    surface_raw[:,2] = terminal_values.reshape(-1)

    surface = surface_raw.reshape(n_monte_carlo,-1,3)
    surface = np.apply_along_axis(sum, 0, surface) / n_monte_carlo

    return surface





# plots
# -----------------------------------------------------------------------------------

def plot_region(n_draw_samples, dpi):

    '''plot the region'''

    # get data
    boundary_coords = get_region_boundary(n_draw_samples)

    # set up plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # region
    plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], linewidth=0, color='lightsteelblue')

    # boundary
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], color='slateblue', linewidth=0, s=5, zorder=4, alpha=1)

    # text
    ax.text(x=3.8, y=-0.3, s='$U$', fontsize=20, zorder=6)
    ax.text(x=4, y=-5, s='$\partial U$', fontsize=20, zorder=6)

    # save
    plt.axis('off')
    plt.savefig("./graphics/plot_region.png", dpi=dpi)


def plot_region_and_boundary_condition(n_draw_samples, dpi):

    '''plot the region and its boundary condition'''

    boundary_coords = get_region_boundary(n_draw_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)

    # set up plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0, zorder=0))

    # region boundary
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], color='slateblue', linewidth=0, s=5, zorder=4, alpha=1)

    # phi values
    cmap = cm.rainbow
    norm = Normalize(vmin=min(boundary_values), vmax=max(boundary_values))
    cols = cmap(norm(boundary_values[:,0]))
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], zs=boundary_values, linewidths=0, s=8, color=cols, zorder=5)

    # vertical shading
    ax.add_collection3d(data_to_polygon(boundary_coords[0:n_draw_samples,:], boundary_values[0:n_draw_samples], 0, 'k', alpha=0.1))
    ax.add_collection3d(data_to_polygon(boundary_coords[n_draw_samples:,:], boundary_values[n_draw_samples:,:], 0, 'k', alpha=0.1))

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

    plt.savefig("./graphics/plot_region_and_boundary_condition.png", dpi=dpi)

    return


def plot_few_bm_paths(n_draw_samples, timestep, total_time, dpi):

    '''plot the region, boundary conditions and some bm sample paths'''

    # get data
    boundary_coords = get_region_boundary(n_draw_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)

    # set up plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0))

    # vertical shading
    ax.add_collection3d(data_to_polygon(boundary_coords[0:n_draw_samples,:], boundary_values[0:n_draw_samples], 0, 'k', alpha = 0.1))
    ax.add_collection3d(data_to_polygon(boundary_coords[n_draw_samples:,:], boundary_values[n_draw_samples:,:], 0, 'k', alpha = 0.1))

    # region boundary
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], color='slateblue', linewidth=0, s=5, zorder=4, alpha=1)

    # phi values
    cmap = cm.rainbow
    norm = Normalize(vmin=min(boundary_values), vmax=max(boundary_values))
    cols = cmap(norm(boundary_values[:,0]))
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], zs=boundary_values, linewidths=0, s=8, color=cols, zorder=5)

    # start value
    xy = [3,0]
    ax.plot([xy[0]], [xy[1]], 'ko', zorder=8, markersize=3)

    # BMs
    np.random.seed(seed=1)
    n_paths = 2

    xys = n_paths * [xy]
    bms = sim_bms(xys, timestep, total_time)
    stopped_bms = stop_bms(bms)
    terminal_xys = stopped_bms[:,-1,:]
    terminal_values = terminal_values_stopped_bms(stopped_bms)

    for i in range(n_paths):
        col = ['red','green'][i]

        # terminal values
        ax.plot(2*[terminal_xys[i,0]], 2*[terminal_xys[i,1]], [0, terminal_values[i]], '-o', markersize=5, zorder=8, linewidth=2, color=col)

        # bm paths
        ax.plot(stopped_bms[i,:,0], stopped_bms[i,:,1], zorder=6, linewidth=0.5, color=col)

    # axis limits
    ax.set_xlim([-4,6])
    ax.set_ylim([-5,5])
    ax.set_zlim([0,40])
    plt.axis('off')

    # viewpoint
    ax.view_init(elev=60, azim=250)

    plt.savefig("./graphics/plot_few_bm_paths.png", dpi=1000)

    return


def plot_final_surface(n_monte_carlo, timestep, fidelity, total_time, n_draw_samples, dpi):

    '''plot the estimated surface'''

    np.random.seed(seed=2)

    surface = make_final_surface(n_monte_carlo, timestep, fidelity, total_time)

    boundary_coords = get_region_boundary(n_draw_samples)
    boundary_values = apply_to_coords(boundary_coords, phi)

    # set up plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # region
    ax.add_collection3d(plt.fill_between(boundary_coords[:,0], boundary_coords[:,1], 0, color='lightsteelblue', linewidth=0))

    # vertical shading
    ax.add_collection3d(data_to_polygon(boundary_coords[0:n_draw_samples,:], boundary_values[0:n_draw_samples], 0, 'k', alpha = 0.1))
    ax.add_collection3d(data_to_polygon(boundary_coords[n_draw_samples:,:], boundary_values[n_draw_samples:,:], 0, 'k', alpha = 0.1))

    # region boundary
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], color='slateblue', linewidth=0, s=5, zorder=4, alpha=1)

    # phi values
    cmap = cm.rainbow
    norm = Normalize(vmin=min(boundary_values), vmax=max(boundary_values))
    cols = cmap(norm(boundary_values[:,0]))
    ax.scatter(boundary_coords[:,0], boundary_coords[:,1], zs=boundary_values, linewidths=0, s=8, color=cols, zorder=5)

    # surface
    cmap = cm.rainbow
    norm = Normalize(vmin=min(surface[:,2]), vmax=max(surface[:,2]))
    cols = cmap(norm(surface[:,2]))
    ax.scatter(surface[:,0], surface[:,1], surface[:,2], linewidths=0, zorder=7, color=cols, s=5, alpha=1)

    # axis limits
    ax.set_xlim([-4,6])
    ax.set_ylim([-5,5])
    ax.set_zlim([0,40])
    plt.axis('off')

    # viewpoint
    ax.view_init(elev=60, azim=250)

    plt.savefig("./graphics/plot_final_surface.png", dpi=dpi)

    return
