import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
	
	
def get_region_boundary(num_samples):

	offset_circles_coords = get_offset_circles((1,0),(0,0),5,2,num_samples)
	
	return offset_circles_coords
   

def apply_to_coords(coords, func):

    zs = np.apply_along_axis(func, 1, coords).reshape(-1,1)

    return zs
	
	
def phi(xy):
    
	x = xy[0]
	y = xy[1]
    
	temp = x**2 + y + 10
	
	return temp
	


def data_to_polygon(dU, boundary_values, ref_height, color, alpha):

    xs = dU[:,0]
    ys = dU[:,1]
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
	
	
def plot_region_and_boundary_condition(dU, boundary_values, num_samples):
	
	# set up plot
	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111, projection='3d')
	
	# region
	ax.add_collection3d(plt.fill_between(dU[:,0], dU[:,1], 0, color='lightsteelblue', linewidth=0))
	
	# region boundary
	ax.plot(xs=dU[0:num_samples,0], ys=dU[0:num_samples,1], zs=0, color='slateblue', zorder=10, linewidth=2)
	ax.plot(xs=dU[num_samples:,0], ys=dU[num_samples:,1], zs=0, color='slateblue', zorder=10, linewidth=2)
	
	# vertical shading
	ax.add_collection3d(data_to_polygon(dU[0:num_samples,:], boundary_values[0:num_samples], 0, 'r', 0.5))
	ax.add_collection3d(data_to_polygon(dU[num_samples:,:], boundary_values[num_samples:,:], 0, 'r', 0.5))
	
	# phi values
	ax.plot(xs=dU[0:num_samples,0], ys=dU[0:num_samples,1], zs=boundary_values[0:num_samples,0], color='r', zorder=10, linewidth=2)
	ax.plot(xs=dU[num_samples:,0], ys=dU[num_samples:,1], zs=boundary_values[num_samples:,0], color='r', zorder=10, linewidth=2)
	
	# axis limits
	ax.set_xlim([-4,6])
	ax.set_ylim([-5,5])
	ax.set_zlim([0,40])
	plt.axis('off')
	
	# viewpoint
	ax.view_init(elev=60, azim=250)

	plt.show()
	
	return
	
	
	

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
