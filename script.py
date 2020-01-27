import dplm

n_draw_samples = 1000
dpi = 2000
total_time = 100

print("plot region")
dplm.plot_region(n_draw_samples, dpi)

print("plot region and boundary condition")
dplm.plot_region_and_boundary_condition(n_draw_samples, dpi)

print("plot flat bm paths")
timestep = 0.001
dplm.plot_flat_bm_paths(n_draw_samples, timestep, total_time, dpi)

print("plot few bm paths")
dplm.plot_few_bm_paths(n_draw_samples, timestep, total_time, dpi)

print("plot final surface")
timestep = 0.1
for n_monte_carlo in [1,30]:
    for fidelity in [0.2,0.1]:
        print("\tMonte Carlo: {}, fidelity: {}".format(n_monte_carlo,fidelity))
        dplm.plot_final_surface(n_monte_carlo, timestep, fidelity, total_time, n_draw_samples, dpi)
