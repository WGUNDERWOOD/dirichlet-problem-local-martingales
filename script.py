import dplm

n_draw_samples = 1000
dpi = 1000
total_time = 100

print("plot region")
#dplm.plot_region(n_draw_samples, dpi)

print("plot region and boundary condition")
#dplm.plot_region_and_boundary_condition(n_draw_samples, dpi)

print("plot few bm paths")
timestep = 0.001
#dplm.plot_few_bm_paths(n_draw_samples, timestep, total_time, dpi)

print("plot final surface")
timestep = 0.1
for n_monte_carlo in [1,50]:
    for fidelity in [0.2,0.05]:
        print("Monte Carlo: {}, fidelity: {}".format(n_monte_carlo,fidelity))
        dplm.plot_final_surface(n_monte_carlo, timestep, fidelity, total_time, n_draw_samples, dpi)
