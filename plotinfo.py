import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('dstyle')
plt.style.use('dstyle')

def k2L(k):
    return 2. * np.pi / k

def L2k(L):
    return 2. * np.pi / L

def plot_Pk(n_dims, kvals, Pk, k_binned_norm, Pk_binned_norm, ks, As, ns, kmin, kmax, Lbox):
	'''
	Plot the power spectrum for a 2D density fluctuation

	Args:
		n_dims (int): number of dimensions
		kvals (arr): k-mode values for injected power spectrum
		Pk (arr): defined power spectrum
		k_binned_norm (arr): binned k-mode values measured
		Pk_binned_norm (arr): binned power spectrum
		ks (float): k-mode at which power spectrum amplitude is defined
		As (float): power spectrum amplitude at ks
		ns (float): log-slope of power spectrum
		kmin (float): minimum k-mode probed
		kmax (float): maximum k-mode probed
		Lbox (float): length along one dimension
	Returns:
		...
	'''

	def Pk2Deltak(Pk):
	    return Pk * (2. * np.pi / Lbox)**n_dims

	def Deltak2Pk(Deltak):
	    return Deltak / (2. * np.pi / Lbox)**n_dims


	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

	_ = ax.plot(kvals, Pk, label='Input '+rf'$n_s={ns:.3f}$')
	_ = ax.plot(k_binned_norm[:-1], Pk_binned_norm[:-1], label='Recovered')

	_ = ax.axvline(kmin, c='k', ls='--')
	_ = ax.axvline(kmax, c='k', ls='--')

	As_base10_exponent = int(np.log10(As))
	As_base10_coeff = As / 10**(As_base10_exponent)
	Pk_ks = np.interp(ks, kvals, Pk)  # [] = Length^(n_dims)

	_ = ax.hlines(Pk_ks, color='g', ls='--', xmin = kmin, xmax = ks,
	               label=rf'$A_s = {As_base10_coeff:.2f} \times 10^{{{As_base10_exponent:.0f}}}$'+ rf'$(\rm{{Mpc / h}})^{n_dims}$')

	ks_base10_exponent = int(np.log10(ks))
	ks_base10_coeff = ks / 10**(ks_base10_exponent)
	_ = ax.vlines(ks, color='r', ls='--', ymin = 0., ymax = Pk_ks, 
	               label=rf'$k_s = {ks_base10_coeff:.2f} \times 10^{{{ks_base10_exponent:.0f}}}\ $' + r'$\rm{h / Mpc}$')

	_ = ax.set_xlabel(r"$k\ [\rm{h / Mpc}]$")
	_ = ax.set_ylabel(rf"$P(k) [(\rm{{Mpc/h}})^{n_dims}]$")

	ax2y = ax.secondary_yaxis('right', functions=(Pk2Deltak, Deltak2Pk))
	_ = ax2y.set_ylabel(rf"$\Delta^2(k) = P(k) (2 \pi / L)^{n_dims}$", rotation=270, labelpad=30)

	ax2x = ax.secondary_xaxis('top', functions=(k2L, L2k))
	_ = ax2x.set_xlabel(r"$L = 2 \pi / k\ [\rm{Mpc / h}]$", labelpad=15)

	_ = ax.set_xscale('log')
	_ = ax.set_yscale('log')

	_ = ax.legend(fontsize=20)

	_ = ax.grid(which='both', alpha=0.2)

	_ = plt.tight_layout()

	_ = plt.savefig('PowerSpectrum.png', dpi=512, bbox_inches='tight')

	_ = plt.close()



def plot_info_xi_1D(rmag_center, noise):
	'''
	Plot the real-space noise for one-dimensional case

	Args:
		rmag_center (arr): central radial magnitude valies
		noise (arr): noise to be plotted
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	_ = ax.scatter(rmag_center, noise)

	_ = ax.axhline(y=0, c='k', ls='--')

	_ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	_ = ax.set_ylabel(r"$\xi(x)$")

	_ = ax.grid(which='both', alpha=0.2)

	_ = plt.tight_layout()

	_ = plt.savefig('Noise_1D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_deltak_1D(k1, delta_k, rfft_bool):
	'''
	Plot the real-space noise for one-dimensional case

	Args:
		k1 (arr): k-mode bins along one dimension
		delta_k (arr): noise with power spectrum applied to be plotted
		rfft_bool (bool, optional): whether to use rFFT
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))


	if rfft_bool:
		k1_shifted = k1
		deltak_shifted = delta_k
	else:
		k1_shifted = np.fft.fftshift(k1)
		deltak_shifted = np.fft.fftshift(delta_k)

	_ = ax.scatter(k1_shifted, deltak_shifted.real)
	_ = ax.scatter(k1_shifted, deltak_shifted.imag)

	_ = ax.axhline(y=0, c='k', ls='--')
	_ = ax.axvline(x=0, c='k', ls='--')

	_ = ax.set_xlabel(r"$k_x\ [\rm{h / Mpc}]$")
	_ = ax.set_ylabel(r"$\tilde{\delta}(k)$")

	_ = ax.grid(which='both', alpha=0.2)

	_ = plt.tight_layout()

	_ = plt.savefig('Deltak_1D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_deltax_1D(rmag_center, delta_x):
	'''
	Plot the noise applied with a power spectrum in real-space

	Args:
		rmag_center (arr): central radial magnitude valies
		delta_x (arr): noise with power spectrum applied to be plotted in 
					real-space
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	_ = ax.scatter(rmag_center, delta_x)

	_ = ax.axhline(y=0, c='k', ls='--')

	_ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	_ = ax.set_ylabel(r"$\delta(x)$")

	_ = ax.grid(which='both', alpha=0.2)

	_ = plt.tight_layout()

	_ = plt.savefig('Deltax_1D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_xi_2D(Lbox, noise):
	'''
	Plot the real-space noise for two-dimensional case

	Args:
		Lbox (float): length along one dimension
		noise (arr): noise to be plotted
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	img = ax.imshow(noise, origin='lower', 
	                extent=(0, Lbox, 0, Lbox))

	divider = make_axes_locatable(ax)
	cbar_ax = divider.append_axes('right', size='5%', pad=0.1)
	_ = plt.colorbar(img, cax=cbar_ax)
	_ = cbar_ax.set_ylabel(r"$\xi(x,y)$", rotation=270)
	cbar_ax.yaxis.labelpad = 20

	_ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	_ = ax.set_ylabel(r"$y\ [\rm{Mpc / h}]$")

	_ = plt.tight_layout()

	_ = plt.savefig('Noise_2D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_deltak_2D(kx_min, kx_max, ky_min, ky_max, delta_k):
	'''
	Plot the real-space noise for two-dimensional case

	Args:
		kx_min (float): minimum k-mode along x-dimension
		kx_max (float): maximum k-mode along x-dimension
		ky_min (float): minimum k-mode along y-dimension
		ky_max (float): maximum k-mode along y-dimension
		delta_k (arr): noise with power spectrum applied to be plotted
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

	# again shift to easily visualize
	deltak_shifted = np.fft.fftshift(delta_k)

	ax_Re, ax_Imag = ax

	img_Re = ax_Re.imshow(deltak_shifted.real, origin='lower', 
	                      extent=(kx_min, kx_max, ky_min, ky_max))
	img_Imag = ax_Imag.imshow(deltak_shifted.imag, origin='lower', 
	                      extent=(kx_min, kx_max, ky_min, ky_max))

	divider = make_axes_locatable(ax_Re)
	cbar_ax = divider.append_axes('top', size='5%', pad=0.1)
	_ = plt.colorbar(img_Re, cax=cbar_ax, orientation='horizontal')
	_ = cbar_ax.set_xlabel(r"$Re[\tilde{\delta}(k_x, k_y)]$", labelpad=10)
	_ = cbar_ax.xaxis.set_ticks_position('top')
	_ = cbar_ax.xaxis.set_label_position('top')

	divider = make_axes_locatable(ax_Imag)
	cbar_ax = divider.append_axes('top', size='5%', pad=0.1)
	_ = plt.colorbar(img_Imag, cax=cbar_ax, orientation='horizontal')
	_ = cbar_ax.set_xlabel(r"$Im[\tilde{\delta}(k_x, k_y)]$", labelpad=10)
	_ = cbar_ax.xaxis.set_ticks_position('top')
	_ = cbar_ax.xaxis.set_label_position('top')

	_ = ax_Imag.tick_params(labelleft=False)

	_ = ax_Re.set_ylabel(r"$k_y\ [\rm{h / Mpc}]$")

	for curr_ax in ax:
	    _ = curr_ax.set_xlabel(r"$k_x\ [\rm{h / Mpc}]$")
	    	    
	_ = plt.tight_layout()

	_ = plt.savefig('Deltak_2D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_deltax_2D(Lbox, delta_x):
	'''
	Plot the noise applied with a power spectrum in real-space for 
		two-dimensional case

	Args:
		Lbox (float): length along one dimension
		delta_x (arr): noise with power spectrum applied to be plotted in 
					real-space
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	img = ax.imshow(delta_x, origin='lower', 
	                extent=(0, Lbox, 0, Lbox))

	divider = make_axes_locatable(ax)
	cbar_ax = divider.append_axes('right', size='5%', pad=0.1)
	_ = plt.colorbar(img, cax=cbar_ax)
	_ = cbar_ax.set_ylabel(r"$\delta(x,y)$", rotation=270)
	cbar_ax.yaxis.labelpad = 20

	_ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	_ = ax.set_ylabel(r"$y\ [\rm{Mpc / h}]$")

	_ = plt.tight_layout()

	_ = plt.savefig('Deltax_2D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()


def plot_info_Pk_2D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc):
	'''
	Plot the calculated power spectrum for two-dimensional case

	Args:
		kx_min (float): minimum k-mode along x-dimension
		kx_max (float): maximum k-mode along x-dimension
		ky_min (float): minimum k-mode along y-dimension
		ky_max (float): maximum k-mode along y-dimension
		Pk_grid_calc (arr): calculated power spectrum applied to be plotted
	Returns:
		...
	'''

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	# again shift to easily visualize
	Pk_grid_shifted = np.fft.fftshift(Pk_grid_calc)

	# take the log to easily visualize
	log10_Pk_grid_shifted = np.log10(Pk_grid_shifted)

	img = ax.imshow(log10_Pk_grid_shifted, origin='lower',
	               extent=(kx_min, kx_max, ky_min, ky_max))

	divider = make_axes_locatable(ax)
	cbar_ax = divider.append_axes('right', size='5%', pad=0.1)
	_ = plt.colorbar(img, cax=cbar_ax)
	_ = cbar_ax.set_ylabel(rf"$\log_{{{10}}} [ P(k) / (\rm{{Mpc/h}})^{{{2}}} ]$", rotation=270)
	cbar_ax.yaxis.labelpad = 30

	_ = ax.set_ylabel(r"$k_y\ [\rm{h / Mpc}]$")
	_ = ax.set_xlabel(r"$k_x\ [\rm{h / Mpc}]$")

	_ = plt.tight_layout()

	_ = plt.savefig('PowerSpectrum_2D.png', dpi=512, bbox_inches='tight')

	_ = plt.close()



def plot_info_xi_3D(Lbox, noise, n_project, n_window, Ng):
	'''
	Plot the real-space noise for three-dimensional case

	Args:
		Lbox (float): length along one dimension
		noise (arr): noise to be plotted
		n_project (int): number of cells to project through
		n_window (int): number of cells to overlap between projections
		Ng (int): number of cells along one dimension
	Returns:
		...
	'''
	# # 5% projection, 1% overlap
	# n_project = int(Ng * 0.05)
	# n_window = int(Ng * 0.01)

	# Count the first plot from (0,n_project)
	# The rest of the plots from will run from ((n_project-n_window),Ng) in chunks of (n_project-n_window) cells
	n_plots = 2 + ((Ng - n_project) // (n_project - n_window))

	noise_slab = np.zeros((Ng, Ng, n_project))

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        noise_slab[ : , : , : Ng - coord_start] = noise[:, :, coord_start : Ng]
	        
	        noise_slab[ : , : , Ng - coord_start :] = noise[:, :, : coord_end  - Ng]
	    else:
	        noise_slab[:, :, :] = noise[:,:, coord_start:coord_end]
	    
	    noise_projection = np.sum(noise_slab, axis=2)
	    title_str = rf"Projecting cells $n_z={coord_start}$ to $n_z={coord_end % Ng}$"

	    fig, ax = plt.subplots(figsize=(8,8), nrows=1, ncols=1)

	    img = ax.imshow(noise_projection, origin='lower', 
		                extent=(0, Lbox, 0, Lbox))

	    divider = make_axes_locatable(ax)
	    cbar_ax = divider.append_axes("right", size="5%", pad = 0.05)
	    _ = fig.colorbar(img, cax=cbar_ax)
	    _ = cbar_ax.set_ylabel(r"$\xi(x,y,z)$", rotation=270, labelpad=20)
	    
	    _ = ax.set_title(title_str, fontsize=20)
	    
	    _ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	    _ = ax.set_ylabel(r"$y\ [\rm{Mpc / h}]$")

	    _ = plt.tight_layout()
	    _ = plt.savefig(f'Noise_3D_{i:.0f}.png', dpi=512, bbox_inches='tight')
	    _ = plt.close()



def plot_info_deltak_3D(kx_min, kx_max, ky_min, ky_max, delta_k, n_project, n_window, Ng):
	'''
	Plot the real-space noise for two-dimensional case

	Args:
		kx_min (float): minimum k-mode along x-dimension
		kx_max (float): maximum k-mode along x-dimension
		ky_min (float): minimum k-mode along y-dimension
		ky_max (float): maximum k-mode along y-dimension
		delta_k (arr): noise with power spectrum applied to be plotted
		n_project (int): number of cells to project through
		n_window (int): number of cells to overlap between projections
		Ng (int): number of cells along one dimension
	Returns:
		...
	'''
	# 5% projection, 1% overlap
	# n_project = int(Ng * 0.05)
	# n_window = int(Ng * 0.01)

	# Count the first plot from (0,n_project)
	# The rest of the plots from will run from ((n_project-n_window),Ng) in chunks of (n_project-n_window) cells
	n_plots = 2 + ((Ng - n_project) // (n_project - n_window))

	# shift to easily visualize
	deltak_shifted = np.fft.fftshift(delta_k)

	deltak_slab = np.zeros((Ng, Ng, n_project), dtype=np.complex128)

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	        
	    if coord_end > Ng:
	        deltak_slab[ : , : , : Ng - coord_start] = deltak_shifted[:, :, coord_start : Ng]
	        
	        deltak_slab[ : , : , Ng - coord_start :] = deltak_shifted[:, :, : coord_end - Ng]
	    else:
	        deltak_slab[:, :, :] = deltak_shifted[:,:, coord_start:coord_end]
	    
	    deltak_projection = np.sum(deltak_slab, axis=2)
	    title_str = rf"Projecting cells $n_{{k,z}}={coord_start}$ to $n_{{k,z}}={coord_end % Ng}$"
	    
	    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

	    ax_Re, ax_Imag = ax

	    img_Re = ax_Re.imshow(deltak_projection.real, origin='lower', 
	                          extent=(kx_min, kx_max, ky_min, ky_max))
	    img_Imag = ax_Imag.imshow(deltak_projection.imag, origin='lower', 
	                          extent=(kx_min, kx_max, ky_min, ky_max))

	    divider = make_axes_locatable(ax_Re)
	    cbar_ax = divider.append_axes('top', size='5%', pad=0.1)
	    _ = plt.colorbar(img_Re, cax=cbar_ax, orientation='horizontal')
	    _ = cbar_ax.set_xlabel(r"$Re[\tilde{\delta}(k_x, k_y)]$", labelpad=10)
	    _ = cbar_ax.xaxis.set_ticks_position('top')
	    _ = cbar_ax.xaxis.set_label_position('top')

	    divider = make_axes_locatable(ax_Imag)
	    cbar_ax = divider.append_axes('top', size='5%', pad=0.1)
	    _ = plt.colorbar(img_Imag, cax=cbar_ax, orientation='horizontal')
	    _ = cbar_ax.set_xlabel(r"$Im[\tilde{\delta}(k_x, k_y)]$", labelpad=10)
	    _ = cbar_ax.xaxis.set_ticks_position('top')
	    _ = cbar_ax.xaxis.set_label_position('top')

	    _ = ax_Imag.tick_params(labelleft=False)

	    _ = ax_Re.set_ylabel(r"$k_y\ [\rm{h / Mpc}]$")
	    
	    _ = fig.suptitle(title_str, y=0.93)
	    

	    for curr_ax in ax:
	        _ = curr_ax.set_xlabel(r"$k_x\ [\rm{h / Mpc}]$")

	    _ = plt.tight_layout()

	    _ = plt.savefig(f'Deltak_3D_{i:.0f}.png', dpi=512, bbox_inches='tight')

	    _ = plt.close()


def plot_info_deltax_3D(Lbox, delta_x, n_project, n_window, Ng):
	'''
	Plot the noise applied with a power spectrum in real-space for 
		three-dimensional case

	Args:
		Lbox (float): length along one dimension
		delta_x (arr): noise with power spectrum applied to be plotted in 
					real-space
		n_project (int): number of cells to project through
		n_window (int): number of cells to overlap between projections
		Ng (int): number of cells along one dimension
	Returns:
		...
	'''

	# # 5% projection, 1% overlap
	# n_project = int(Ng * 0.05)
	# n_window = int(Ng * 0.01)

	# Count the first plot from (0,n_project)
	# The rest of the plots from will run from ((n_project-n_window),Ng) in chunks of (n_project-n_window) cells
	n_plots = 2 + ((Ng - n_project) // (n_project - n_window))

	delta_slab = np.zeros((Ng, Ng, n_project))

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        delta_slab[ : , : , : Ng - coord_start] = delta_x[:, :, coord_start : Ng]
	        
	        delta_slab[ : , : , Ng - coord_start :] = delta_x[:, :, : coord_end  - Ng]
	    else:
	        delta_slab[ :, :, :] = delta_x[:,:, coord_start:coord_end]
	    
	    delta_projection = np.sum(delta_slab, axis=2)
	    title_str = rf"Projecting cells $n_z={coord_start}$ to $n_z={coord_end % Ng}$"

	    fig, ax = plt.subplots(figsize=(8,8), nrows=1, ncols=1)

	    img = ax.imshow(delta_projection, origin='lower', extent=(0, Lbox, 0, Lbox))

	    divider = make_axes_locatable(ax)
	    cbar_ax = divider.append_axes("right", size="5%", pad = 0.05)
	    _ = fig.colorbar(img, cax=cbar_ax)
	    _ = cbar_ax.set_ylabel(r"$\delta(x,y,z)$", rotation=270, labelpad=20)
	    
	    _ = ax.set_title(title_str, fontsize=20)
	    
	    _ = ax.set_xlabel(r"$x\ [\rm{Mpc / h}]$")
	    _ = ax.set_ylabel(r"$y\ [\rm{Mpc / h}]$")

	    _ = plt.tight_layout()

	    _ = plt.savefig(f'Deltax_3D_{i:.0f}.png', dpi=512, bbox_inches='tight')

	    _ = plt.close()



def plot_info_Pk_3D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc, n_project, n_window, Ng):
	'''
	Plot the calculated power spectrum for two-dimensional case

	Args:
		kx_min (float): minimum k-mode along x-dimension
		kx_max (float): maximum k-mode along x-dimension
		ky_min (float): minimum k-mode along y-dimension
		ky_max (float): maximum k-mode along y-dimension
		Pk_grid_calc (arr): calculated power spectrum applied to be plotted
		n_project (int): number of cells to project through
		n_window (int): number of cells to overlap between projections
		Ng (int): number of cells along one dimension
	Returns:
		...
	'''

	# # 5% projection, 1% overlap
	# n_project = int(Ng * 0.05)
	# n_window = int(Ng * 0.01)

	# Count the first plot from (0,n_project)
	# The rest of the plots from will run from ((n_project-n_window),Ng) in chunks of (n_project-n_window) cells
	n_plots = 2 + ((Ng - n_project) // (n_project - n_window))

	# again shift to easily visualize
	Pk_grid_shifted = np.fft.fftshift(Pk_grid_calc)

	Pk_grid_slab = np.zeros((Ng, Ng, n_project))

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        Pk_grid_slab[ : , : , : Ng - coord_start] = Pk_grid_shifted[:, :, coord_start : Ng]
	        
	        Pk_grid_slab[ : , : , Ng - coord_start :] = Pk_grid_shifted[:, :, : coord_end  - Ng]
	    else:
	        Pk_grid_slab[:, :, :] = Pk_grid_shifted[:,:, coord_start:coord_end]
	        
	    Pk_grid_projection = np.sum(Pk_grid_slab, axis=2)
	    
	    # take the log to easily visualize
	    log10_Pk_grid_projection = np.log10(Pk_grid_projection)
	    
	    title_str = rf"Projecting cells $n_{{k,z}}={coord_start}$ to $n_{{k,z}}={coord_end % Ng}$"

	    fig, ax = plt.subplots(figsize=(8,8), nrows=1, ncols=1)

	    img = ax.imshow(log10_Pk_grid_projection, origin='lower')

	    divider = make_axes_locatable(ax)
	    cbar_ax = divider.append_axes("right", size="5%", pad = 0.05)
	    _ = fig.colorbar(img, cax=cbar_ax)
	    _ = cbar_ax.set_ylabel(rf"$\log_{{{10}}} [ P(k) / (\rm{{Mpc/h}})^{2} ]$", rotation=270, labelpad=30)
	    
	    _ = ax.set_title(title_str, fontsize=20)
	    
	    _ = ax.set_xlabel(r"$k_x\ [\rm{h / Mpc}]$")
	    _ = ax.set_ylabel(r"$k_y\ [\rm{h / Mpc}]$")

	    _ = plt.tight_layout()

	    _ = plt.savefig(f'PowerSpectrum_3D_{i:.0f}.png', dpi=512, bbox_inches='tight')

	    _ = plt.close()
