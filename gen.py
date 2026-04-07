import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('dstyle')
plt.style.use('dstyle')

def k2L(k):
    return 2. * np.pi / k

def L2k(L):
    return 2. * np.pi / L

def get_kmag(n_dims, Ng, dx_sample):
	'''
	Create a meshgrid of magnitude k values

	Args:
		n_dims (int): number of dimensions
		Ng (int): number of cells along one dimension
		dx_sample (float): sampling of grid in real space
			equivalent to 1/(2 k_Nyq)
	Returns:
		(arr): array of dimensions n_dims with k magnitudes
	'''

	k1 = np.fft.fftfreq(Ng, d=dx_sample)

	if n_dims == 1:
		kx = k1
		k2 = kx**2
	elif n_dims == 2:
		kx, ky = np.meshgrid(k1, k1, indexing='ij')
		k2 = kx**2 + ky**2
	elif n_dims == 3:
		kx, ky, kz = np.meshgrid(k1, k1, k1, indexing='ij')
		k2 = kx**2 + ky**2 + kz**2

	kmag = np.sqrt(k2)

	return kmag

def get_rmag(n_dims, Ng, dx_sample):
	'''
	Create a meshgrid of magnitude r values

	Args:
		n_dims (int): number of dimensions
		Ng (int): number of cells along one dimension
		dx (float): sampling of grid in real space by Lbox/Ng
	Returns:
		(arr): array of dimensions n_dims with k magnitudes
	'''

	# r1 = np.fft.fftfreq(Ng, d=dk_sample)
	r1 = np.arange(Ng) * dx_sample

	if n_dims == 1:
		rx = r1
		r2 = rx**2
	elif n_dims == 2:
		rx, ry = np.meshgrid(r1, r1, indexing='ij')
		r2 = rx**2 + ry**2
	elif n_dims == 3:
		rx, ry, rz = np.meshgrid(r1, r1, r1, indexing='ij')
		r2 = rx**2 + ry**2 + rz**2

	rmag = np.sqrt(r2)

	return rmag


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


def plot_info_deltak_1D(k1, delta_k):
	'''
	Plot the real-space noise for one-dimensional case

	Args:
		k1 (arr): k-mode bins along one dimension
		delta_k (arr): noise with power spectrum applied to be plotted
	Returns:
		...
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	k1_shifted = np.fft.fftshift(k1)
	_ = ax.scatter(k1_shifted, delta_k.real)
	_ = ax.scatter(k1_shifted, delta_k.imag)

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

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        noise_slab = np.zeros((Ng, Ng, n_project))
	        
	        noise_slab[ : , : , : Ng - coord_start] = noise[:, :, coord_start : Ng]
	        
	        noise_slab[ : , : , Ng - coord_start :] = noise[:, :, : coord_end  - Ng]
	    else:
	        noise_slab = noise[:,:, coord_start:coord_end]
	    
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

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	        
	    if coord_end > Ng:
	        deltak_slab = np.zeros((Ng, Ng, n_project), dtype=np.complex128)
	        
	        deltak_slab[ : , : , : Ng - coord_start] = deltak_shifted[:, :, coord_start : Ng]
	        
	        deltak_slab[ : , : , Ng - coord_start :] = deltak_shifted[:, :, : coord_end - Ng]
	    else:
	        deltak_slab = deltak_shifted[:,:, coord_start:coord_end]
	    
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

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        delta_slab = np.zeros((Ng, Ng, n_project))
	        
	        delta_slab[ : , : , : Ng - coord_start] = delta_x[:, :, coord_start : Ng]
	        
	        delta_slab[ : , : , Ng - coord_start :] = delta_x[:, :, : coord_end  - Ng]
	    else:
	        delta_slab = delta_x[:,:, coord_start:coord_end]
	    
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

	for i in range(n_plots):
	    coord_start = i * (n_project - n_window)
	    coord_end = coord_start + n_project
	    
	    if coord_end > Ng:
	        Pk_grid_slab = np.zeros((Ng, Ng, n_project))
	        
	        Pk_grid_slab[ : , : , : Ng - coord_start] = Pk_grid_shifted[:, :, coord_start : Ng]
	        
	        Pk_grid_slab[ : , : , Ng - coord_start :] = Pk_grid_shifted[:, :, : coord_end  - Ng]
	    else:
	        Pk_grid_slab = Pk_grid_shifted[:,:, coord_start:coord_end]
	        
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




def main():
	'''
	Main function
	'''

	program_intro_str = """
Program aims to generate cubic random field and apply a power spectrum that 
	scales to the power of the k-modes in the form

	P(k) = A_s (k/k_s)^n_s

	where the shape of the power spectrum is determined by the three 
	parameters

	1. A_s - power amplitude at k_s
	2. k_s - scale to apply A_s
	3. n_s - log-slope of P(k)
	"""
	print(program_intro_str)

	n_dims_str = "First, we need to know how many dimensions (i,j,k,...) to apply P(k) \n"
	n_dims_str += "What is the number of dimensions in real-space? (max 3)"
	print(n_dims_str)
	n_dims_in = input()
	n_dims = int(n_dims_in)
	assert 0 < n_dims and n_dims <= 3

	print(f"Great, we will generate random field in {n_dims:.0f} dimensions")
	print(f"Let's now define the real-space random field")

	Lbox_str = "What is the length of the cube along one dimension in units of Mpc/h"
	print(Lbox_str)
	Lbox_in = input()
	Lbox = float(Lbox_in)
	assert 0 < Lbox

	Ng_str = "What is the number of cells along one dimension?"
	print(Ng_str)
	Ng_in = input()
	Ng = int(Ng_in)
	assert 0 < Ng

	## Define global variables
	# Define fundamental and Nyquist frequency, and dx
	kfund = 2*np.pi/Lbox 
	kNyq = kfund * Ng / 2.
	dx = Lbox/Ng

	# Calculate smallest and greatest kmag probed
	kmin = kfund
	kmax = np.sqrt(n_dims) * kNyq

	space_info_str = f"""
We will create a {n_dims:.0f}-dimension cube where each side length is of size
	{Lbox} Mpc/h with {Ng} number of cells.

	delta(x) = {dx:.4e} Mpc / h - length along an edge of a cell
	kfund = {kfund:.4e} h / Mpc - fundamental k-mode
	kNyq = {kNyq:.4e} h / Mpc - Nyquist frequency, largest k-mode along a 
			dimension
	kmax = {kmax:.4e} h / Mpc - largest k magnitude probed
"""
	print(space_info_str)

	# 8 bytes in 64 bits, 1e6 bytes = 1MB, heads up if arrs will be >100MBs
	arr_MBs = (Ng**n_dims) * 8. / 1.e6
	mem_check = arr_MBs > 1.e2
	if mem_check:
		if arr_MBs < 1.e3:
			mem_arr_str = f"{arr_MBs:.4f} MBs"
		else:
			arr_GBs = arr_MBs / 1.e3
			mem_arr_str = f"{arr_GBs:.4f} GBs"

		mem_str = f"""
Hey big dawg, just a heads up that we're going to be creating a kmag and rmag
	meshgrid that will have size {mem_arr_str}. 
"""
		print(mem_str)


	dx_sample = dx / (2. * np.pi) # same as 1/(2 kNyq)
	kmag = get_kmag(n_dims, Ng, dx_sample)
	# To avoid divide by zero errors later, set the zero mode to something small
	zero_mode_indx = tuple([0] * n_dims)
	kmag[zero_mode_indx] = 1.e-10

	dk_sample = 1. / dx_sample # same as 2 kNyq
	rmag = get_rmag(n_dims, Ng, dx)


	n_kbins_str = "In defining P(k), we use 1000 k-bins linearly spaced in log-space. \n"
	n_kbins_str += "Would you like to change the number of k-bins? (y/n)"
	print(n_kbins_str)
	n_kbins_in_bool = input()
	assert (n_kbins_in_bool == 'y') or (n_kbins_in_bool == 'n')
	if n_kbins_in_bool == 'y':
		n_kbins_str = "Okay, how many kbins would you like to use?"
		n_kbins_in = input()
		n_kbins = int(n_kbins_in)
		assert 2 < n_kbins 
	else:
		n_kbins = 1000

	# Create our own power spectrum in log-space
	l_kmin = np.log10(kmin)
	l_kmax = np.log10(kmax)
	kvals = np.logspace(l_kmin, l_kmax, n_kbins)
	Pk = np.ones_like(kvals, dtype=np.float64) * 1e-16 # set a P(k) floor

	# Grab P(k) values
	As_str = "Time to define our P(k) values: \n"
	As_str += f"\t set value A_s in units of [Mpc/h]^{n_dims}"
	print(As_str)
	As_in = input()
	As = float(As_in)
	assert 0 < As

	ks_str = f"\t set value k_s in units of [h / Mpc]"
	print(ks_str)
	ks_in = input()
	ks = float(ks_in)
	assert (kmin < ks) and (ks < kmax)

	ns_str = f"\t set value n_s"
	print(ns_str)
	ns_in = input()
	ns = float(ns_in)

	# Define whether to plot or not
	plot_str = "Would you like to plot information along the way including \n"
	plot_str += "the (1) noise, (2) delta_k, (3) delta_x, (4) P(vec(k))? (y/n)"
	print(plot_str)
	plot_in_bool = input()
	assert (plot_in_bool == 'y') or (plot_in_bool == 'n')
	if plot_in_bool == 'y':
		plot_info = True
	else:
		plot_info = False



	Pk = As * (kvals / ks)**ns # [] = Length^(dims)

	if (np.abs(ns) > 32. / np.log10(kmax / kmin)):
	    mags = int(np.abs(ns) * np.log10(kmax / kmin))
	    print(f"Spanning >{mags:.0f} orders of magnitude, expect round-off error.")

	# Interpolate power spectrum onto the kmag grid
	Pk_interp = np.interp(kmag, kvals, Pk)  # [] = Length^(n_dims)


	## Apply Pk onto random field
	np.random.seed(123456) # set the random seed for reproducibility
	noise_shape = rmag.shape # same as tuple([Ng]*n_dims)

	# step 1 - Sample xi(m) with N**n_dims variance
	print(f"Sampling xi(m) with variance=N^{n_dims}...")
	variance = Ng**n_dims
	std_dev = np.sqrt(variance)
	noise = np.random.normal(size=noise_shape, scale=std_dev) # [] = Length^(-n_dims/2)
	    
	# step 2 - Evaluate xi(k) with FFT and normalize to N**(-n_dims)
	print(f"Evaluating xi(k)...")
	noise_k = np.fft.fftn(noise) / variance # [] = Length^(n_dims/2)

	# step 3 - Multiply xi(k) by Transfer Function
	print(f"Applying P(k) onto xi(k)...")
	Tk2_interp = (2. * np.pi / Lbox)**(n_dims) * Pk_interp # [] = 0
	delta_k = noise_k * np.sqrt(Pk_interp) # [] = Length^(n_dims/2)

	# step 4 - Evaluate delta(m) by taking iFFT
	print(f"Evaluate delta(m) with iFFT...")
	delta_x = np.fft.ifftn(delta_k).real # [] = Length^(-n_dims/2)


	print(f"From delta(m), calculating P(k)...")
	# Recover P(k) from delta(m)
	delta_k_calc = np.fft.fftn(delta_x) # [] = Length^(n_dims/2)
	Pk_grid_calc = np.abs(delta_k_calc)**(2) # [] = Length^(n_dims)

	print(f"Taking bins of fundamental mode...")
	# Bin into bins of fundamental mode
	nk_bins = int(np.sqrt(n_dims) * Ng)
	Pk_binned = np.zeros(nk_bins, dtype=np.float64)
	k_binned = np.zeros(nk_bins, dtype=np.float64)
	counts = np.zeros(nk_bins, dtype=np.float64)

	ikbins = (kmag / kfund).astype(np.int64) - 1

	_ = np.add.at(Pk_binned, ikbins, Pk_grid_calc)
	_ = np.add.at(k_binned, ikbins, kmag)
	_ = np.add.at(counts, ikbins, 1)

	Pk_binned_norm = Pk_binned / counts # [] = Length^(n_dims)
	k_binned_norm = k_binned / counts

	print(f"Plotting P(k)...")

	
	plot_Pk(n_dims, kvals, Pk, k_binned_norm, Pk_binned_norm, ks, As, ns, 
			kmin, kmax, Lbox)



	if plot_info:

		if n_dims == 1:
			rmag_center = rmag + 0.5 * dx
			k1 = np.fft.fftfreq(Ng, d=dx_sample)
			print(f"Plotting xi(m)...")
			plot_info_xi_1D(rmag_center, noise)
			print(f"Plotting delta(k)...")
			plot_info_deltak_1D(k1, delta_k)
			print(f"Plotting delta(x)...")
			plot_info_deltax_1D(rmag_center, delta_x)
		elif n_dims == 2:
			kx_min, kx_max = kmin, kmax
			ky_min, ky_max = kmin, kmax
			print(f"Plotting xi(m)...")
			plot_info_xi_2D(Lbox, noise)
			print(f"Plotting delta(k)...")
			plot_info_deltak_2D(kx_min, kx_max, ky_min, ky_max, delta_k)
			print(f"Plotting delta(x)...")
			plot_info_deltax_2D(Lbox, delta_x)
			print(f"Plotting P(vec(k))...")
			plot_info_Pk_2D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc)
		else:
			# 5% projection, 1% overlap
			n_project = int(Ng * 0.05)
			n_window = int(Ng * 0.01)
			kx_min, kx_max = kmin, kmax
			ky_min, ky_max = kmin, kmax
			project_str = f"We are projecting across {n_project} cells \n"
			project_str += f"\t corresponding to {n_project*dx:.4e} Mpc / h in real space"
			print(project_str)
			print(f"Plotting xi(m)...")
			plot_info_xi_3D(Lbox, noise, n_project, n_window, Ng)
			print(f"Plotting delta(k)...")
			plot_info_deltak_3D(kx_min, kx_max, ky_min, ky_max, delta_k, n_project, n_window, Ng)
			print(f"Plotting delta(x)...")
			plot_info_deltax_3D(Lbox, delta_x, n_project, n_window, Ng)
			print(f"Plotting P(vec(k))...")
			plot_info_Pk_3D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc, n_project, n_window, Ng)



if __name__=="__main__":
    main()










