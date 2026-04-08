import numpy as np
from plotinfo import *


def get_kmag(n_dims, Ng, dx_sample, rfft_bool=False):
	'''
	Create a meshgrid of magnitude k values

	Args:
		n_dims (int): number of dimensions
		Ng (int): number of cells along one dimension
		dx_sample (float): sampling of grid in real space
			equivalent to 1/(2 k_Nyq)
		rfft_bool (bool, optional): whether to take rFFT
	Returns:
		(arr): array of dimensions n_dims with k magnitudes
	'''

	k1 = np.fft.fftfreq(Ng, d=dx_sample)
	if rfft_bool:
		k1_r = np.fft.rfftfreq(Ng, d=dx_sample)

	if n_dims == 1:
		if rfft_bool:
			kx = k1_r
		else:
			kx = k1
		k2 = kx**2
	elif n_dims == 2:
		if rfft_bool:
			kx, ky = np.meshgrid(k1, k1_r, indexing='ij')
		else:
			kx, ky = np.meshgrid(k1, k1, indexing='ij')
		k2 = kx**2 + ky**2
	elif n_dims == 3:
		if rfft_bool:
			kx, ky, kz = np.meshgrid(k1, k1, k1_r, indexing='ij')
		else:
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

	rfft_str = "Since our starting random field is Real, we could take the rFFT \n"
	rfft_str += "\t Instead of FFT to ease the memory constraints during calculations. \n"
	rfft_str += "\t Would you like to use rFFT? (y/n)"
	print(rfft_str)
	rfft_in_bool = input()
	assert (rfft_in_bool == 'y') or (rfft_in_bool == 'n')
	if rfft_in_bool == 'y':
		rfft_bool = True
	else:
		rfft_bool = False


	dx_sample = dx / (2. * np.pi) # same as 1/(2 kNyq)
	kmag = get_kmag(n_dims, Ng, dx_sample, rfft_bool)
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
		print(n_kbins_str)
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

	# Define whether to plot miscellanious info or not
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
	Tk2_interp = (2. * np.pi / Lbox)**(n_dims) * Pk_interp # [] = 0

	## Apply Pk onto random field
	np.random.seed(123456) # set the random seed for reproducibility
	noise_shape = rmag.shape # same as tuple([Ng]*n_dims)

	# step 1 - Sample xi(m) with N**n_dims variance
	print(f"Sampling xi(m) with variance=N^{n_dims}...")
	variance = Ng**n_dims
	std_dev = np.sqrt(variance)
	noise = np.random.normal(size=noise_shape, scale=std_dev) # [] = 0 != Length^(-n_dims/2)
	    
	# step 2 - Evaluate xi(k) with FFT and normalize to N**(-n_dims)
	print(f"Evaluating xi(k)...")
	if rfft_bool:
		noise_k = np.fft.rfftn(noise) / variance # [] = 0 != Length^(n_dims/2)
	else:
		noise_k = np.fft.fftn(noise) / variance # [] = 0 != Length^(n_dims/2)

	# step 3 - Multiply xi(k) by Transfer Function
	print(f"Applying P(k) onto xi(k)...")
	delta_k = noise_k * np.sqrt(Tk2_interp) # [] = 0 != Length^(n_dims/2)

	# step 4 - Evaluate delta(m) by taking iFFT
	print(f"Evaluating delta(m) with iFFT...")
	if rfft_bool:
		delta_x = np.fft.irfftn(delta_k).real # [] = 0 != Length^(-n_dims/2)
	else:
		delta_x = np.fft.ifftn(delta_k).real # [] = 0 != Length^(-n_dims/2)

	print(f"From delta(m), calculating P(k)...")
	# Recover P(k) from delta(m)
	if rfft_bool:
		delta_k_calc = np.fft.rfftn(delta_x) # [] = 0 != Length^(n_dims/2)
	else:
		delta_k_calc = np.fft.fftn(delta_x) # [] = 0 != Length^(n_dims/2)
	Tk2_grid_calc = np.abs(delta_k_calc)**(2) # [] = 0
	Pk_grid_calc = Tk2_grid_calc / (2. * np.pi / Lbox)**(n_dims) # [] = Length^(n_dims)

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
			if rfft_bool:
				k1 = np.fft.rfftfreq(Ng, d=dx_sample)
			else:
				k1 = np.fft.fftfreq(Ng, d=dx_sample)
			print(f"Plotting xi(m)...")
			plot_info_xi_1D(rmag_center, noise)
			print(f"Plotting delta(k)...")
			plot_info_deltak_1D(k1, delta_k, rfft_bool)
			print(f"Plotting delta(x)...")
			plot_info_deltax_1D(rmag_center, delta_x)
		elif n_dims == 2:
			kx_min, kx_max = -1. * kNyq, kNyq - kfund
			if rfft_bool:
				ky_min, ky_max = 0., kNyq
			else:
				ky_min, ky_max = -1. * kNyq, kNyq - kfund
			print(f"Plotting xi(m)...")
			plot_info_xi_2D(Lbox, noise)
			print(f"Plotting delta(k)...")
			plot_info_deltak_2D(kx_min, kx_max, ky_min, ky_max, delta_k, rfft_bool)
			print(f"Plotting delta(x)...")
			plot_info_deltax_2D(Lbox, delta_x)
			print(f"Plotting P(vec(k))...")
			plot_info_Pk_2D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc, rfft_bool)
		else:
			if rfft_bool:
				Ng_kz = (Ng // 2) + 1
			else:
				Ng_kz = Ng
			# 5% projection, 1% overlap
			n_project = int(Ng * 0.05)
			n_window = int(Ng * 0.01)

			nk_project = int(Ng_kz * 0.05)
			nk_window = int(Ng_kz * 0.01)

			kx_min, kx_max = kmin, kmax
			ky_min, ky_max = kmin, kmax
			project_str = f"We are projecting across 5% of {Ng} or {n_project} cells \n"
			project_str += f"\t corresponding to {n_project*dx:.4e} Mpc / h in real space"
			print(project_str)
			print(f"Plotting xi(m)...")
			plot_info_xi_3D(Lbox, noise, n_project, n_window, Ng)
			print(f"Plotting delta(k)...")
			plot_info_deltak_3D(kx_min, kx_max, ky_min, ky_max, delta_k, 
								nk_project, nk_window, Ng, rfft_bool)
			print(f"Plotting delta(x)...")
			plot_info_deltax_3D(Lbox, delta_x, n_project, n_window, Ng)
			print(f"Plotting P(vec(k))...")
			plot_info_Pk_3D(kx_min, kx_max, ky_min, ky_max, Pk_grid_calc, 
								nk_project, nk_window, Ng, rfft_bool)

	print(f"Complete !")

if __name__=="__main__":
    main()
