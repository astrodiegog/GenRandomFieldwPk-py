import numpy as np

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

def get_rmag(n_dims, Ng, dk_sample):
	'''
	Create a meshgrid of magnitude r values

	Args:
		n_dims (int): number of dimensions
		Ng (int): number of cells along one dimension
		dk_sample (float): sampling of grid in frequency space
			equivalent to (2 k_Nyq)
	Returns:
		(arr): array of dimensions n_dims with k magnitudes
	'''

	r1 = np.fft.fftfreq(Ng, d=dk_sample)
	
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

	Ng_str = "What is the number of cells along one dimension?"
	print(Ng_str)
	Ng_in = input()
	Ng = int(Ng_in)

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
	rmag = get_rmag(n_dims, Ng, dk_sample)

	


if __name__=="__main__":
    main()










