# GenRandomFieldwPk-py
Generate a Gaussian random field and apply power spectrum in Python

We follow [Bertschinger, E (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..137....1B/abstract) where we convolve a Gaussian random field with a given power spectrum:

1. Generate random field in real-space
2. Take FFT of random field 
3. Multiply random field in frequency-space by power spectrum
4. Take iFFT to produce a displacement field

We apply a logarithmic power spectrum of the form

$$
P(|k|) = A_s \left( \frac{|k|}{k_s} \right)^{n_s}
$$

The script will take a couple parameters:

1. Number of dimensions of random field
2. Length of cube along one dimension in units of Mpc/h
3. Number of cells along one dimension
4. Power spectrum values: $A_s, k_s, n_s$

Script can also plot the initial random field, random field in frequency-space multiplied by power spectrum, and N-dimensional power spectrum.

Script also allows to use real FFT instead of full FFT since initial random field is completely real.

Only required packages are `numpy` for FFT calculations and array manipulation, and `matplotlib` for plotting.

To run the script, simply run

```
python3 gen.py
```
