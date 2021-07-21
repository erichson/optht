"""Optimal hard threshold example from README."""

from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg
from optht import optht
import logging

logging.basicConfig(level=logging.INFO)

# Create some data
t = np.arange(-2, 2, 0.01)
Utrue = np.array(([np.cos(17 * t) * np.exp(-t**2), np.sin(11 * t)])).T
Strue = np.array(([2, 0], [0, .5]))
Vtrue = np.array(([np.sin(5 * t) * np.exp(-t**2), np.cos(13 * t)])).T

# Construct image
X = Utrue.dot(Strue).dot(Vtrue.T)

# Define the noise level and add
sigma = 0.5
X_noisy = X + sigma * np.random.standard_normal(X.shape)

# Compute SVD
U, s, Vh = linalg.svd(X_noisy, full_matrices=False)

# Determine optimal hard threshold and reconstruct image
k = optht(X_noisy, sv=s, sigma=None)
X_denoised = (U[:, range(k)] * s[range(k)]).dot(Vh[range(k), :])

# Plot the results
plt.subplot(131)
plt.imshow(X, cmap='gray', interpolation='bicubic')
plt.title('Original image')
plt.axis('off')

plt.subplot(132)
plt.imshow(X_noisy, cmap='gray', interpolation='bicubic')
plt.title('Noisy image, sigma=%s' % sigma)
plt.axis('off')

plt.subplot(133)
plt.imshow(X_denoised, cmap='gray', interpolation='bicubic')
rmseSVD = np.sqrt(np.sum((X - X_denoised)**2) / np.sum(X**2))
plt.title('Denoised image,  nrmse=%s ' % np.round(rmseSVD, 2))
plt.axis('off')
plt.show()

# Plot the singular value spectrum
plt.plot((np.arange(1, s.shape[0] + 1)),
         np.log(s),
         c='b',
         marker='o',
         linestyle='--')
plt.xlabel('k', fontsize=25)
plt.ylabel('Log-scaled singular values')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.title('Singular value spectrum')
plt.axvline(k, c='r', linewidth=2, linestyle='--')
plt.show()
