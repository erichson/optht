import numpy as np
from scipy import linalg
from optht import optht


def test_optht():
    """Test optht using example from ``README.md``.

    This is a regression test, not a unit test! In the future, should unit test
    against the MATLAB implementation.
    """

    # Load matrices
    X_noisy = np.load('./X_noisy.npy')
    X_denoised_desired = np.load('./X_denoised.npy')

    # Compute SVD
    U, s, Vh = linalg.svd(X_noisy, full_matrices=False)

    # Determine optimal hard threshold and reconstruct image
    k = optht(X_noisy, sv=s, sigma=None)
    X_denoised = (U[:, range(k)] * s[range(k)]).dot(Vh[range(k), :])

    np.testing.assert_allclose(X_denoised, X_denoised_desired)
