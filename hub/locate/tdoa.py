"""GCC-PHAT time-difference-of-arrival (Knapp & Carter 1976)."""
from __future__ import annotations

import numpy as np


def gcc_phat(x_i, x_j, fs, max_tau=None, interp=8):
    """Return (lags_s, cc). Peak at lag tau means x_i(t) ~ x_j(t - tau), i.e.
    mic i hears the wavefront tau seconds LATER than mic j (tau = t_i - t_j).

    R(tau) = IFFT{ X_i(f) conj(X_j(f)) / |X_i(f) conj(X_j(f))| }  (PHAT weight).
    """
    n = len(x_i) + len(x_j)
    nfft = 1 << (n - 1).bit_length()
    Xi = np.fft.rfft(x_i, nfft)
    Xj = np.fft.rfft(x_j, nfft)
    G = Xi * np.conj(Xj)
    G /= np.maximum(np.abs(G), 1e-12)
    cc = np.fft.irfft(G, nfft * interp)
    nmax = nfft * interp // 2 - 1
    if max_tau is not None:
        nmax = min(nmax, int(max_tau * fs * interp) + 1)
    cc = np.concatenate([cc[-nmax:], cc[: nmax + 1]])
    lags = np.arange(-nmax, nmax + 1) / (fs * interp)
    return lags, cc
