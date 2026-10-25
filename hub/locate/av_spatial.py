"""Camera-ray to microphone-pair delay geometry and spatial evidence (spec Section 5.5).

Conventions
  * Array frame: +x right, +y up, +z forward. Bearing theta is measured from +z toward +x.
  * Camera frame for `ray_from_pixel`: pinhole intrinsics on UNDISTORTED pixels (image y grows
    down); the pixel ray is first expressed with x right / y up / z forward and then rotated into
    the array frame by `R_cam_to_array`.
  * Rays point TOWARD the source.
  * Delay sign matches `locate.tdoa.gcc_phat` and `locate.pair.frame_delays`: positive means
    channel 0 (position p1) hears the wavefront LATER than channel 1 (position p2). A source
    along b = p2 - p1 therefore gives a positive delay.

The existing `locate.pair` model is an empirical fit, `D*sin(theta - axis_deg - yaw) + c`.
`axis_deg` is that sine's phase, so its equivalent baseline azimuth is `axis_deg + 90 deg`
(`b_hat = (cos phi, 0, -sin phi)`). `delay_amp_s` is already an effective amplitude: it must not
be multiplied by `baffle_scale` again. `baffle_scale` only applies to the physical-geometry model.
Nothing here changes `pair.py` or `tdoa.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .tdoa import gcc_phat  # noqa: F401  (re-exported: the reference sign convention)


# ------------------------------------------------------------------ physics
def c_sound(temp_c: float = 20.0) -> float:
    """Dry-air speed of sound, c ~ 331.3 + 0.606*T_C (m/s)."""
    return 331.3 + 0.606 * float(temp_c)


def ray_from_pixel(u: float, v: float, fx: float, fy: float, cx: float, cy: float,
                   r_cam_to_array: np.ndarray | None = None) -> np.ndarray:
    """Unit ray (array frame) toward the scene point seen at undistorted pixel (u, v)."""
    cam = np.array([(u - cx) / fx, -(v - cy) / fy, 1.0])   # y flipped: image-down -> up
    cam /= np.linalg.norm(cam)
    R = np.eye(3) if r_cam_to_array is None else np.asarray(r_cam_to_array, float)
    r = R @ cam
    return r / np.linalg.norm(r)


def direction_from_az_el(az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    """Unit vector for bearing az (from +z toward +x) and elevation el (toward +y)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def tau_geom(r_hat, p1, p2, temp_c: float = 20.0, baffle_scale: float = 1.0) -> np.ndarray:
    """Far-field delay (s) of the pair for rays toward the source: d*dot(r_hat, b_hat)/c.

    `baffle_scale` (>= 1) is an effective-baseline factor for the geometric model only."""
    b = np.asarray(p2, float) - np.asarray(p1, float)
    return baffle_scale * (np.asarray(r_hat, float) @ b) / c_sound(temp_c)


def tau_exact(s, p1, p2, temp_c: float = 20.0) -> float:
    """Exact near-field delay for a source POSITION s: (|s-p1| - |s-p2|)/c."""
    s, p1, p2 = (np.asarray(v, float) for v in (s, p1, p2))
    return float((np.linalg.norm(s - p1) - np.linalg.norm(s - p2)) / c_sound(temp_c))


def source_position(r_hat, range_m: float, r_array_from_cam=None, t_cam_to_array=None) -> np.ndarray:
    """Known-depth source position in the array frame: s = R*(r*r_hat_cam) + t."""
    s = range_m * np.asarray(r_hat, float)
    if r_array_from_cam is not None:
        s = np.asarray(r_array_from_cam, float) @ s
    if t_cam_to_array is not None:
        s = s + np.asarray(t_cam_to_array, float)
    return s


# ---- adapters for the fitted PairCalibration (never double-scaled)
def baseline_from_pair_axis(axis_deg: float) -> np.ndarray:
    """Equivalent horizontal baseline direction for the sine phase `axis_deg`."""
    phi = np.radians(axis_deg)
    return np.array([np.cos(phi), 0.0, -np.sin(phi)])


def fitted_pair_tau(delay_amp_s: float, axis_deg: float, offset_s: float, r_hat) -> np.ndarray:
    """Delay predicted by a fitted pair for rays r_hat: delay_amp * dot(r_hat, b_hat) + offset.

    Equals `PairCalibration.predict(theta, yaw=0)` for horizontal rays. `delay_amp_s` is used as
    fitted: baffle effects are already inside it."""
    return delay_amp_s * (np.asarray(r_hat, float) @ baseline_from_pair_axis(axis_deg)) + offset_s


def naive_bearing_from_tau(tau_s: float, d_m: float, temp_c: float = 20.0) -> float:
    """Horizontal-only inversion asin(c*tau/d) (deg). Kept to quantify the elevation error."""
    return float(np.degrees(np.arcsin(np.clip(c_sound(temp_c) * tau_s / d_m, -1, 1))))


def naive_bearing_from_ray(r_hat) -> float:
    """Horizontal camera bearing atan2(ray_x, ray_z) (deg), which discards elevation."""
    r = np.asarray(r_hat, float)
    return float(np.degrees(np.arctan2(r[0], r[2])))


# ------------------------------------------------------------------ uncertainty
@dataclass
class Parallax:
    sigma_rad: float
    confident: bool    # False: far-field match must be withheld (source may be closer than the offset)


def parallax_sigma(t_cam_to_array_m, r_min_m: float = 0.5) -> Parallax:
    """Angular uncertainty floor from the camera-to-array translation at minimum range.

    atan(|t|/r_min) is the lateral-offset floor; when r_min > |t| the stronger asin(|t|/r_min)
    bound is used. Otherwise the geometry cannot support a confident far-field match."""
    t = float(np.linalg.norm(np.asarray(t_cam_to_array_m, float)))
    if t == 0.0:
        return Parallax(0.0, True)
    if r_min_m > t:
        return Parallax(float(np.arcsin(t / r_min_m)), True)
    return Parallax(float(np.pi / 2), False)


def angular_variance(base_sigma_rad: float, parallax: Parallax) -> float:
    return float(base_sigma_rad ** 2 + parallax.sigma_rad ** 2)


def fraunhofer_range(aperture_m: float, freq_hz: float, temp_c: float = 20.0) -> float:
    """r_FF = 2*D^2/lambda: below this range the plane-wave approximation degrades."""
    lam = c_sound(temp_c) / freq_hz
    return 2.0 * aperture_m ** 2 / lam


def curvature_delay_sigma(aperture_m: float, range_m: float, freq_hz: float, temp_c: float = 20.0) -> float:
    """Extra delay uncertainty (s) when the source is inside the Fraunhofer screen (else 0).

    Uses the wavefront sagitta D^2/(8 r) over the aperture."""
    if range_m >= fraunhofer_range(aperture_m, freq_hz, temp_c):
        return 0.0
    return float(aperture_m ** 2 / (8.0 * range_m) / c_sound(temp_c))


# ------------------------------------------------------------------ aliasing
def pair_alias_freq(d_m: float, temp_c: float = 20.0) -> float:
    """f_alias,ij = c / (2 d_ij)."""
    return c_sound(temp_c) / (2.0 * d_m)


def enumerate_delay_aliases(tau0_s: float, freq_hz: float, d_m: float, temp_c: float = 20.0) -> list[float]:
    """Delays tau0 + k/f that are physically admissible (|tau| <= d/c) for a pure tone."""
    limit = d_m / c_sound(temp_c)
    period = 1.0 / freq_hz
    k_min = int(np.floor((-limit - tau0_s) / period)) - 1
    k_max = int(np.ceil((limit - tau0_s) / period)) + 1
    return [tau0_s + k * period for k in range(k_min, k_max + 1) if abs(tau0_s + k * period) <= limit + 1e-12]


# ------------------------------------------------------------------ spectra and weighted GCC
def stft_frames(x: np.ndarray, nfft: int = 1024, hop: int = 512) -> np.ndarray:
    """(frames, bins) complex STFT with a Hann window and no padding."""
    _, _, z = signal.stft(x, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False, window="hann")
    return z.T


def wdo_weights(stem_mags: list[np.ndarray], min_weight: float = 0.1, eps: float = 1e-9) -> list[np.ndarray]:
    """W-disjoint-orthogonality dominance W_k = |S_k| / (sum_j |S_j| + eps) on a shared grid.

    Bins whose dominance is below `min_weight` (wdo_min_weight) are set to 0 so they supply no
    spatial evidence. Weights stay SOFT otherwise."""
    mags = [np.abs(m) for m in stem_mags]
    total = np.sum(mags, axis=0) + eps
    out = []
    for m in mags:
        w = m / total
        out.append(np.where(w >= min_weight, w, 0.0))
    return out


def weighted_gcc_phat(Xi: np.ndarray, Xj: np.ndarray, fs: int, nfft: int, max_tau_s: float, interp: int = 16,
                      W: np.ndarray | None = None, min_power_db: float = -40.0):
    """GCC-PHAT from STFT frames with optional WDO weights applied AFTER PHAT normalisation.

    Xi, Xj: (frames, bins) from `stft_frames`. Bins whose cross-spectrum magnitude is more than
    |min_power_db| below the strongest are excluded: PHAT would otherwise lift numerical dust to
    unit magnitude. Applying one mask to both channels BEFORE PHAT multiplies the cross-spectrum
    by W^2, which PHAT divides straight back out; the dominance would be lost.

    Returns (lags_s, cc) with the same convention as `tdoa.gcc_phat`: peak at +tau means channel
    i hears the wavefront tau later than channel j."""
    G = Xi * np.conj(Xj)
    mag = np.abs(G)
    gate = mag >= mag.max() * 10 ** (min_power_db / 20.0)
    P = np.where(gate, G / np.maximum(mag, 1e-30), 0.0)
    if W is not None:
        P = P * W
    acc = P.sum(axis=0)
    cc = np.fft.irfft(acc, nfft * interp)
    nmax = min(nfft * interp // 2 - 1, int(max_tau_s * fs * interp) + 1)
    cc = np.concatenate([cc[-nmax:], cc[: nmax + 1]])
    lags = np.arange(-nmax, nmax + 1) / (fs * interp)
    return lags, cc


def retained_peaks(lags: np.ndarray, cc: np.ndarray, rel_height: float = 0.5, max_tau_s: float | None = None):
    """Local maxima of the GCC curve within +-max_tau_s above `rel_height` of the global peak."""
    sel = np.ones(len(lags), bool) if max_tau_s is None else np.abs(lags) <= max_tau_s
    c, l = cc[sel], lags[sel]
    idx, _ = signal.find_peaks(c, height=rel_height * c.max())
    return [(float(l[i]), float(c[i])) for i in idx]


@dataclass
class PairAnalysis:
    pair: tuple[int, int]
    d_m: float
    f_alias_hz: float
    dominant_hz: float
    bandwidth_hz: float
    narrowband: bool
    peaks: list[tuple[float, float]] = field(default_factory=list)
    admissible_aliases_s: list[float] = field(default_factory=list)
    aliasing_multimodal_flag: bool = False

    @property
    def best_tau_s(self) -> float | None:
        """Strongest delay, or None when aliasing makes the peaks a tie (never force an argmax)."""
        if self.aliasing_multimodal_flag or not self.peaks:
            return None
        return max(self.peaks, key=lambda p: p[1])[0]


def spectral_centroid_bandwidth(x: np.ndarray, fs: int, floor_db: float = -40.0) -> tuple[float, float]:
    """Power-weighted centroid and occupied bandwidth (4 sigma) of the active bins (Hz)."""
    X = np.abs(np.fft.rfft(np.asarray(x, float) * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    keep = X >= X.max() * 10 ** (floor_db / 10.0)
    p, ff = X[keep], f[keep]
    fc = float((p * ff).sum() / p.sum())
    sigma = float(np.sqrt((p * (ff - fc) ** 2).sum() / p.sum()))
    return fc, 4.0 * sigma


def analyze_pair(xi: np.ndarray, xj: np.ndarray, fs: int, d_m: float, pair: tuple[int, int] = (0, 1),
                 temp_c: float = 20.0, nfft: int = 1024, hop: int = 512, interp: int = 16,
                 narrowband_ratio: float = 1.0, peak_rel_height: float = 0.5,
                 W: np.ndarray | None = None) -> PairAnalysis:
    """Retain multimodal GCC peaks and decide the per-pair aliasing flag.

    The flag is evaluated against THIS pair's own cutoff c/(2 d_ij): a longer baseline flags at a
    lower frequency than a shorter one. Narrowband means occupied bandwidth < narrowband_ratio *
    centre frequency; broadband content resolves the ambiguity incoherently and is not flagged."""
    fc, bw = spectral_centroid_bandwidth(0.5 * (np.asarray(xi, float) + np.asarray(xj, float)), fs)
    f_alias = pair_alias_freq(d_m, temp_c)
    max_tau = d_m / c_sound(temp_c)
    lags, cc = weighted_gcc_phat(stft_frames(xi, nfft, hop), stft_frames(xj, nfft, hop), fs, nfft, max_tau, interp, W)
    peaks = retained_peaks(lags, cc, peak_rel_height, max_tau)
    narrow = bw < narrowband_ratio * fc
    flag = bool(narrow and fc > f_alias)
    aliases = enumerate_delay_aliases(peaks and max(peaks, key=lambda p: p[1])[0] or 0.0, fc, d_m, temp_c) if flag else []
    return PairAnalysis(pair, d_m, f_alias, fc, bw, narrow, peaks, aliases, flag)


def analyze_array(x: np.ndarray, fs: int, mic_positions_m, temp_c: float = 20.0, **kw) -> list[PairAnalysis]:
    """Run `analyze_pair` for every calibrated microphone pair (i < j)."""
    pos = np.asarray(mic_positions_m, float)
    out = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            out.append(analyze_pair(x[:, i], x[:, j], fs, float(np.linalg.norm(pos[j] - pos[i])), (i, j), temp_c, **kw))
    return out


# ------------------------------------------------------------------ onset precedence
def onset_flux_weights(x: np.ndarray, fs: int, nfft: int = 512, hop: int = 128, window_ms: float = 20.0,
                       sustain_weight: float = 0.2, k_mad: float = 3.0):
    """Haas-inspired onset weighting from positive spectral flux (heuristic, not a direct-sound proof).

    Returns (frame_times_s, weights, onset_times_s). weight = 1 for `window_ms` after each detected
    onset, `sustain_weight` otherwise, so sustained vowels/reverberant tails count less. No
    reliable onset -> all frames get `sustain_weight` and the caller should preserve ambiguity."""
    z = np.abs(stft_frames(np.asarray(x, float), nfft, hop))
    flux = np.maximum(np.diff(np.log1p(z * 100.0), axis=0), 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    t = (np.arange(len(flux)) * hop + nfft / 2) / fs
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-12
    thr = med + k_mad * 1.4826 * mad
    cand, _ = signal.find_peaks(flux, height=thr, distance=max(1, int(0.05 * fs / hop)))
    onsets = t[cand]
    w = np.full(len(t), sustain_weight)
    for o in onsets:
        w[(t >= o - hop / fs) & (t <= o + window_ms / 1000.0)] = 1.0
    return t, w, [float(o) for o in onsets]


def predict_pair_tau(cal, r_hat, i: int = 0, j: int = 1, temp_c: float = 20.0) -> np.ndarray:
    """Predicted delay for pair (i, j) from an `av_scene.schemas.ArrayCalibration`.

    Exactly ONE prediction model is used. `fitted_pair`: the empirical sine fit as-is (its
    amplitude already contains any baffle effect). `geometric`: physical positions with the
    optional `baffle_scale` effective-baseline factor and the pair's channel-offset difference."""
    if cal.prediction_model == "fitted_pair":
        fp = cal.fitted_pair
        if fp is None:
            raise ValueError("prediction_model='fitted_pair' requires fitted_pair parameters")
        return fitted_pair_tau(fp.delay_amp_s, fp.axis_deg, fp.offset_s, r_hat)
    pos = np.asarray(cal.mic_positions_m, float)
    off = cal.channel_offsets_s
    pair_offset = (off[i] - off[j]) if len(off) > max(i, j) else 0.0
    return tau_geom(r_hat, pos[i], pos[j], temp_c, cal.baffle_scale) + pair_offset


def ray_delay_score(lags: np.ndarray, cc: np.ndarray, tau_pred_s: float, sigma_s: float) -> float:
    """Evidence in [0, 1] that a ray's predicted delay is supported by the measured GCC curve.

    Gaussian-weighted maximum of the (normalised, non-negative) curve around the prediction."""
    c = np.maximum(cc, 0.0) / max(float(np.max(cc)), 1e-12)
    wgt = np.exp(-0.5 * ((lags - tau_pred_s) / max(sigma_s, 1e-9)) ** 2)
    return float(np.max(c * wgt))
