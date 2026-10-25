"""3D pair-delay projection (spec 5.5 item 2; Phase 1 contract 1). Seeded, CPU-only."""

import numpy as np
import pytest

from av_scene.schemas import ArrayCalibration, FittedPair
from locate.av_spatial import (baseline_from_pair_axis, c_sound, curvature_delay_sigma, direction_from_az_el,
                               fitted_pair_tau, fraunhofer_range, naive_bearing_from_ray, naive_bearing_from_tau,
                               parallax_sigma, predict_pair_tau, ray_from_pixel, source_position, tau_exact,
                               tau_geom)
from locate.pair import PairCalibration, frame_delays

P1, P2 = np.array([-0.07, 0.0, 0.0]), np.array([0.07, 0.0, 0.0])   # 14 cm baseline along +x
D = 0.14


def frac_delay(x, d_samples):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), len(x))


def test_speed_of_sound_temperature_model():
    assert c_sound(20.0) == pytest.approx(343.42, abs=0.01)
    assert c_sound(0.0) == pytest.approx(331.3)
    assert c_sound(35.0) > c_sound(20.0) > c_sound(5.0)


@pytest.mark.parametrize("el", [0.0, 20.0, 45.0, 60.0])
def test_ray_dot_baseline_matches_exact_far_field_limit(el):
    az = 40.0
    r = direction_from_az_el(az, el)
    tau = tau_geom(r, P1, P2)
    # Closed form: d*cos(el)*sin(az)/c
    assert tau == pytest.approx(D * np.cos(np.radians(el)) * np.sin(np.radians(az)) / c_sound(), abs=1e-12)
    # Exact hyperboloid delay of a distant point converges to the far-field value.
    assert tau_exact(1e5 * r, P1, P2) == pytest.approx(tau, abs=2e-9)


def test_naive_horizontal_inversion_errs_with_elevation():
    az = 40.0
    errs = []
    for el in (0.0, 30.0, 60.0):
        r = direction_from_az_el(az, el)
        tau = tau_geom(r, P1, P2)
        assert naive_bearing_from_ray(r) == pytest.approx(az, abs=1e-9)   # atan2(rx, rz) keeps azimuth
        errs.append(abs(naive_bearing_from_tau(tau, D) - az))              # asin(c tau / d) does not
    assert errs[0] < 1e-6 and errs[1] > 5.0 and errs[2] > 15.0 and errs[1] < errs[2]
    # The 3D comparison is exact: elevation changes the predicted delay and the ray predicts it.
    assert tau_geom(direction_from_az_el(az, 60.0), P1, P2) != pytest.approx(
        tau_geom(direction_from_az_el(az, 0.0), P1, P2), abs=1e-5)


def test_channel0_later_is_positive_and_matches_frame_delays():
    rng = np.random.default_rng(10)
    fs = 48000
    a = rng.standard_normal(fs)
    r = direction_from_az_el(40.0, 0.0)
    tau = float(tau_geom(r, P1, P2))
    assert tau > 0                                    # source toward +b: channel 0 (p1) is farther
    x = np.stack([frac_delay(a, tau * fs), a], 1)    # channel 0 later by tau
    _, measured, _ = frame_delays(x, fs, 4096, 2400)
    assert np.median(measured) == pytest.approx(tau, abs=3e-6)


def test_swapping_channels_negates_delay():
    r = direction_from_az_el(-25.0, 15.0)
    assert tau_geom(r, P2, P1) == pytest.approx(-tau_geom(r, P1, P2))
    assert tau_exact(3 * r, P2, P1) == pytest.approx(-tau_exact(3 * r, P1, P2))


@pytest.mark.parametrize("axis_deg", [-110.0, -45.0, 0.0, 30.0, 135.0])
def test_axis_deg_conversion_matches_pair_model(axis_deg):
    cal = PairCalibration(delay_amp_s=15e-6, axis_deg=axis_deg, offset_s=-1e-6, noise_s=3e-6)
    b_hat = baseline_from_pair_axis(axis_deg)
    # equivalent baseline azimuth is axis + 90 deg, not axis
    az = np.degrees(np.arctan2(b_hat[0], b_hat[2])) % 360
    assert az == pytest.approx((axis_deg + 90.0) % 360, abs=1e-9)
    for theta in np.arange(-180, 180, 15.0):
        r = direction_from_az_el(theta, 0.0)
        assert fitted_pair_tau(cal.delay_amp_s, cal.axis_deg, cal.offset_s, r) == pytest.approx(
            float(cal.predict(theta, 0.0)), abs=1e-12)


def test_baffle_scale_is_not_applied_to_a_fitted_amplitude():
    r = direction_from_az_el(35.0, 10.0)
    fp = FittedPair(delay_amp_s=15e-6, axis_deg=-110.0, offset_s=-1e-6, noise_s=3e-6)
    base = dict(calibration_id="t", mic_positions_m=[list(P1), list(P2)], channel_order=["a", "b"])
    plain = ArrayCalibration(**base, prediction_model="fitted_pair", fitted_pair=fp, baffle_scale=1.0)
    scaled = ArrayCalibration(**base, prediction_model="fitted_pair", fitted_pair=fp, baffle_scale=1.5)
    assert predict_pair_tau(scaled, r) == pytest.approx(predict_pair_tau(plain, r), abs=1e-15)
    # The physical-geometry branch is where baffle_scale belongs (offset-free pair).
    g1 = ArrayCalibration(**base, prediction_model="geometric", baffle_scale=1.0)
    g2 = ArrayCalibration(**base, prediction_model="geometric", baffle_scale=1.5)
    assert predict_pair_tau(g2, r) == pytest.approx(1.5 * predict_pair_tau(g1, r))
    with pytest.raises(ValueError):
        ArrayCalibration(**base, baffle_scale=0.9)


def test_pixel_rays_and_camera_translation_with_known_depth():
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    centre = ray_from_pixel(cx, cy, fx, fy, cx, cy)
    assert centre == pytest.approx([0, 0, 1])
    right = ray_from_pixel(cx + 200, cy, fx, fy, cx, cy)
    up = ray_from_pixel(cx, cy - 200, fx, fy, cx, cy)      # image-up is array +y
    assert right[0] > 0 and up[1] > 0 and np.linalg.norm(right) == pytest.approx(1.0)
    assert tau_geom(centre, P1, P2) == pytest.approx(0.0, abs=1e-15)
    assert tau_geom(right, P1, P2) > 0
    # camera 20 cm to the right of the array centre, source 1.5 m deep along the camera ray
    t = np.array([0.2, 0.0, 0.0])
    s = source_position(right, 1.5, None, t)
    far = float(tau_geom(right, P1, P2))
    near = tau_exact(s, P1, P2)
    assert abs(near - far) > 5e-6                        # translation matters at finite depth
    assert abs(near - far) < 3e-4


def test_parallax_floor_guard_and_withhold():
    assert parallax_sigma([0, 0, 0]).sigma_rad == 0.0
    p = parallax_sigma([0.1, 0.0, 0.0], r_min_m=0.5)
    assert p.confident and p.sigma_rad == pytest.approx(np.arcsin(0.2))
    assert p.sigma_rad >= np.arctan(0.2)                 # never below the lateral-offset floor
    w = parallax_sigma([0.6, 0.0, 0.0], r_min_m=0.5)     # source may sit closer than the offset
    assert not w.confident and w.sigma_rad == pytest.approx(np.pi / 2)


def test_fraunhofer_screen_and_curvature_inflation():
    rff = fraunhofer_range(0.14, 4000.0)
    assert rff == pytest.approx(2 * 0.14 ** 2 / (c_sound() / 4000.0))
    assert curvature_delay_sigma(0.14, 2 * rff, 4000.0) == 0.0
    assert curvature_delay_sigma(0.14, 0.5 * rff, 4000.0) > 0.0
