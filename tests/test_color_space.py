import numpy as np

from utils.color_space import rgb_to_hsv, rgb_to_lab, lab_to_rgb, detect_skin_mask


def test_rgb_lab_roundtrip_reasonable_error():
    rng = np.random.default_rng(0)
    img = rng.random((64, 80, 3), dtype=np.float32)
    lab = rgb_to_lab(img)
    back = lab_to_rgb(lab)
    err = np.abs(img - back)
    assert float(err.mean()) < 3e-3
    assert float(err.max()) < 2e-2


def test_rgb_to_hsv_range():
    rng = np.random.default_rng(1)
    img = rng.random((32, 40, 3), dtype=np.float32)
    hsv = rgb_to_hsv(img)
    assert hsv.shape == img.shape
    assert np.isfinite(hsv).all()
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    assert (h >= -1e-3).all() and (h <= 360.0 + 1e-3).all()
    assert (s >= -1e-6).all() and (s <= 1.0 + 1e-6).all()
    assert (v >= -1e-6).all() and (v <= 1.0 + 1e-6).all()


def test_detect_skin_mask_output_sane():
    img = np.zeros((80, 120, 3), dtype=np.float32)
    mask = detect_skin_mask(img, blur_radius=3)
    assert mask.shape[:2] == img.shape[:2]
    assert mask.dtype == np.float32
    assert np.isfinite(mask).all()
    assert (mask >= 0.0).all() and (mask <= 1.0).all()
    assert float(mask.mean()) < 0.25


def test_detect_skin_mask_prefers_skin_like_patch():
    h, w = 120, 160
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = np.array([0.12, 0.14, 0.18], dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.55, w * 0.5
    ry, rx = h * 0.22, w * 0.18
    region = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    img[region] = np.array([0.78, 0.58, 0.48], dtype=np.float32)

    mask = detect_skin_mask(img, blur_radius=3)
    inside = float(mask[region].mean())
    outside = float(mask[~region].mean())
    assert inside > outside
