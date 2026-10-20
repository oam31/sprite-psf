import numpy as np

from conftest import assert_rel
from sprite.first_guess import first_guess, lanczos_stack, lanczos_upsample, mr_support_filter
from sprite.wavelets import Undecimated79

RTOL = 1e-6


def test_lanczos_stack_cpp_compat(units, cube):
    stack = lanczos_stack(cube, 2, units.shifts / 2, cpp_compat=True)
    assert_rel(stack, units.image("stack.fits"), RTOL)


def test_mr_support_filter(units):
    out = mr_support_filter(units.image("stack.fits"), Undecimated79(6))
    assert_rel(out, units.image("stack_mrsupp.fits"), RTOL)


def test_first_guess(units, cube):
    x0 = first_guess(cube, 2, units.shifts / 2, Undecimated79(6), cpp_compat=True)
    assert_rel(x0, units.image("model.fits"), RTOL)


def _gaussian(shape, cy, cx, s=1.3):
    yy, xx = np.indices(shape)
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))


def test_lanczos_upsample_registration():
    """A frame displaced by (dy, dx) is re-centred when resampled at (+dy, +dx)."""
    dy, dx = 0.3, -0.2
    frame = _gaussian((32, 32), 16 + dy, 16 + dx)
    ref = lanczos_upsample(_gaussian((32, 32), 16, 16), 2, (0.0, 0.0))
    assert np.abs(lanczos_upsample(frame, 2, (dy, dx)) - ref).max() < 0.01 * ref.max()
    # what the C++ stacking does: opposite sign, swapped axes
    assert np.abs(lanczos_upsample(frame, 2, (-dx, -dy)) - ref).max() > 0.05 * ref.max()


def test_stack_fix_registers_frames():
    """The corrected stack aligns shifted frames; the C++ one does not."""
    shifts = np.array([[0.0, 0.0], [0.1, 0.45], [0.4, -0.3], [-0.2, 0.3]])
    frames = np.array([_gaussian((32, 32), 16 + dy, 16 + dx) for dy, dx in shifts])
    truth = lanczos_upsample(frames[0], 2, (0.0, 0.0))
    fixed = lanczos_stack(frames, 2, shifts, cpp_compat=False)
    compat = lanczos_stack(frames, 2, shifts, cpp_compat=True)
    err = lambda s: np.abs(s - truth).max() / truth.max()
    assert err(fixed) < 0.02
    assert err(compat) > 5 * err(fixed)
