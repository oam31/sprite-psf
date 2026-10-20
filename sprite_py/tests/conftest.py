"""Reference data produced by the C++ SPRITE v1.0 (see ../reference/README.md).

Conventions of the dumps: 2D images written from iSAP ``Ifloat`` objects are the
transpose of the natural orientation used by the Python port; 3D cubes are
already in natural orientation.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

GOLDEN = Path(__file__).resolve().parents[2] / "reference" / "golden"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long end-to-end runs")


class Golden:
    def __init__(self, path):
        self.path = path

    def fits(self, name):
        return fits.getdata(self.path / name).astype(float)

    def image(self, name):
        """2D iSAP image in natural orientation."""
        return self.fits(name).T

    def txt(self, name):
        return np.loadtxt(self.path / name)

    @property
    def shifts(self):
        """C++ shifts as ``(dy, dx)`` in HR pixels."""
        return np.c_[self.txt("shift1.txt"), self.txt("shift0.txt")]

    @property
    def centroids(self):
        return np.c_[self.txt("cent1.txt"), self.txt("cent0.txt")]


def _golden(sub):
    path = GOLDEN / sub
    if not path.exists():
        pytest.skip(f"reference data not found in {path}")
    return Golden(path)


@pytest.fixture(scope="session")
def units():
    return _golden("units_t2")


@pytest.fixture(scope="session")
def units_t24():
    return _golden("units_t24")


@pytest.fixture(scope="session")
def golden_root():
    return _golden("")


@pytest.fixture(scope="session")
def cube(golden_root):
    return golden_root.fits("cube_test.fits")


def assert_rel(actual, expected, rtol):
    """Max absolute error relative to the max of ``expected``."""
    err = np.abs(np.asarray(actual) - np.asarray(expected)).max() / np.abs(expected).max()
    assert err < rtol, f"relative error {err:.3e} >= {rtol:.1e}"
