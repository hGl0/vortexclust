import logging
import pyproj
import pytest
import numpy as np
from vortexclust.analysis.geometry import compute_ellipse

logger = logging.getLogger(__name__)

def test_compute_ellipse():
    # Valid inputs
    area = 1000000  # km^2
    ar = 2.0
    theta = np.radians(30)  # radians
    loncent = 0.0
    latcent = 85.0
    num_points = 360

    x, y, proj = compute_ellipse(area, ar, theta, loncent, latcent, num_points)

    # Check shapes
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(proj, pyproj.Proj)

    # roughly centered?
    center_x, center_y = proj(loncent, latcent)
    assert np.allclose(np.mean(x), center_x, atol=1e4)
    assert np.allclose(np.mean(y), center_y, atol=1e4)

    # Check ellipse is closed: first and last points should be near (since endpoint=False)
    assert not np.allclose(x[0], x[-1])
    assert not np.allclose(y[0], y[-1])


@pytest.mark.parametrize("area, ar, theta, loncent, latcent, expected_exception", [
    (-100, 2.0, 0.0, 0.0, 85, ValueError), # invalid area
    (1000, -1.0, 0.0, 0.0, 85, ValueError), # invalid ar
    ("not_a_number", 2.0, 0.0, 0.0, 85, TypeError),
    (1000, 2.0, 0.0, 181, 85, ValueError), # invalid loncent
    (1000, 2.0, 0.0, -181, 85, ValueError),
    (1000, 2.0, 0.0, 0.0, 91, ValueError)
])
def test_compute_ellipse_invalid(area, ar, theta, loncent, latcent, expected_exception):
    with pytest.raises(expected_exception):
        compute_ellipse(area, ar, theta, loncent, latcent)

