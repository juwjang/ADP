import numpy as np
import pytest
from mppi_racer.track import Track


@pytest.mark.parametrize('window', [(3, 12), (1, 5)])
def test_projection_matches_original_search_at_lap_boundary(window):
    th = np.linspace(0, 2*np.pi, 100, endpoint=False)
    tr = Track(10*np.cos(th), 10*np.sin(th), np.ones(100), np.ones(100), np.ones(100))
    rng = np.random.default_rng(2026)
    hint = rng.integers(-tr.n, 2*tr.n, size=1024)
    x, y = tr.x[hint % tr.n]+rng.normal(0,.2,1024), tr.y[hint % tr.n]+rng.normal(0,.2,1024)
    back, fwd = window
    cand = (hint[:, None]+np.arange(-back, fwd+1)) % tr.n
    d2 = (tr.x[cand]-x[:,None])**2+(tr.y[cand]-y[:,None])**2
    idx = cand[np.arange(len(hint)), np.argmin(d2, axis=-1)]
    dx, dy = x-tr.x[idx], y-tr.y[idx]
    expected = (idx, (idx*tr.spacing+dx*tr.cos_psi[idx]+dy*tr.sin_psi[idx]) % tr.length,
                -dx*tr.sin_psi[idx]+dy*tr.cos_psi[idx])
    actual = tr.project(x,y,hint,back,fwd)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-13)
