import numpy as np
from mppi_racer.mppi import MPPI, MPPIParams
from mppi_racer.track import Track


def test_fresh_candidates_can_escape_saturated_braking_warm_start():
    theta = np.linspace(0, 2*np.pi,100,endpoint=False)
    track = Track(10*np.cos(theta),10*np.sin(theta),np.full(100,3.),np.full(100,1.5),np.full(100,1.5))
    params = MPPIParams(samples=256,exploration_fraction=.25)
    ctrl = MPPI(track,params,seed=2026)
    ctrl.u[:,1] = -params.a_max
    ctrl.prev_u[1] = -params.a_max
    result = ctrl.step([10.,0.,np.pi/2,0.],[])
    assert result.accel > 0, 'a stale saturated braking plan must not trap a stopped car on a clear track'


def test_exploration_is_opt_in_for_existing_configuration():
    assert MPPIParams().exploration_fraction == 0.0
