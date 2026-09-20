"""Check physical starts survive NumPyro's bounded-parameter transforms."""
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import constrain_fn
from binary_masses.hierarchical_metallicity import _unconstrain_chain_initial_values


def test_both_shape_starts_roundtrip_through_mcmc_coordinates():
    bounds = np.log([[0.0007, 0.0033], [24, 44], [2.5, 13.5]])
    def model():
        numpyro.sample("f_outlier", dist.Beta(3, 12))
        for name, (lo, hi) in zip(["log_b", "log_uc", "log_c"], bounds):
            numpyro.sample(name, dist.Uniform(lo, hi))
        numpyro.sample("c0", dist.Normal(0, 1))

    starts = {
        "f_outlier": np.array([0.2, 0.21]),
        "log_b": np.log([0.002544, 0.00103]),
        "log_uc": np.log([35.67, 40.98]),
        "log_c": np.log([3.1, 11.53]),
        "c0": np.array([0.1, 0.12]),
    }
    raw = _unconstrain_chain_initial_values(model, (), starts)
    np.testing.assert_allclose(raw["f_outlier"], np.log(starts["f_outlier"]/(1-starts["f_outlier"])), rtol=1e-6)
    for i in range(2):
        recovered = constrain_fn(model, (), {}, {k: v[i] for k, v in raw.items()})
        for name in starts:
            np.testing.assert_allclose(recovered[name], starts[name][i], rtol=1e-6, atol=1e-6)
