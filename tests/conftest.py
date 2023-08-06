import pytest

import numpy as np
import numpy.typing as npt


@pytest.fixture
def computed_inputs():
    """Provide results of a fixed dimension DFT calculation for a non-
    spin-polarized DFT calculation."""

    def _computed_inputs(
        fixed_dim: bool = False,
        spin_pol: bool = False,
    ) -> npt.ArrayLike:
        number_of_metals = 10
        if fixed_dim:
            number_of_datapoints = [100 for _ in range(number_of_metals)]
        else:
            number_of_datapoints = [
                100 + np.random.randint(-10, 10) for _ in range(number_of_metals)
            ]
        for n in number_of_datapoints:
            if not spin_pol:
                eps = np.linspace(-10, 10, n)
                pdos = np.abs(np.random.rand(n))
                coupling_sd = np.random.rand(1)
            else:
                _eps = np.linspace(-10, 10, n)
                _pdos = np.abs(np.random.rand(n))
                eps = np.vstack((_eps, _eps))
                pdos = np.vstack((_pdos, _pdos))
                coupling_sd = np.random.rand(2).reshape(-1, 1)
            yield {
                "eps": eps,
                "pdos": pdos,
                "coupling_sd": coupling_sd,
            }

    return _computed_inputs
