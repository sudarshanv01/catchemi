import pytest

from typing import Dict, List, Union, Tuple

import numpy as np
import numpy.typing as npt

from catchemi.core import BaseCalculation, FixedDimensionCalculation


@pytest.fixture
def fixed_dimension_inputs() -> Dict[str, Union[np.ndarray, List[np.ndarray], float]]:
    """Returns data form a calculation with electronic structure outputs that
    are of a fixed dimension (i.e. energies and pdos have fixed ndim)."""
    number_of_metals = 10
    _eps = np.linspace(-10, 10, 100)
    eps = np.vstack([_eps for _ in range(number_of_metals)])
    pdos = np.random.rand(number_of_metals, len(_eps))
    pdos = np.abs(pdos)
    coupling_sd = np.random.rand(number_of_metals, 1)
    coupling_sd = np.abs(coupling_sd)
    eps_a = np.random.rand(1)[0]
    alpha = np.abs(np.random.rand(1)[0])
    beta = np.abs(np.random.rand(1)[0])
    Delta0 = np.abs(np.random.rand(1)[0])
    data = {
        "eps": eps,
        "pdos": pdos,
        "coupling_sd": coupling_sd,
        "eps_a": eps_a,
        "alpha": alpha,
        "beta": beta,
        "Delta0": Delta0,
    }
    return data


@pytest.fixture
def variable_dimension_inputs() -> Dict[
    str, Union[np.ndarray, List[np.ndarray], float]
]:
    """Returns data from a calculation with electronic structure outputs
    that are of a variable dimension (i.e. eps and pdos have variable
    ndim)."""
    number_of_metals = 10
    number_of_datapoints = [
        100 + np.random.randint(-10, 10) for _ in range(number_of_metals)
    ]
    eps = [np.linspace(-10, 10, n) for n in number_of_datapoints]
    pdos = [np.abs(np.random.rand(n)) for n in number_of_datapoints]
    coupling_sd = np.random.rand(number_of_metals, 1)
    coupling_sd = np.abs(coupling_sd)
    eps_a = np.random.rand(1)[0]
    alpha = np.abs(np.random.rand(1)[0])
    beta = np.abs(np.random.rand(1)[0])
    Delta0 = np.abs(np.random.rand(1)[0])
    data = {
        "eps": eps,
        "pdos": pdos,
        "coupling_sd": coupling_sd,
        "eps_a": eps_a,
        "alpha": alpha,
        "beta": beta,
        "Delta0": Delta0,
    }
    return data


def test_BaseCalculation(fixed_dimension_inputs, variable_dimension_inputs):
    expected_hybridization_energy = np.repeat(
        -1.2, fixed_dimension_inputs["eps"].shape[0]
    )
    expected_orthogonalization_energy = np.repeat(
        1.0, fixed_dimension_inputs["eps"].shape[0]
    )
    expected_chemisorption_energy = (
        expected_hybridization_energy + expected_orthogonalization_energy
    )

    class TestCalculation(BaseCalculation):
        def get_hybridization_energy(self) -> npt.ArrayLike:
            return expected_hybridization_energy

        def get_orthogonalization_energy(self) -> npt.ArrayLike:
            return expected_orthogonalization_energy

        def get_chemisorption_energy(self) -> npt.ArrayLike:
            return self.get_hybridization_energy() + self.get_orthogonalization_energy()

    test_calculation_fixed_dim_inputs = TestCalculation(**fixed_dimension_inputs)
    assert test_calculation_fixed_dim_inputs.eps.ndim == 2
    assert test_calculation_fixed_dim_inputs.pdos.ndim == 2
    assert test_calculation_fixed_dim_inputs.coupling_sd.ndim == 2
    assert test_calculation_fixed_dim_inputs.eps_a.ndim == 0
    assert np.allclose(
        test_calculation_fixed_dim_inputs.get_hybridization_energy(),
        expected_hybridization_energy,
    )
    assert np.allclose(
        test_calculation_fixed_dim_inputs.get_orthogonalization_energy(),
        expected_orthogonalization_energy,
    )
    assert np.allclose(
        test_calculation_fixed_dim_inputs.get_chemisorption_energy(),
        expected_chemisorption_energy,
    )

    test_calculation_variable_dim_inputs = TestCalculation(**variable_dimension_inputs)
    assert len(test_calculation_variable_dim_inputs.eps) == len(
        variable_dimension_inputs["eps"]
    )
    assert len(test_calculation_variable_dim_inputs.pdos) == len(
        variable_dimension_inputs["pdos"]
    )
    assert len(test_calculation_variable_dim_inputs.coupling_sd) == len(
        variable_dimension_inputs["coupling_sd"]
    )
    assert test_calculation_variable_dim_inputs.eps_a.ndim == 0


def test_FixedDimensionCalulation(fixed_dimension_inputs):
    test_calculation = FixedDimensionCalculation(**fixed_dimension_inputs)
    assert test_calculation.eps.ndim == 2
    assert test_calculation.pdos.ndim == 2
    assert test_calculation.coupling_sd.ndim == 2
    assert test_calculation.eps_a.ndim == 0
    hybridization_energy = test_calculation.get_hybridization_energy()
    assert hybridization_energy.ndim == 2
    assert np.all(hybridization_energy <= 0)
    orthogonalization_energy = test_calculation.get_orthogonalization_energy()
    assert orthogonalization_energy.ndim == 2
    assert np.all(orthogonalization_energy >= 0)
