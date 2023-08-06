import pytest

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput, CombinedInput, FixedDimCombinedInput


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
                eps_a = np.random.rand(1)
                coupling_sd = np.random.rand(1)
            else:
                _eps = np.linspace(-10, 10, n)
                _pdos = np.abs(np.random.rand(n))
                eps = np.vstack((_eps, _eps))
                pdos = np.vstack((_pdos, _pdos))
                eps_a = np.random.rand(2).reshape(-1, 1)
                coupling_sd = np.random.rand(2).reshape(-1, 1)
            yield {
                "eps": eps,
                "pdos": pdos,
                "coupling_sd": coupling_sd,
                "eps_a": eps_a,
            }

    return _computed_inputs


def test_BaseInput(computed_inputs):
    for computed_input in computed_inputs():
        dummy_input = BaseInput(**computed_input)
        assert np.allclose(dummy_input.eps, computed_input["eps"])
        assert np.allclose(dummy_input.pdos, computed_input["pdos"])
        assert np.allclose(dummy_input.coupling_sd, computed_input["coupling_sd"])
        assert np.allclose(dummy_input.eps_a, computed_input["eps_a"])
        assert dummy_input.is_spin_pol() == False


def test_CombinedInput(computed_inputs):
    class DummyCombinedInput(CombinedInput):
        def get_combined_eps(self):
            pass

        def get_combined_pdos(self):
            pass

        def get_combined_coupling_sd(self):
            pass

    all_inputs = []
    for computed_input in computed_inputs():
        dummy_input = BaseInput(**computed_input)
        all_inputs.append(dummy_input)
    combined_input = DummyCombinedInput(all_inputs)
    for _combined_input, _computed_input in zip(combined_input, all_inputs):
        assert np.allclose(_combined_input.eps, _computed_input.eps)
        assert np.allclose(_combined_input.pdos, _computed_input.pdos)
        assert np.allclose(_combined_input.coupling_sd, _computed_input.coupling_sd)
        assert np.allclose(_combined_input.eps_a, _computed_input.eps_a)
        assert _combined_input.is_spin_pol() == False


def test_FixedDimCombinedInput(computed_inputs):
    all_inputs = []
    for computed_input in computed_inputs(fixed_dim=True):
        dummy_input = BaseInput(**computed_input)
        all_inputs.append(dummy_input)
    combined_input = FixedDimCombinedInput(all_inputs)
    for _combined_input, _computed_input in zip(combined_input, all_inputs):
        assert np.allclose(_combined_input.eps, _computed_input.eps)
        assert np.allclose(_combined_input.pdos, _computed_input.pdos)
        assert np.allclose(_combined_input.coupling_sd, _computed_input.coupling_sd)
        assert np.allclose(_combined_input.eps_a, _computed_input.eps_a)
        assert _combined_input.is_spin_pol() == False

    print(combined_input)

    assert len(combined_input) == len(all_inputs)
    expected = [input.eps for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_eps()
    assert np.allclose(output, expected)

    expected = [input.pdos for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_pdos()
    assert np.allclose(output, expected)

    expected = [input.coupling_sd for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_coupling_sd()
    assert np.allclose(output, expected)


def test_FixedDimCombinedInput_spin_pol(computed_inputs):
    all_inputs = []
    for computed_input in computed_inputs(fixed_dim=True, spin_pol=True):
        dummy_input = BaseInput(**computed_input)
        all_inputs.append(dummy_input)
    combined_input = FixedDimCombinedInput(all_inputs)
    for _combined_input, _computed_input in zip(combined_input, all_inputs):
        assert np.allclose(_combined_input.eps, _computed_input.eps)
        assert np.allclose(_combined_input.pdos, _computed_input.pdos)
        assert np.allclose(_combined_input.coupling_sd, _computed_input.coupling_sd)
        assert np.allclose(_combined_input.eps_a, _computed_input.eps_a)
        assert _combined_input.is_spin_pol() == True

    assert len(combined_input) == len(all_inputs)
    expected = [input.eps for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_eps()
    assert np.allclose(output, expected)

    expected = [input.pdos for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_pdos()
    assert np.allclose(output, expected)

    expected = [input.coupling_sd for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_coupling_sd()
    assert np.allclose(output, expected)
