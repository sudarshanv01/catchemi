import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput, CombinedInput, FixedDimCombinedInput

from conftest import computed_inputs


def test_BaseInput(computed_inputs):
    for computed_input in computed_inputs():
        dummy_input = BaseInput(**computed_input)
        assert np.allclose(dummy_input.eps, computed_input["eps"])
        assert np.allclose(dummy_input.pdos, computed_input["pdos"])
        assert np.allclose(dummy_input.coupling_sd, computed_input["coupling_sd"])
        assert np.allclose(dummy_input.dft_energy, computed_input["dft_energy"])


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
        assert np.allclose(_combined_input.dft_energy, _computed_input.dft_energy)


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
        assert np.allclose(_combined_input.dft_energy, _computed_input.dft_energy)

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

    expected = [input.dft_energy for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_dft_energy()
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

    expected = [input.dft_energy for input in all_inputs]
    expected = np.asarray(expected)
    output = combined_input.get_combined_dft_energy()
    assert np.allclose(output, expected)
