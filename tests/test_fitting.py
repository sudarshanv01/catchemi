import pytest

import numpy as np
import numpy.typing as npt

from conftest import computed_inputs

from catchemi.input import BaseInput
from catchemi.input import FixedDimCombinedInput
from catchemi.fitting import BaseFitting, FixedDimFitting


@pytest.fixture
def initial_guess_parameters():
    def _initial_guess_parameters(
        num_parameters: int = 2, spin_polarized=False
    ) -> npt.ArrayLike:
        if spin_polarized:
            spin_dimension = 2
        else:
            spin_dimension = 1
        eps_a = np.random.rand(num_parameters, spin_dimension)
        alpha = np.random.rand(num_parameters, spin_dimension)
        beta = np.random.rand(num_parameters, spin_dimension)
        Delta0 = 0.1
        gamma = 0.1
        data = {
            "alpha": alpha,
            "beta": beta,
            "Delta0": Delta0,
            "gamma": gamma,
            "eps_a": eps_a,
        }
        return data

    return _initial_guess_parameters


@pytest.fixture
def sequence_of_inputs(computed_inputs):
    def _sequence_of_inputs(*args, **kwargs):
        inputs = []
        for computed_input in computed_inputs(*args, **kwargs):
            inputs.append(BaseInput(**computed_input))
        return inputs

    return _sequence_of_inputs


def test_BaseFitting(sequence_of_inputs):
    class DummyFitting(BaseFitting):
        def get_predicted_chemisorption_energies(self, *args, **kwargs):
            pass

        def get_combined_eps(self):
            pass

        def get_combined_pdos(self):
            pass

        def get_combined_coupling_sd(self):
            pass

        def get_mean_absolute_error(self, *args, **kwargs):
            pass

    fitting_inputs = sequence_of_inputs()
    fitting = DummyFitting(fitting_inputs)
    assert fitting.inputs == fitting_inputs


def test_FixedDimFitting(sequence_of_inputs, initial_guess_parameters):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=False)
    fitting = FixedDimFitting(fitting_inputs)
    predicted_chemisorption_energies = fitting.get_predicted_chemisorption_energies(
        **initial_guess_parameters()
    )
    assert predicted_chemisorption_energies.shape == (len(fitting.inputs), 1)
    assert np.all(predicted_chemisorption_energies <= 0)


def test_FixedDimFitting_spin_pol(sequence_of_inputs, initial_guess_parameters):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=True)
    fitting = FixedDimFitting(fitting_inputs)
    predicted_chemisorption_energies = fitting.get_predicted_chemisorption_energies(
        **initial_guess_parameters(spin_polarized=True)
    )
    assert predicted_chemisorption_energies.shape == (len(fitting.inputs), 1)
    assert np.all(predicted_chemisorption_energies <= 0)


def test_FixedDimFitting_mean_absolute_error(
    sequence_of_inputs, initial_guess_parameters
):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=False)
    fitting = FixedDimFitting(fitting_inputs)
    mean_absolute_error = fitting.get_mean_absolute_error(**initial_guess_parameters())
    assert isinstance(mean_absolute_error, float)
    assert mean_absolute_error >= 0


def test_FixedDimFitting_mean_absolute_error_spin_pol(
    sequence_of_inputs, initial_guess_parameters
):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=True)
    fitting = FixedDimFitting(fitting_inputs)
    mean_absolute_error = fitting.get_mean_absolute_error(
        **initial_guess_parameters(spin_polarized=True)
    )
    assert isinstance(mean_absolute_error, float)
    assert mean_absolute_error >= 0


def test_FixedDimFitting_least_squares_error(
    sequence_of_inputs, initial_guess_parameters
):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=False)
    fitting = FixedDimFitting(fitting_inputs)
    least_squares_error = fitting.get_least_squares_error(**initial_guess_parameters())
    assert isinstance(least_squares_error, float)
    assert least_squares_error >= 0


def test_FixedDimFitting_least_squares_error_spin_pol(
    sequence_of_inputs, initial_guess_parameters
):
    fitting_inputs = sequence_of_inputs(fixed_dim=True, spin_pol=True)
    fitting = FixedDimFitting(fitting_inputs)
    least_squares_error = fitting.get_least_squares_error(
        **initial_guess_parameters(spin_polarized=True)
    )
    assert isinstance(least_squares_error, float)
    assert least_squares_error >= 0
