import pytest

import numpy as np
import numpy.typing as npt

from conftest import computed_inputs

from catchemi.fitting import BaseFitting, FixedDimFitting


@pytest.fixture
def initial_guess_parameters():
    def _initial_guess_parameters(num_parameters: int = 2) -> npt.ArrayLike:
        alpha = np.random.rand(num_parameters)
        beta = np.random.rand(num_parameters)
        Delta0 = 0.1
        gamma = 0.1
        data = {
            "alpha": alpha,
            "beta": beta,
            "Delta0": Delta0,
            "gamma": gamma,
        }
        return data

    return _initial_guess_parameters


@pytest.fixture
def sequence_of_inputs(computed_inputs):
    inputs = []
    for computed_input in computed_inputs():
        inputs.append(computed_input)
    return inputs


def test_BaseFitting(sequence_of_inputs):
    class DummyFitting(BaseFitting):
        def error(self, *args, **kwargs):
            pass

        def get_combined_eps(self):
            pass

        def get_combined_pdos(self):
            pass

        def get_combined_coupling_sd(self):
            pass

    fitting = DummyFitting(sequence_of_inputs)
    assert fitting.inputs == sequence_of_inputs


def test_FixedDimFitting(sequence_of_inputs, initial_guess_parameters):
    fitting = FixedDimFitting(sequence_of_inputs)
