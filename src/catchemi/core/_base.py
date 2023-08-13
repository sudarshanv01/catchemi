import abc

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseCalculation(abc.ABC):
    """Abstract base class for calculations.

    This class is used to perform calculations on the inputs. This
    class is not meant to be used directly, but rather to be subclassed.
    """

    eps: npt.ArrayLike
    pdos: npt.ArrayLike
    coupling_sd: npt.ArrayLike
    eps_a: float
    alpha: float
    beta: float
    Delta0: float
    spin_polarized: bool = False
    eps_f: float = 0

    @abc.abstractmethod
    def get_hybridization_energy(self) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_orthogonalization_energy(self) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_chemisorption_energy(self) -> npt.ArrayLike:
        pass
