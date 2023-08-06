import abc

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseInput:
    eps: npt.ArrayLike
    pdos: npt.ArrayLike
    coupling_sd: npt.ArrayLike
    eps_a: npt.ArrayLike

    def __post_init__(self):
        self.eps = np.asarray(self.eps)
        self.pdos = np.asarray(self.pdos)
        self.coupling_sd = np.asarray(self.coupling_sd)
        self.eps_a = np.asarray(self.eps_a)

    def __repr__(self):
        return f"""{self.__class__.__name__}:
                    eps: {self.eps}
                    pdos: {self.pdos}
                    coupling_sd: {self.coupling_sd}
                    eps_a: {self.eps_a}"""

    def is_spin_pol(self):
        return self.eps.ndim == 2

    @property
    def get_number_of_points(self):
        return self.eps.shape[-1]
