import abc

from dataclasses import dataclass

from typing import Sequence

import numpy as np
import numpy.typing as npt

from catchemi.input import BaseInput


@dataclass
class CombinedInput(abc.ABC):
    inputs: Sequence[BaseInput]

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)

    def __next__(self):
        return next(self.inputs)

    @abc.abstractmethod
    def get_combined_eps(self):
        pass

    @abc.abstractmethod
    def get_combined_pdos(self):
        pass


@dataclass
class FixedDimCombinedInput(CombinedInput):
    inputs: Sequence[BaseInput]

    def get_combined_eps(self):
        return np.asarray([input.eps for input in self.inputs])

    def get_combined_pdos(self):
        return np.asarray([input.pdos for input in self.inputs])
