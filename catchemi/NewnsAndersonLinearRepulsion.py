"""Account for linear repulsive term from overlap."""

from dataclasses import dataclass
import numpy as np
from scipy import signal
from scipy import integrate
from scipy import optimize
import warnings
from pprint import pprint
import mpmath as mp
from flint import acb, arb, ctx

@dataclass
class NewnsAndersonLinearRepulsion:
    """Class for fitting parameters for the Newns-Anderson and the
    d-band model with calculations performed with DFT. The class 
    expects all metal quantities for a particular eps_a (adsorbate) value.
    
    """
    width: list
    eps: list
    eps_a: float
    Vsd: list
    eps_sp_min: float = -15
    eps_sp_max: float = 15
    Delta0_mag: float = 0.0
    precision: int = 50

    def __post_init__(self):
        """Extra variables that are needed for the model."""
        # convert all lists to numpy arrays
        self.width = np.array(self.width)
        self.Vsd = np.array(self.Vsd)

    def fit_parameters(self, args, eps_ds):
        """Fit the parameters alpha, beta"""
        alpha, beta, constant_offset = args

        # Make sure that all the quantities are positive
        # Constant offset can be any sign
        alpha = abs(alpha)
        beta = abs(beta)

        # Store the hybridisation energy for all metals to compare later
        spd_hybridisation_energy = np.zeros(len(eps_ds))

        # Generate Vak based on the provided beta
        Vak = np.sqrt(beta) * self.Vsd 

        # We will need the occupancy of the single particle state
        self.na = np.zeros(len(eps_ds))
        self.filling = np.zeros(len(eps_ds))

        # Loop over all the metals
        for i, eps_d in enumerate(eps_ds):
            hybridisation = NewnsAndersonNumerical(
                Vak = Vak[i],
                eps_a = self.eps_a,
                eps_d = eps_d,
                width = self.width[i],
                eps = self.eps,
                eps_sp_max=self.eps_sp_max,
                eps_sp_min=self.eps_sp_min,
                Delta0_mag = self.Delta0_mag,
                precision=self.precision,
                verbose=False,
            )
            # The first component of the hybridisation energy
            # is the hybdridisation coming from the sp and d bands
            hybridisation.calculate_energy()
            spd_hybridisation_energy[i] = hybridisation.get_energy()
            # Get and store the occupancy because it will be needed
            # for the orthogonalisation energy term
            hybridisation.calculate_occupancy()
            self.na[i] = hybridisation.get_occupancy()
            self.filling[i] = hybridisation.get_dband_filling()

        # Ensure that the hybridisation energy is negative always
        assert all(spd_hybridisation_energy <= 0), "Hybridisation energy is negative"
        # Store the spd hybridisation energy
        self.spd_hybridisation_energy = spd_hybridisation_energy

        # orthonogonalisation energy
        ortho_energy = 2 * ( self.na +  self.filling ) * alpha * Vak**2
        ortho_energy = np.array(ortho_energy)

        # Ensure that the orthonogonalisation energy is positive always
        assert all(ortho_energy >= 0), "Orthogonalisation energy is positive now it is (%1.2f)"%(ortho_energy)

        # Add the orthogonalisation energy to the hybridisation energy
        hybridisation_energy = spd_hybridisation_energy + ortho_energy + constant_offset

        # Store the orthogonalisation energy for all metals
        self.ortho_energy = ortho_energy

        # Store the hybridisation energy for all metals
        self.hybridisation_energy = hybridisation_energy

        print('Return list:')
        print(alpha, beta, constant_offset)
        print(hybridisation_energy)

        return hybridisation_energy