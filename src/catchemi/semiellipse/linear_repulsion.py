"""Account for linear repulsive term from overlap."""

import numpy as np
from catchemi import NewnsAndersonNumerical
from flint import acb, arb, ctx


class NewnsAndersonLinearRepulsion(NewnsAndersonNumerical):
    def __init__(
        self,
        Vsd: float,
        eps_a: float,
        eps_d: float,
        width: float,
        eps: float,
        Delta0_mag: float = 0.0,
        eps_sp_max: float = 15,
        eps_sp_min: float = -15,
        precision: float = 50,
        verbose: bool = False,
        alpha: float = 0.0,
        beta: float = 0.0,
        constant_offset: float = 0.0,
        spin: int = 2,
        add_largeS_contribution: bool = False,
    ):
        """Class that provides the Newns-Anderson hybridisation
        energy along with the linear orthogonalisation energy.
        It subclasses NewnsAndersonNumerical for the Hybridisation
        energy and adds the orthogonalisation penalty separately.

        Args:
            Vsd (float): The adsorbate-metal hybridisation energy.
            eps_a (float): The adsorbate energy.
            eps_d (float): The adsorbate energy.
            width (float): The adsorbate energy.
            eps (float): The adsorbate energy.
            Delta0_mag (float, optional): The bare adsorbate-metal hybridisation energy. Defaults to 0.0.
            eps_sp_max (float, optional): The maximum energy for the self-energy. Defaults to 15.
            eps_sp_min (float, optional): The minimum energy for the self-energy. Defaults to -15.
            precision (int, optional): The precision for the self-energy. Defaults to 50.
            verbose (bool, optional): Whether to print out the progress. Defaults to False.
            alpha (float, optional): The linear repulsion coefficient. Defaults to 0.0.
            beta (float, optional): The linear repulsion coefficient. Defaults to 0.0.
            constant_offset (float, optional): A constant energy offset. Defaults to 0.0.
            spin (int, optional): The spin of the adsorbate. Defaults to 2.
            add_largeS_contribution (bool, optional): Whether to add the contribution for large S. Defaults to False.
        """
        Vak = np.sqrt(beta) * Vsd
        super().__init__(
            Vak,
            eps_a,
            eps_d,
            width,
            eps,
            Delta0_mag,
            eps_sp_max,
            eps_sp_min,
            precision,
            verbose,
            spin,
        )
        self.alpha = alpha
        self.beta = beta
        assert self.alpha >= 0.0, "alpha must be positive."
        assert self.beta >= 0.0, "beta must be positive."
        self.constant_offset = constant_offset

        # The goal is to find the chemisorption energy
        self.chemisorption_energy = None
        # Also store the orthogonalisation energy
        self.orthogonalisation_energy = None
        # Store the spin
        self.spin = spin
        # If required, add the contribution for large S
        self.add_largeS_contribution = add_largeS_contribution

    def get_chemisorption_energy(self):
        """Utility function for returning
        the chemisorption energy."""
        if self.verbose:
            print("Computing the chemisorption energy...")
        if self.chemisorption_energy is not None:
            return float(self.chemisorption_energy.real)
        else:
            self.compute_chemisorption_energy()
            return float(self.chemisorption_energy.real)

    def get_orthogonalisation_energy(self):
        """Utility function for returning
        the orthogonalisation energy."""
        if self.verbose:
            print("Computing the orthogonalisation energy...")
        if self.orthogonalisation_energy is not None:
            return float(self.orthogonalisation_energy.real)
        else:
            self.compute_chemisorption_energy()
            return float(self.orthogonalisation_energy.real)

    def compute_chemisorption_energy(self):
        """Compute the chemisorption energy based on the
        parameters of the class, a linear repulsion term
        and the hybridisation energy coming from the
        Newns-Anderson model."""

        self.get_hybridisation_energy()
        self.get_occupancy()
        self.get_dband_filling()
        self._convert_to_acb()

        # orthonogonalisation energy
        self.orthogonalisation_energy = -1 * self.alpha * self.Vak**2
        # Add large S contribution
        if self.add_largeS_contribution:
            largeS_cont1 = acb.pow(self.eps_a - self.eps_d, 2)
            largeS_cont1 += 4 * self.alpha * self.Vak**2 * (self.eps_a + self.eps_d)
            largeS_cont1 += 4 * self.Vak**2
            largeS_cont1 = acb.pow(largeS_cont1, 0.5)
            largeS_cont2 = acb.pow(self.eps_a - self.eps_d, 2)
            largeS_cont2 += 4 * self.Vak**2
            largeS_cont2 = acb.pow(largeS_cont2, 0.5)
            self.orthogonalisation_energy += 0.5 * (largeS_cont1 - largeS_cont2)
            # assert largeS_cont1.real - largeS_cont2.real <= arb('0.0')
        # Multiply by the pre-factor
        self.orthogonalisation_energy *= (
            -1 * self.spin * (self.occupancy.real + self.filling.real)
        )

        assert self.orthogonalisation_energy.real >= arb("0.0")

        # chemisorption energy is the sum of the hybridisation
        # and the orthogonalisation energy
        self.chemisorption_energy = (
            self.hybridisation_energy + self.orthogonalisation_energy
        )
        # Add the constant offset which is helpful for fitting routines
        self.chemisorption_energy += self.constant_offset
