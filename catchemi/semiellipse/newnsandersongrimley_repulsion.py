"""Determine the elements of the Newns-Anderson-Grimley model."""
import numpy as np
from catchemi import NewnsAndersonGrimleyNumerical


class NewnsAndersonGrimleyRepulsion(NewnsAndersonGrimleyNumerical):
    """Class meant to enable fitting of parameters to the Newns-Anderson
    Grimley model of chemisorption. This class is meant to facilitate
    incorporating repulsive interations through the Newns-Anderson Grimley
    model of chemisorption."""

    def __init__(
        self,
        Vsd,
        eps_a,
        eps_d,
        width,
        eps,
        Delta0_mag=0.0,
        eps_sp_max=15,
        eps_sp_min=-15,
        precision=50,
        verbose=False,
        alpha=0.0,
        beta=0.0,
        constant_offset=0,
        spin=2,
    ):
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
            alpha,
            spin,
        )
        self.alpha = alpha
        # store the initial value of alpha fed in
        self.alpha_initial = alpha
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

    def get_chemisorption_energy(self):
        """Utility function for returning
        the chemisorption energy."""
        if self.verbose:
            print("Computing the chemisorption energy...")
        if self.chemisorption_energy is not None:
            return self.chemisorption_energy
        else:
            self.compute_chemisorption_energy()
            return float(self.chemisorption_energy)

    def get_orthogonalisation_energy(self):
        """Utility function for returning
        the orthogonalisation energy."""
        if self.verbose:
            print("Computing the orthogonalisation energy...")
        if self.orthogonalisation_energy is not None:
            return self.orthogonalisation_energy
        else:
            self.compute_chemisorption_energy()
            return float(self.orthogonalisation_energy)

    def compute_chemisorption_energy(self):
        """Compute the chemisorption energy based on the
        parameters of the class, a linear repulsion term
        and the hybridisation energy coming from the
        Newns-Anderson model."""

        self.get_hybridisation_energy()
        self.get_occupancy()
        self.get_dband_filling()

        # The hybridisation energy is the chemisorption energy
        # for the Newns-Anderson Grimley model because it includes overlap.
        self.chemisorption_energy = self.hybridisation_energy

        # To isolate the orthogonolsation energy, set alpha to
        # zero and determine the no-repulsion energy, then
        # subtract it from the chemisorption energy.
        self.alpha = 0.0
        self.calculate_hybridisation_energy()
        self.orthogonalisation_energy = (
            self.chemisorption_energy - self.hybridisation_energy
        )
        print("Orthogonalisation energy is %1.2f" % self.orthogonalisation_energy)
        assert (
            self.orthogonalisation_energy >= 0.0
        ), "Orthogonalisation energy must be positive."
        # Store the hybridisation energy for no alpha
        hyb_energy = self.hybridisation_energy

        # Now run it with the initial alpha again
        # just to get the same quantities as what was asked for
        self.alpha = self.alpha_initial
        self.calculate_hybridisation_energy()
        # Replace the hybridisation energies with the alpha=0 value
        self.hybridisation_energy = hyb_energy

        # Add the constant offset to the chemisorption energy
        self.chemisorption_energy += self.constant_offset
