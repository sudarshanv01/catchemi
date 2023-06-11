import numpy as np
import numpy.typing as npt

from flint import acb, arb

from catchemi.semiellipse.numerical import NewnsAndersonNumerical


class NewnsAndersonGrimleyNumerical(NewnsAndersonNumerical):
    def __init__(
        self,
        Vak: float,
        eps_a: float,
        eps_d: float,
        width: float,
        eps: npt.ArrayLike,
        Delta0_mag: float = 0.0,
        eps_sp_max: float = 15.0,
        eps_sp_min: float = -15.0,
        precision: int = 50,
        verbose: bool = False,
        alpha: float = 0.0,
        spin: int = 2,
    ):
        """Perform the Newns-Anderson-Grimley model for
        chemisorption by subclassing the Newns-Anderson model.
        and then supplying the overlap elements. This class is meant
        to modify the Delta and Lambda terms of the Newns-Anderson
        model to those derived by Grimley.

        Args:
            Vak (float): The adsorbate-metal hybridisation energy.
            eps_a (float): The adsorbate energy.
            eps_d (float): The adsorbate energy.
            width (float): The adsorbate energy.
            eps (npt.ArrayLike): Energy grid.
            Delta0_mag (float, optional): The bare adsorbate-metal hybridisation energy. Defaults to 0.0.
            eps_sp_max (float, optional): The maximum energy for the self-energy. Defaults to 15.
            eps_sp_min (float, optional): The minimum energy for the self-energy. Defaults to -15.
            precision (int, optional): The precision for the self-energy. Defaults to 50.
            verbose (bool, optional): Whether to print out the progress. Defaults to False.
            alpha (float, optional): The linear repulsion coefficient. Defaults to 0.0.
            spin (int, optional): The spin of the adsorbate. Defaults to 2.
        """

        # Initialise the quantities using the Newns-Anderson parameters
        # In this class we will replace how Delta and Lambda are determined
        # including the overlap elements.
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

        print("Incorporating orthogonalisation using the Newns-Anderson-Grimley model.")

    def _create_Delta_arb(self, eps) -> acb:
        """Create a function for Delta based on arb.
        This function is supposed to modify the behaviour
        of the Newns-Anderson model when the overlap
        is non-negligible."""
        # The overlap elements are determined consistent witht the
        # proportionality between Sak and Vak used in the paper.
        self.Sak = -1 * self.alpha * self.Vak

        eps_ref = self.create_reference_eps(eps)
        # If the absolute value of the reference is
        # lower than 1 (in the units of wd) then
        # the Delta will be non-zero
        if acb.abs_lower(eps_ref) < arb("1.0"):
            Delta = acb(1.0) - acb.pow(eps_ref, 2)
            Delta = acb.pow(Delta, 0.5)
            # Multiply by the prefactor
            Delta *= (self.Sak * eps - self.Vak) ** 2
            # Normalise the area
            Delta /= self.wd
            Delta *= acb(2)
        else:
            # If eps is out of bounds there will
            # no Delta contribution
            Delta = acb(0.0)
        return Delta

    def _create_Delta_reg(self, eps) -> float:
        """Create a function for Delta for regular manipulations."""
        # The overlap elements are determined consistent with the
        # proportionality between Sak and Vak used in the paper.
        self.Sak = -1 * self.alpha * self.Vak

        eps_ref = self.create_reference_eps(eps)
        # If the absolute value of the reference is
        # lower than 1 (in the units of wd) then
        # the Delta will be non-zero
        if np.abs(eps_ref) < 1:
            Delta = 1.0 - eps_ref**2
            Delta = Delta**0.5
            # Multiply by the prefactor
            Delta *= (self.Sak * eps - self.Vak) ** 2
            # Normalise the area
            Delta /= self.wd
            Delta *= 2
        else:
            # If eps is out of bounds there will
            # no Delta contribution
            Delta = 0.0
        return Delta

    def _create_Lambda_arb(self, eps) -> acb:
        """Create the hilbert transform of Delta with arb."""
        eps_ref = self.create_reference_eps(eps)

        if eps_ref.real < arb(-1):
            # Below the lower edge of the d-band
            Lambda = eps_ref + acb.pow(eps_ref**2 - acb(1), 0.5)
        elif eps_ref.real > arb(1):
            # Above the upper edge of the d-band
            Lambda = eps_ref - acb.pow(eps_ref**2 - acb(1), 0.5)
        elif acb.abs_lower(eps_ref) <= arb(1):
            # Inside the d-band
            Lambda = eps_ref
        else:
            raise ValueError(f"eps_ = {eps} cannot be considered in Lambda")

        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to pi Vak^2
        Lambda *= (self.Sak * eps - self.Vak) ** 2
        Lambda /= self.wd
        Lambda *= acb(2)
        return Lambda

    def _create_Lambda_reg(self, eps) -> float:
        """Create the hilbert transform of Delta for regular manipulations."""
        eps_ref = self.create_reference_eps(eps)

        if eps_ref < -1:
            # Below the lower edge of the d-band
            Lambda = eps_ref + (eps_ref**2 - 1) ** 0.5
        elif eps_ref > 1:
            # Above the upper edge of the d-band
            Lambda = eps_ref - (eps_ref**2 - 1) ** 0.5
        elif eps_ref <= 1:
            # Inside the d-band
            Lambda = eps_ref
        else:
            raise ValueError(f"eps_ = {eps} cannot be considered in Lambda")

        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to pi Vak^2
        Lambda *= (self.Sak * eps - self.Vak) ** 2
        Lambda /= self.wd
        Lambda *= 2
        return Lambda
