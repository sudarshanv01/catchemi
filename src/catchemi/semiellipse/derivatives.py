"""Compute the derivatives of the Newns-Anderson model."""

import numpy as np
from flint import acb, arb, ctx
from scipy import integrate
from catchemi import NewnsAndersonLinearRepulsion, NewnsAndersonNumerical
from typing import Callable


class NewnsAndersonDerivativeEpsd(NewnsAndersonNumerical):
    def __init__(
        self,
        f_Vsd: Callable[[float], float],
        f_Vsd_p: Callable[[float], float],
        eps_a: float,
        f_wd: Callable[[float], float],
        f_wd_p: Callable[[float], float],
        eps: np.ndarray,
        Delta0_mag=0.0,
        eps_sp_max=15,
        eps_sp_min=-15,
        precision=50,
        verbose=False,
        alpha=0.0,
        beta=0.0,
        diff_grid=np.linspace(-4, -1),
        use_multiprec=False,
    ):
        """Class for computing the derivate of quantities
        from the Newns-Anderson model with respect to the
        d-band centre. The code computes the derivate both
        from the analytical expression derived in the
        manuscript as well as by numerically differentiating
        quantities on a grid. The latter option is useful for
        confirming that the derivative is computed as expected.

        Args:
            f_Vsd (Callable[[float], float]): Function for the
                Vsd parameter in the Newns-Anderson model.
            f_Vsd_p (Callable[[float], float]): Derivative of
                the function for the Vsd parameter in the
                Newns-Anderson model.
            eps_a (float): The adsorbate energy.
            f_wd (Callable[[float], float]): Function for the
                wd parameter in the Newns-Anderson model.
            f_wd_p (Callable[[float], float]): Derivative of
                the function for the wd parameter in the
                Newns-Anderson model.
            eps (np.ndarray): The energy grid.
            Delta0_mag (float, optional): The magnitude of the
                hybridisation energy. Defaults to 0.0.
            eps_sp_max (int, optional): The maximum value of
                the energy grid for the s and p bands.
                Defaults to 15.
            eps_sp_min (int, optional): The minimum value of
                the energy grid for the s and p bands.
                Defaults to -15.
            precision (int, optional): The precision to use
                for the multiprecision calculations.
                Defaults to 50.
            verbose (bool, optional): Whether to print
                information about the calculations.
                Defaults to False.
            alpha (float, optional): The alpha parameter
                in the Newns-Anderson model. Defaults to 0.0.
            beta (float, optional): The beta parameter
                in the Newns-Anderson model. Defaults to 0.0.
            diff_grid (np.ndarray, optional): The grid on
                which to compute the numerical derivative.
                Defaults to np.linspace(-4, -1).
            use_multiprec (bool, optional): Whether to use
                multiprecision for the calculations.
                Defaults to False.
        """

        # Since we are taking the derivate with respece
        # to the d-band centre, we need to set the d-band
        eps_d = None

        # We also do not need to set quantities such as
        # Vak and wd as those will be determined by the
        # d-band centre.
        wd = None
        Vak = None

        # Store the functions for the Vsd and wd and
        # their respective derivatives (_p)
        self.f_Vsd = f_Vsd
        self.f_Vsd_p = f_Vsd_p
        self.f_wd = f_wd
        self.f_wd_p = f_wd_p

        # If the calculation needs to use multiprecision
        self.use_multiprec = use_multiprec
        if self.use_multiprec:
            raise NotImplementedError("Multiprecision not implemented yet.")

        # For this class, currently only Delta0 > 0 is supported
        if Delta0_mag <= 0.0:
            raise NotImplementedError("Delta0 must be positive.")

        # Store quantities related to the LinearRepulsion treatment
        self.alpha = alpha
        self.beta = beta

        super().__init__(
            Vak,
            eps_a,
            eps_d,
            wd,
            eps,
            Delta0_mag,
            eps_sp_max,
            eps_sp_min,
            precision,
            verbose,
        )

        # The grid on which the numerical derivative is computed
        # and the analytical derivative is reported
        self.diff_grid = diff_grid

    def get_Delta_prime_epsd(self, eps):
        """Get the derivative of Delta prime with epsd
        for the diff grid."""
        if self.use_multiprec:
            (eps,) = self._convert_to_acb(eps)
        else:
            (eps,) = self._convert_to_float(eps)

        Delta_prime_epsd = np.zeros(len(self.diff_grid))

        for i, eps_d in enumerate(self.diff_grid):
            if self.use_multiprec:
                self.eps_d = acb(str(eps_d))
                Delta_prime_epsd[i] = self._calculate_Delta_prime_epsd(eps).real
            else:
                self.eps_d = eps_d
                # Generate the current Vak and wd
                self._generate_current_Vak_wd()

                Delta_prime_epsd[i] = self._calculate_Delta_prime_epsd_reg(eps)

        return Delta_prime_epsd

    def get_Lambda_prime_epsd(self, eps):
        """Get the derivative of Lambda with respect to epsd
        for the diff grid."""
        if self.use_multiprec:
            (eps,) = self._convert_to_acb(eps)
        else:
            (eps,) = self._convert_to_float(eps)
        Lambda_prime_epsd = np.zeros(len(self.diff_grid))

        for i, eps_d in enumerate(self.diff_grid):
            if self.use_multiprec:
                self.eps_d = acb(str(eps_d))
                Lambda_prime_epsd[i] = self._calculate_Lambda_prime_epsd(eps).real
            else:
                self.eps_d = eps_d
                # Generate the current Vak and wd
                self._generate_current_Vak_wd()

                Lambda_prime_epsd[i] = self._calculate_Lambda_prime_epsd_reg(eps)
        return Lambda_prime_epsd

    def _calculate_Delta_prime_epsd(self, eps: acb) -> acb:
        """Compute the derivative of Delta with respect to
        the d-band centre. This calculation will require
        multiprecision to compute the energy derivative
        later on."""
        eps_r = self.create_reference_eps(eps)
        if acb.abs_upper(eps_r) < arb("1.0"):
            # Within the d-band
            Delta_prime_epsd = acb("2.0") * acb.pow(self.Vak / self.wd, acb("2.0"))
            Delta_prime_epsd *= eps_r
            Delta_prime_epsd *= acb.pow(
                acb("1.0") - acb.pow(eps_r, acb("2.0")), acb("-0.5")
            )
        else:
            Delta_prime_epsd = acb("0.0")
        return Delta_prime_epsd

    def _generate_current_Vak_wd(self):
        """Utility to generate the current Vak and wd
        based on the value of eps_d stored in self.eps_d."""
        Vak = np.sqrt(self.beta) * self.f_Vsd(self.eps_d)
        Vak_p = np.sqrt(self.beta) * self.f_Vsd_p(self.eps_d)
        wd = self.f_wd(self.eps_d)
        wd_p = self.f_wd_p(self.eps_d)

        assert Vak >= 0.0, "Coupling element must be positive."
        assert wd >= 0.0, "Width must be positive."

        self.wd = wd
        self.wd_p = wd_p
        self.Vak = Vak
        self.Vak_p = Vak_p

        return Vak, Vak_p, wd, wd_p

    def _calculate_Delta_prime_epsd_reg(self, eps: float) -> float:
        """Compute the derivative of Delta with respect to
        the d-band centre. This calculation does not require
        multiprecision."""

        # Generate the current relative eps_d
        eps_r = self.create_reference_eps(eps)

        if np.abs(eps_r) < 1.0:
            # Within the d-band is the only place
            # where the Delta value is not zero
            Delta_prime_epsd_1 = 2.0 * self.Vak**2 / self.wd
            Delta_prime_epsd_1 /= np.sqrt(1.0 - eps_r**2)
            Delta_prime_epsd_1 *= eps_r
            Delta_prime_epsd_1 *= 1 / self.wd + self.wd_p / self.wd * eps_r

            Delta_prime_epsd_2 = 2 * np.sqrt(1.0 - eps_r**2)
            Delta_prime_epsd_2 *= (
                2 * self.Vak * self.Vak_p / self.wd
                - self.wd_p * self.Vak**2 / self.wd**2
            )
            Delta_prime_epsd = Delta_prime_epsd_1 + Delta_prime_epsd_2
        else:
            Delta_prime_epsd = 0.0

        return Delta_prime_epsd

    def get_Delta_prime_epsd_numerical(self, eps):
        """Compute the derivate of Delta with respect to
        the d-band centre using numerical differentiation.
        Since this is a numerical calculation, we just take
        the derivative using standard floating point numbers."""
        self._convert_to_float()
        # Get quantities that are needed for the numerical derivative
        Delta_values = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            # Operting on the differential grid, find
            # the derivative of Delta
            self.eps_d = eps_d
            self._generate_current_Vak_wd()

            Delta = self._create_Delta_reg(eps)
            Delta_values[i] = Delta

        # Take the gradient of Delta
        Delta_prime_epsd_numerical = np.gradient(Delta_values, self.diff_grid)
        return Delta_prime_epsd_numerical

    def _calculate_Lambda_prime_epsd(self, eps: acb) -> acb:
        """Compute the derivative of Lambda with respect
        to the d-band centre."""
        eps_r = self.create_reference_eps(eps)
        if acb.abs_lower(eps_r) <= arb("1.0"):
            # Within the d-band
            Lambda_prime = acb("-1.0")
        elif eps_r.real < arb("-1.0"):
            # Outside the d-band and at lower energies
            Lambda_prime = acb("-1.0")
            Lambda_prime -= eps_r * acb.pow(
                acb.pow(eps_r, acb("2.0")) - acb("1."), acb("-0.5")
            )
        elif eps_r.real > arb("1.0"):
            # Outside the d-band and at higher energies
            Lambda_prime = acb("-1.0")
            Lambda_prime += eps_r * acb.pow(
                acb.pow(eps_r, acb("2.0")) - acb("1."), acb("-0.5")
            )
        else:
            raise ValueError("eps_r is not in the right range.")

        prefactor = acb("2.0") * acb.pow(self.Vak / self.wd, acb("2.0"))
        Lambda_prime_epsd = prefactor * Lambda_prime
        return Lambda_prime_epsd

    def _calculate_Lambda_prime_epsd_reg(self, eps: float) -> float:
        """Compute the derivative of Lambda with respect
        to the d-band centre. This calculation does not
        require multiprecision."""

        # Generate the current relative eps_d
        eps_r = self.create_reference_eps(eps)

        # Generate some utility functions to help
        # with the computation of the derivative
        def _get_derivative_1():
            """Compute the derivative of the first term
            2Vak^2/wd * (eps - eps_d / wd)"""
            D1 = 2.0 * self.Vak**2 / self.wd**2 * (-1.0 - self.wd_p * eps_r)
            D1 += (
                2.0
                * eps_r
                / self.wd
                * (2.0 * self.Vak * self.Vak_p - self.wd_p / self.wd * self.Vak**2)
            )
            return D1

        def _get_derivative_2():
            """Compute the derivative of the second term
            2Vak^2/wd * ( (eps-eps_d / wd)^2 - 1 )^0.5"""
            D2_1 = 2 * (self.Vak / self.wd) ** 2
            D2_1 /= np.sqrt(eps_r**2 - 1)
            D2_1 *= eps_r
            D2_1 *= -1 - self.wd_p * eps_r
            D2_2 = 2 * np.sqrt(eps_r**2 - 1)
            D2_2 /= self.wd
            D2_2 *= 2 * self.Vak * self.Vak_p - self.wd_p / self.wd * self.Vak**2
            return D2_1 + D2_2

        if np.abs(eps_r) <= 1.0:
            # Within the d-band
            Lambda_prime_epsd = _get_derivative_1()
        elif eps_r < -1.0:
            # Outside the d-band and at lower energies
            Lambda_prime_epsd = _get_derivative_1() + _get_derivative_2()
        elif eps_r > 1.0:
            # Outside the d-band and at higher energies
            Lambda_prime_epsd = _get_derivative_1() - _get_derivative_2()
        else:
            raise ValueError("eps_r is not in the right range.")

        return Lambda_prime_epsd

    def get_Lambda_prime_epsd_numerical(self, eps):
        """Compute the derivative of Lambda with respect
        to the d-band centre."""
        self._convert_to_float()
        Lambda_values = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            # Operting on the differential grid, find
            # the derivative of Lambda
            self.eps_d = eps_d
            self._generate_current_Vak_wd()

            Lambda = self._create_Lambda_reg(eps)
            Lambda_values[i] = Lambda
        # Take the gradient of Lambda
        Lambda_prime_epsd_numerical = np.gradient(Lambda_values, self.diff_grid)
        return Lambda_prime_epsd_numerical

    def _create_dEhyb_deps(self, eps):
        """Create the integrand for the derivative of the
        hybridisation energy with the d-band centre."""
        # Derivative functions
        Delta_prime = self._calculate_Delta_prime_epsd
        Lambda_prime = self._calculate_Lambda_prime_epsd

        # Functions from the Newns Anderson model
        eps_function = self._create_adsorbate_line
        Delta = self._create_Delta_arb
        Delta0 = self._create_Delta0_arb
        Lambda = self._create_Lambda_arb

        # Create the integrand
        integrand_numerator = Delta_prime(eps) * (eps_function(eps) - Lambda(eps))
        integrand_numerator += (Delta0(eps) + Delta(eps)) * Lambda_prime(eps)
        integrand_denominator = acb.pow(eps_function(eps) - Lambda(eps), acb("2.0"))
        integrand_denominator += acb.pow(Delta(eps) + Delta0(eps), acb("2.0"))

        return integrand_numerator / integrand_denominator

    def _create_dEhyb_deps_reg(self, eps):
        """Create the integrand for the derivative of the
        hybridisation energy with the d-band centre. This
        calculation does not require multiprecision."""
        # Derivative functions
        Delta_prime = self._calculate_Delta_prime_epsd_reg
        Lambda_prime = self._calculate_Lambda_prime_epsd_reg

        # Functions from the Newns Anderson model
        eps_function = self._create_adsorbate_line
        Delta = self._create_Delta_reg
        Delta0 = self._create_Delta0_reg
        Lambda = self._create_Lambda_reg

        # Create the integrand
        integrand_numerator = Delta_prime(eps) * (eps_function(eps) - Lambda(eps))
        integrand_numerator += (Delta0(eps) + Delta(eps)) * Lambda_prime(eps)
        integrand_denominator = (eps_function(eps) - Lambda(eps)) ** 2
        integrand_denominator += (Delta(eps) + Delta0(eps)) ** 2

        assert integrand_denominator > 0.0

        return integrand_numerator / integrand_denominator

    def _calculate_hybridisation_energy_prime_epsd(self):
        """Calculate the hybridisation energy derivative with eps_d
        at a certain eps_d value."""

        # Numerically integrate the dos to find the occupancy
        if self.use_multiprec:
            dhyb_depsd = acb.integral(
                lambda x, _: self._create_dEhyb_deps(x),
                self.eps_min,
                arb("0.0"),
            )
            dhyb_depsd *= acb("2.0") / acb.pi()
            return dhyb_depsd.real
        else:
            dhyb_depsd = integrate.quad(
                lambda x: self._create_dEhyb_deps_reg(x), self.eps_min, 0.0, limit=100
            )[0]
            dhyb_depsd *= 2.0 / np.pi
            return dhyb_depsd

    def get_hybridisation_energy_prime_epsd(self):
        """Calculate the derivative of the hybridisation energy
        as a function of the d-band centre. if the flag get_hyb
        is True, the hybridisation energy will also be returned."""

        if self.use_multiprec:
            self._convert_to_acb()
        else:
            self._convert_to_float()

        hybridisation_energy_prime_epsd = np.zeros(len(self.diff_grid))

        for i, eps_d in enumerate(self.diff_grid):
            # Treat eps_d correctly based on the
            # chosen numerical routine
            if self.use_multiprec:
                self.eps_d = acb(str(eps_d))
            else:
                self.eps_d = eps_d
            # Regenerate Vak and wd
            self._generate_current_Vak_wd()

            dEhyb_depsd = self._calculate_hybridisation_energy_prime_epsd()

            hybridisation_energy_prime_epsd[i] = dEhyb_depsd

        return hybridisation_energy_prime_epsd

    def get_hybridisation_energy_prime_epsd_numerical(self, get_hyb=False):
        """Calculate the derivative of the hybridisation energy
        as a function of the d-band centre, but using the
        numerical differentiation grid."""
        self._convert_to_float()
        hybridisation_energy = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            self.eps_d = eps_d
            # Regenerate Vak and wd
            self._generate_current_Vak_wd()
            self.calculate_hybridisation_energy()
            hybridisation_energy[i] = self.hybridisation_energy

        # Take the gradient of the hybridisation energy
        hybridisation_energy_prime_epsd_numerical = np.gradient(
            hybridisation_energy, self.diff_grid
        )
        if get_hyb:
            return hybridisation_energy_prime_epsd_numerical, hybridisation_energy
        else:
            return hybridisation_energy_prime_epsd_numerical
