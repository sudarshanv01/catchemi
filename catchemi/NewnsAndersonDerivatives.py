"""Compute the derivatives of the Newns-Anderson model."""

import numpy as np
from flint import acb, arb, ctx
from catchemi import NewnsAndersonLinearRepulsion

class NewnsAndersonDerivativeEpsd(NewnsAndersonLinearRepulsion):
    """Class for computing the derivate of quantities 
    from the Newns-Anderson model with respect to the
    d-band centre. The code computes the derivate both
    from the analytical expression derived in the 
    manuscript as well as by numerically differentiating
    quantities on a grid. The latter option is useful for
    confirming that the derivative is computed as expected."""
    def __init__(self, Vsd, eps_a, width, eps, Delta0_mag=0.0,
                 eps_sp_max=15, eps_sp_min=-15,
                 precision=50, verbose=False,
                 alpha=0.0, beta=0.0, constant_offset=0.0,
                 diff_grid=np.linspace(-4,-1)):
        Vak = np.sqrt(beta) * Vsd
        # Since we are taking the derivate with respece
        # to the d-band centre, we need to set the d-band
        eps_d = None
        super().__init__(Vak, eps_a, eps_d, width, 
                         eps, Delta0_mag, eps_sp_max,
                         eps_sp_min, precision, verbose,
                         alpha, beta, constant_offset)
        self.diff_grid = diff_grid
    
    def get_Delta_prime_epsd(self, eps):
        """Get the derivative of Delta prime with epsd
        for the diff grid."""
        eps, = self._convert_to_acb(eps)
        Delta_prime_epsd = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            self.eps_d = eps_d
            Delta_prime_epsd[i] = self._calculate_Delta_prime_epsd(eps).real
        return Delta_prime_epsd
    
    def get_Lambda_prime_epsd(self, eps):
        """Get the derivative of Lambda with respect to epsd
        for the diff grid."""
        eps, = self._convert_to_acb(eps)
        Lambda_prime_epsd = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            self.eps_d = eps_d
            Lambda_prime_epsd[i] = self._calculate_Lambda_prime_epsd(eps).real
        return Lambda_prime_epsd
        
    def _calculate_Delta_prime_epsd(self, eps):
        """Compute the derivative of Delta with respect to 
        the d-band centre. This calculation will require
        multiprecision to compute the energy derivative
        later on."""
        eps_r = self.create_reference_eps(eps)
        if acb.abs_upper(eps_r) < arb('1.0'):
            # Within the d-band
            Delta_prime_epsd = acb('2.0') * acb.pow (self.Vak / self.wd, acb('2.0')) 
            Delta_prime_epsd *= eps_r 
            Delta_prime_epsd *= acb.pow(acb('1.0') - acb.pow(eps_r, acb('2.0')), acb('-0.5'))
        else:
            Delta_prime_epsd = acb('0.0')
        return Delta_prime_epsd
    
    def get_Delta_prime_epsd_numerical(self, eps):
        """Compute the derivate of Delta with respect to
        the d-band centre using numerical differentiation.
        Since this is a numerical calculation, we just take
        the derivative using standard floating point numbers."""
        self._convert_to_float()
        Delta_values = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            # Operting on the differential grid, find
            # the derivative of Delta 
            self.eps_d = eps_d
            Delta = self._create_Delta_reg(eps)
            Delta_values[i] = Delta
        # Take the gradient of Delta
        Delta_prime_epsd_numerical = np.gradient(Delta_values, self.diff_grid)
        return Delta_prime_epsd_numerical

    def _calculate_Lambda_prime_epsd(self, eps):
        """Compute the derivative of Lambda with respect
        to the d-band centre. The derivative is constructed
        based on the calculation type."""
        eps_r = self.create_reference_eps(eps)
        if acb.abs_lower(eps_r) <= arb('1.0'):
            # Within the d-band
            Lambda_prime = acb('-1.0')
        elif eps_r.real < arb('-1.0'):
            # Outside the d-band and at lower energies
            Lambda_prime = acb('-1.0')
            Lambda_prime -= eps_r * acb.pow(acb.pow(eps_r, acb('2.0')) - acb('1.'), acb('-0.5')) 
        elif eps_r.real > arb('1.0'): 
            # Outside the d-band and at higher energies
            Lambda_prime = acb('-1.0')
            Lambda_prime += eps_r * acb.pow(acb.pow(eps_r, acb('2.0')) - acb('1.'), acb('-0.5'))
        else:
            raise ValueError('eps_r is not in the right range.')

        prefactor = acb('2.0') * acb.pow(self.Vak / self.wd, acb('2.0'))
        Lambda_prime_epsd = prefactor * Lambda_prime
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
        integrand_numerator = Delta_prime(eps) * ( eps_function(eps) - Lambda(eps) ) 
        integrand_numerator += Delta0(eps) * Lambda_prime(eps)
        integrand_numerator += Delta(eps) * Lambda_prime(eps)
        integrand_denominator = ( eps_function(eps) - Lambda(eps) )**2 + ( Delta(eps) + Delta0(eps) )**2 

        return integrand_numerator / integrand_denominator


    def _calculate_hybridisation_energy_prime_epsd(self):
        """Calculate the hybridisation energy derivative with eps_d
        at a certain eps_d value."""

        # Numerically integrate the dos to find the occupancy
        dhyb_depsd = acb.integral(lambda x, _: self._create_dEhyb_deps(x), 
                                            self.eps_min, 
                                            arb('0.0'), 
                                            # rel_tol=np.power(2, -self.precision/2))
        )
        dhyb_depsd *= acb('2.0') / acb.pi()
        return dhyb_depsd
    
    def get_hybridisation_energy_prime_epsd(self):
        """Calculate the derivative of the hybridisation energy
        as a function of the d-band centre. if the flag get_hyb 
        is True, the hybridisation energy will also be returned."""
        self._convert_to_acb()
        hybridisation_energy_prime_epsd = np.zeros(len(self.diff_grid))

        for i, eps_d in enumerate(self.diff_grid):
            self.eps_d = eps_d
            dEhyb_depsd = self._calculate_hybridisation_energy_prime_epsd()
            hybridisation_energy_prime_epsd[i] = dEhyb_depsd.real 

        return hybridisation_energy_prime_epsd

    def get_hybridisation_energy_prime_epsd_numerical(self, get_hyb=False):
        """Calculate the derivative of the hybridisation energy 
        as a function of the d-band centre, but using the
        numerical differentiation grid."""
        self._convert_to_float()
        hybridisation_energy = np.zeros(len(self.diff_grid))
        for i, eps_d in enumerate(self.diff_grid):
            self.eps_d = eps_d
            self.calculate_hybridisation_energy()
            hybridisation_energy[i] = self.hybridisation_energy
        # Take the gradient of the hybridisation energy
        hybridisation_energy_prime_epsd_numerical = np.gradient(hybridisation_energy)
        if get_hyb: 
            return hybridisation_energy_prime_epsd_numerical, hybridisation_energy
        else:
            return hybridisation_energy_prime_epsd_numerical

        