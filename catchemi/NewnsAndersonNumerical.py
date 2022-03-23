""" Perform the Newns-Anderson model calculations."""

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
class NewnsAndersonNumerical:
    """Perform numerical calculations of the Newns-Anderson model to get 
    the chemisorption energy."""

    Vak: float
    eps_a: float
    eps_d: float
    width: float
    eps: list 
    Delta0_mag: float
    eps_sp_max: float = 15
    eps_sp_min: float = -15
    precision: int = 50
    verbose: bool = False
    spin: float = 2
    NUMERICAL_NOISE_THRESHOLD = 1e-2

    def __post_init__(self):
        """Perform numerical calculations of the Newns-Anderson model 
        to get the chemisorption energy."""
        self.eps_min = np.min(self.eps)
        self.eps_max = np.max(self.eps) 
        self.wd = self.width
        self.eps = np.array(self.eps)

        if self.verbose:
            print(f'Solving the Newns-Anderson model for eps_a = {self.eps_a:1.2f} eV',
                  f'and eps_d = {self.eps_d:1.2f} and w_d = {self.width:1.2f}')

        # Choose the precision to which the quantities are calculated
        ctx.dps = self.precision

        # The outputs from the model are the hybridisation
        # energy and the occupancy of the single particle state
        self.hybridisation_energy = None
        self.occupancy = None
        
        # Everything start as a float
        self.calctype = 'float'
        
    def _convert_to_acb(self, *args) -> None:
        """Convert the important quantities to arb so 
        that they can be manipulated freely."""
        # Convert all quantities that are args to acb
        convert_to_acb = lambda x: acb(str(x))
        args = [convert_to_acb(arg) for arg in args]
        if self.calctype == 'multiprecision':
            # Everything is already a multiprecision number
            return args 
        self.eps_min = convert_to_acb(self.eps_min)
        self.eps_max = convert_to_acb(self.eps_max)
        self.eps_sp_min = convert_to_acb(self.eps_sp_min)
        self.eps_sp_max = convert_to_acb(self.eps_sp_max)
        self.wd = convert_to_acb(self.wd)
        self.eps_d = convert_to_acb(self.eps_d)
        self.eps_a = convert_to_acb(self.eps_a)
        self.Vak = convert_to_acb(self.Vak)
        self.calctype = 'multiprecision'
        return args
    
    def _convert_to_float(self, *args) -> None:
        """Convert the important quantities to a float."""
        # Convert all quantities that args to float
        args = [float(arg) for arg in args]
        if self.calctype == 'float':
            # everything is already a float
            return args
        self.eps_min = float(self.eps_min.real)
        self.eps_max = float(self.eps_max.real)
        self.eps_sp_min = float(self.eps_sp_min.real)
        self.eps_sp_max = float(self.eps_sp_max.real)
        self.wd = float(self.wd.real)
        self.eps_d = float(self.eps_d.real)
        self.eps_a = float(self.eps_a.real)
        self.Vak = float(self.Vak.real)
        self.calctype = 'float'
        return args

    def get_hybridisation_energy(self) -> float:
        """Get the hybridisation energy."""
        if self.hybridisation_energy is None:
            self.calculate_hybridisation_energy()
        return self.hybridisation_energy

    def get_occupancy(self) -> float:
        """Get the occupancy of the single particle state."""
        if self.occupancy is None:
            self.calculate_occupancy()
        return float(self.occupancy.real)
    
    def get_dos_on_grid(self) -> np.ndarray:
        """Get the density of states."""
        eps_function = self._create_adsorbate_line
        Delta = self._create_Delta_reg
        Delta0 = self._create_Delta0_reg
        Lambda = self._create_Lambda_reg
        self._convert_to_float()
        dos = np.zeros(len(self.eps))
        for i, eps_ in enumerate(self.eps):
            numerator = Delta(eps_) + Delta0(eps_) 
            denominator = ( eps_function(eps_) - Lambda(eps_) )**2 + ( Delta(eps_) + Delta0(eps_) ) **2 
            dos[i] = numerator / denominator / np.pi
        return dos

    def get_Delta_on_grid(self) -> np.ndarray:
        """Get Delta on supplied grid."""
        Delta = self._create_Delta_reg
        Delta0 = self._create_Delta0_reg
        self._convert_to_float()
        Delta_val = np.array([Delta(e) for e in self.eps])
        Delta_val += np.array([Delta0(e) for e in self.eps])
        return Delta_val 
    
    def get_Lambda_on_grid(self) -> np.ndarray:
        """Get Lambda on supplied grid."""
        Lambda = self._create_Lambda_reg
        self._convert_to_float()
        Lambda_val = np.array([Lambda(e) for e in self.eps])
        return Lambda_val 
    
    def get_energy_diff_on_grid(self) -> np.ndarray:
        """Get the line eps - eps_a on a grid."""
        eps_function = self._create_adsorbate_line
        self._convert_to_float()
        return eps_function(self.eps)

    def get_dband_filling(self):
        """Get the filling of the d-band."""
        self._calculate_filling()
        return self.filling.real 

    def create_reference_eps(self, eps):
        """Create the reference energy for finding Delta and Lambda."""
        return ( eps - self.eps_d ) / self.wd 

    def _create_Delta0_arb(self, eps) -> acb:
        """Create a function for Delta0 based on arb."""
        # The function creates Delta0 is if it between eps_sp max 
        # and eps_sp min, otherwise it is zero.
        if eps.real > self.eps_sp_min.real and eps.real < self.eps_sp_max.real:
            return self.Delta0_mag
        else:
            return acb('0.0')
    
    def _create_Delta0_reg(self, eps) -> float:
        """Create a function for Delta0 based on regular python."""
        if eps > self.eps_sp_min and eps < self.eps_sp_max:
            return self.Delta0_mag
        else:
            return 0.0

    def _create_Delta_arb(self, eps) -> acb:
        """Create a function for Delta based on arb."""
        eps_ref = self.create_reference_eps(eps) 
        # If the absolute value of the reference is 
        # lower than 1 (in the units of wd) then
        # the Delta will be non-zero 
        if acb.abs_lower(eps_ref) < arb('1.0'): 
            Delta = acb(1.)  -  acb.pow(eps_ref, 2)
            Delta = acb.pow(Delta, 0.5)
            # Multiply by the prefactor
            Delta *= self.Vak**2
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
        eps_ref = self.create_reference_eps(eps) 
        # If the absolute value of the reference is 
        # lower than 1 (in the units of wd) then
        # the Delta will be non-zero 
        if np.abs(eps_ref) < 1: 
            Delta = 1.  -  eps_ref**2
            Delta = Delta**0.5 
            # Multiply by the prefactor
            Delta *= self.Vak**2
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
            Lambda = eps_ref + acb.pow( eps_ref**2 - acb(1), 0.5 )
        elif eps_ref.real > arb(1):
            # Above the upper edge of the d-band
            Lambda = eps_ref - acb.pow( eps_ref**2 - acb(1), 0.5 )
        elif acb.abs_lower(eps_ref) <= arb(1): 
            # Inside the d-band
            Lambda = eps_ref
        else:
            raise ValueError(f'eps_ = {eps} cannot be considered in Lambda')

        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to pi Vak^2
        Lambda *= self.Vak**2
        Lambda /= self.wd
        Lambda *= acb(2)
        return Lambda

    def _create_Lambda_reg(self, eps) -> float:
        """Create the hilbert transform of Delta for regular manipulations."""
        eps_ref = self.create_reference_eps(eps)

        if eps_ref < -1: 
            # Below the lower edge of the d-band
            Lambda = eps_ref + ( eps_ref**2 - 1 )**0.5
        elif eps_ref > 1:
            # Above the upper edge of the d-band
            Lambda = eps_ref - ( eps_ref**2 - 1 )**0.5
        elif eps_ref <= 1: 
            # Inside the d-band
            Lambda = eps_ref
        else:
            raise ValueError(f'eps_ = {eps} cannot be considered in Lambda')

        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to pi Vak^2
        Lambda *= self.Vak**2 
        Lambda /= self.wd
        Lambda *= 2
        return Lambda

    def _create_Lambda_prime_arb(self, eps) -> acb:
        """Create the derivative of the hilbert transform of Lambda with arb."""
        eps_ref = self.create_reference_eps(eps)
        if eps_ref.real < arb(-1):
            # Below the lower edge of the d-band
            Lambda_prime = acb(1) + eps_ref * ( eps_ref**2 - acb(1) )**-0.5
        elif eps_ref.real > arb(1):
            # Above the upper edge of the d-band
            Lambda_prime = acb(1) - eps_ref * ( eps_ref**2 - acb(1) )**-0.5
        elif acb.abs_lower(eps_ref) <= arb(1):
            # Inside the d-band
            Lambda_prime = acb(1)

        Lambda_prime *= self.Vak**2
        Lambda_prime *= acb(2)
        Lambda_prime /= self.wd**2
        return Lambda_prime

    def _create_adsorbate_line(self, eps):
        """Create the line that the adsorbate passes through."""
        return eps - self.eps_a

    def find_poles_green_function(self) -> list:
        """Find the poles of the green function. In the case that Delta=0
        these points will not be the poles, but are important to pass 
        on to the integrator anyhow."""
        self.poles = []

        # In case Delta0 is not-zero, there will be no poles
        if self.Delta0_mag > 0:
            self.poles.append([False, False, False])
            return self.poles

        # If Delta0 is zero, there will be three possible
        # poles, one on the left and one on the right and 
        # one within Delta. 
        # determine where they are based on the intersection
        # between eps - eps_a - Lambda = 0
        eps_function = self._create_adsorbate_line
        Lambda = self._create_Lambda_reg

        # Find the epsilon value as which the Lambda and eps_function are equal
        # These poles will have to be explicitly mentioned during the integration
        # There are three possible regions where there might be a root
        # 1. Region above eps_d + wd
        # 2. Region between eps_d - wd and eps_d + wd
        # 3. Region below eps_d - wd
        # If the optimizer excepts, then there is no intersection 
        # within that sub-region

        # Find the poles in the energy region that is below the d-band
        try:
            pole_lower = optimize.brentq(lambda x: eps_function(x) - Lambda(x), 
                                         self.eps_min,
                                         self.eps_d - self.wd,)
            self.poles.append(pole_lower)
        except ValueError:
            self.poles.append(None)

        # Find the pole in the middle of the d-band, if any
        try:
            pole_middle = optimize.brentq(lambda x: eps_function(x) - Lambda(x),
                                          self.eps_d - self.wd,
                                          self.eps_d + self.wd)
            self.poles.append(pole_middle)
        except ValueError:
            self.poles.append(None)

        # Find the poles in the energy region that is above the d-band
        try:
            pole_higher = optimize.brentq(lambda x: eps_function(x) - Lambda(x),
                                          self.eps_d + self.wd,
                                          self.eps_max )
            self.poles.append(pole_higher)
        except ValueError:
            self.poles.append(None)

        if self.verbose:
            print(f'Poles of the green function:{self.poles}')
        
        return self.poles

    def _create_dos(self, eps) -> acb:
        """Create the density of states."""
        eps_function = self._create_adsorbate_line
        Delta = self._create_Delta_arb
        Delta0 = self._create_Delta0_arb
        Lambda = self._create_Lambda_arb
        # Create the density of states
        numerator = Delta(eps) + Delta0(eps) 
        denominator = ( eps_function(eps) - Lambda(eps) )**2 + ( Delta(eps) + Delta0(eps) )**2 
        return numerator / denominator / acb.pi()

    def _calculate_filling(self) -> float:
        """Calculate the filling from the metal density of states."""
        self._convert_to_float()
        # Filling contribution coming from the d-states
        filling_numerator = integrate.quad(self._create_Delta_reg, 
                            self.eps_min, 0,
                            limit=100)[0]
        # Filling contribution coming from the sp-states
        filling_numerator += integrate.quad(self._create_Delta0_reg,
                             self.eps_min, 0,
                             limit=100)[0]
        # Filling contribution to the denomitor coming from the d-states 
        filling_denominator = integrate.quad(self._create_Delta_reg,
                            self.eps_min, self.eps_max,
                            limit=100)[0]
        # Filling contribution to the denomitor coming from the sp-states
        filling_denominator += integrate.quad(self._create_Delta0_reg,
                                self.eps_min, self.eps_max,
                                limit=100)[0]
        self.filling = filling_numerator / filling_denominator
        return filling_numerator / filling_denominator

    def calculate_occupancy(self):
        """Calculate the density of states from the Newns-Anderson model."""
        # If a dos is required, then switch to arb
        if self.Delta0_mag == 0:
            # Determine the points of the singularity
            self.find_poles_green_function()
            self._convert_to_acb()

            localised_occupancy = acb('0.0')
            for pole in self.poles:
                if pole is not None:
                    # The pole has to exist 
                    if pole.real < 0: 
                        # The pole has to be below the Fermi 
                        # level to be counted in the occupancy
                            if pole.real < self.eps_d.real - self.wd.real \
                               or pole.real > self.eps_d.real + self.wd.real:
                                # Consider for now only the localised
                                # states, that is, the poles that
                                # are the energy where Delta(eps) = 0
                                Lambda_prime = self._create_Lambda_prime_arb(pole)
                                assert Lambda_prime.real <= arb(0.0)
                                localised_occupancy += acb('1.0') / (acb('1.0') - Lambda_prime)

            # # Add in the integral for the states within the Delta function
            lower_integration_bound = min(0.0, float((self.eps_d - self.wd).real) )
            upper_integration_bound = min(0.0, float((self.eps_d + self.wd).real) )
            self.occupancy = acb.integral(lambda x, _: self._create_dos(x),
                                                lower_integration_bound,
                                                upper_integration_bound)
            self.occupancy += localised_occupancy

        else:
            self._convert_to_acb()
            # Numerically integrate the dos to find the occupancy
            self.occupancy = acb.integral(lambda x, _: self._create_dos(x), 
                                                self.eps_min, 
                                                arb('0.0'), 
                                                rel_tol=np.power(2, -self.precision/2))
        if self.verbose:
            print(f'Single particle occupancy: {self.occupancy}')

    def _create_energy_integrand(self, eps):
        """Create the energy integrand of the system."""
        eps_function = self._create_adsorbate_line
        Delta = self._create_Delta_reg
        Delta0 = self._create_Delta0_reg
        Lambda = self._create_Lambda_reg

        # Determine the energy of the system
        numerator = Delta(eps) + Delta0(eps) 
        denominator = eps_function(eps) - Lambda(eps)

        # find where self.eps is lower than 0
        if eps > 0:
            return 0 
        else:
            arctan_integrand = np.arctan2(numerator, denominator)
            arctan_integrand -= np.pi
            assert arctan_integrand <= 0, "Arctan integrand must be negative"
            assert arctan_integrand >= -np.pi, "Arctan integrand must be greater than -pi"
            return arctan_integrand
    
    def calculate_hybridisation_energy(self):
        """Calculate the energy from the Newns-Anderson model."""

        # We do not need multi-precision for this calculation
        self._convert_to_float()
        self.find_poles_green_function()

        poles_to_consider = [pole for pole in self.poles if pole is not None]
        delta_E_ = integrate.quad(self._create_energy_integrand, 
                            self.eps_min, 0,
                            points = tuple(poles_to_consider),
                            limit=100)[0]

        self.hybridisation_energy = delta_E_ * self.spin / np.pi 
        self.hybridisation_energy -= self.spin * self.eps_a

        # Check if DeltaE is positive and within the NUMERICAL_NOISE_THRESHOLD
        if self.hybridisation_energy > 0 and self.hybridisation_energy < self.NUMERICAL_NOISE_THRESHOLD:
            self.hybridisation_energy = 0

        if self.verbose:
            print(f'Energy of the system: {self.hybridisation_energy} eV')