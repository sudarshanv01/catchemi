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
class NorskovNewnsAnderson:
    """Class for fitting parameters for the Newns-Anderson and the
    d-band model with calculations performed with DFT. The class 
    expects all metal quantities for a particular eps_a (adsorbate) value.
    
    Strategy
    --------
    Fit the energies to the d-band centre based on the fitting 
    expression including the newns-anderson model and the orthogonalisation
    energy. The fitting is done by minimizing the sum of squares of the
    energies.
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

@dataclass
class NewnsAndersonNumerical:
    """Perform numerical calculations of the Newns-Anderson model to get 
    the chemisorption energy.
    Vak: float
        Coupling matrix element
    eps_a: float
        Energy of the adsorbate
    eps_d: float
        Energy of the d-band
    eps: list
        Energy range to consider
    width: float
        Width of the d-band
    k: float
        Parameter to control the amount of added extra states
    
    Strategy:
    --------
    The occupancy is calculated using the arb package. This choice
    provides the needed precision for the calculation, as well as an
    estimate of the rounding errors, which are useful for the fitting
    procedure later on. In case Delta0=0, then there is a pole to the 
    dos function, and that is taken care of by using the analytical
    derivative of the Hilbert transform of Delta.
    The hybridisation energy has less strict requirements on its precision,
    and hence is solved using scipy libraries.
    """

    Vak: float
    eps_a: float
    width: float
    eps_d: float
    eps: list 
    Delta0_mag: float
    eps_sp_max: float = 15
    eps_sp_min: float = -15
    precision: int = 50
    verbose: bool = False
    NUMERICAL_NOISE_THRESHOLD = 1e-2

    def __post_init__(self):
        """Perform numerical calculations of the Newns-Anderson model 
        to get the chemisorption energy."""
        self.eps_min = np.min(self.eps)
        self.eps_max = np.max(self.eps) 
        self.wd = self.width / 2
        self.eps = np.array(self.eps)
        if self.verbose:
            print(f'Solving the Newns-Anderson model for eps_a = {self.eps_a:1.2f} eV and eps_d = {self.eps_d:1.2f} and a width = {self.width:1.2f}')

        # Choose the precision to which the quantities are calculated
        ctx.dps = self.precision
        
    def _convert_to_acb(self):
        """Convert the important quantities to arb so 
        that they can be manipulated freely."""
        convert_to_acb = lambda x: acb(str(x))
        self.eps_min = convert_to_acb(self.eps_min)
        self.eps_max = convert_to_acb(self.eps_max)
        self.eps_sp_min = convert_to_acb(self.eps_sp_min)
        self.eps_sp_max = convert_to_acb(self.eps_sp_max)
        self.wd = convert_to_acb(self.wd)
        self.eps_d = convert_to_acb(self.eps_d)
        self.eps_a = convert_to_acb(self.eps_a)
        self.Vak = convert_to_acb(self.Vak)
    
    def _convert_to_float(self):
        """Convert the important quantities to a float."""
        self.eps_min = float(self.eps_min.real)
        self.eps_max = float(self.eps_max.real)
        self.eps_sp_min = float(self.eps_sp_min.real)
        self.eps_sp_max = float(self.eps_sp_max.real)
        self.wd = float(self.wd.real)
        self.eps_d = float(self.eps_d.real)
        self.eps_a = float(self.eps_a.real)
        self.Vak = float(self.Vak.real)

    def get_energy(self) -> float:
        """Get the hybridisation energy."""
        return self.DeltaE

    def get_occupancy(self) -> float:
        """Get the occupancy of the single particle state."""
        return float(self.na.real)
    
    def get_dos_on_grid(self) -> np.ndarray:
        """Get the density of states."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_reg
        Delta0 = self.create_Delta0_reg
        Lambda = self.create_Lambda_reg
        self._convert_to_float()
        dos = np.zeros(len(self.eps))
        for i, eps_ in enumerate(self.eps):
            numerator = Delta(eps_) + Delta0(eps_) 
            denominator = ( eps_function(eps_) - Lambda(eps_) )**2 + ( Delta(eps_) + Delta0(eps_) ) **2 
            dos[i] = numerator / denominator / np.pi
        return dos

    def get_Delta_on_grid(self) -> np.ndarray:
        """Get Delta on supplied grid."""
        Delta = self.create_Delta_reg
        Delta0 = self.create_Delta0_reg
        self._convert_to_float()
        Delta_val = np.array([Delta(e) for e in self.eps])
        Delta_val += np.array([Delta0(e) for e in self.eps])
        return Delta_val 
    
    def get_Lambda_on_grid(self):
        """Get Lambda on supplied grid."""
        Lambda = self.create_Lambda_reg
        self._convert_to_float()
        Lambda_val = np.array([Lambda(e) for e in self.eps])
        return Lambda_val 
    
    def get_energy_diff_on_grid(self):
        """Get the line eps - eps_a on a grid."""
        eps_function = self.create_adsorbate_line
        self._convert_to_float()
        return eps_function(self.eps)

    def get_dband_filling(self):
        return self.calculate_filling()

    def create_reference_eps(self, eps):
        """Create the reference energy for finding Delta and Lambda."""
        return ( eps - self.eps_d ) / self.wd 

    def create_Delta0_arb(self, eps) -> acb:
        """Create a function for Delta0 based on arb."""
        # The function creates Delta0 is if it between eps_sp max 
        # and eps_sp min, otherwise it is zero.
        if eps.real > self.eps_sp_min.real and eps.real < self.eps_sp_max.real:
            return self.Delta0_mag
        else:
            return acb('0.0')
    
    def create_Delta0_reg(self, eps) -> float:
        """Create a function for Delta0 based on regular python."""
        if eps > self.eps_sp_min and eps < self.eps_sp_max:
            return self.Delta0_mag
        else:
            return 0.0

    def create_Delta_arb(self, eps) -> acb:
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

    def create_Delta_reg(self, eps) -> float:
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

    def create_Lambda_arb(self, eps) -> acb:
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

    def create_Lambda_reg(self, eps) -> float:
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

    def create_Lambda_prime_arb(self, eps) -> acb:
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

    def create_adsorbate_line(self, eps):
        """Create the line that the adsorbate passes through."""
        return eps - self.eps_a

    def find_poles_green_function(self):
        """Find the poles of the green function. In the case that Delta=0
        these points will not be the poles, but are important to pass 
        on to the integrator anyhow."""
        self.poles = []

        # In case Delta0 is not-zero, there will be no poles
        if self.Delta0_mag > 0:
            return

        # If Delta0 is zero, there will be three possible
        # poles, one on the left and one on the right and 
        # one within Delta. 
        # determine where they are based on the intersection
        # between eps - eps_a - Lambda = 0
        eps_function = self.create_adsorbate_line
        Lambda = self.create_Lambda_reg

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

    def create_dos(self, eps) -> acb:
        """Create the density of states."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_arb
        Delta0 = self.create_Delta0_arb
        Lambda = self.create_Lambda_arb
        # Create the density of states
        numerator = Delta(eps) + Delta0(eps) 
        denominator = ( eps_function(eps) - Lambda(eps) )**2 + ( Delta(eps) + Delta0(eps) )**2 
        return numerator / denominator / acb.pi()

    def calculate_filling(self) -> float:
        """Calculate the filling from the metal density of states."""
        self._convert_to_float()
        # Filling contribution coming from the d-states
        filling_numerator = integrate.quad(self.create_Delta_reg, 
                            self.eps_min, 0,
                            limit=100)[0]
        # Filling contribution coming from the sp-states
        filling_numerator += integrate.quad(self.create_Delta0_reg,
                             self.eps_min, 0,
                             limit=100)[0]
        # Filling contribution to the denomitor coming from the d-states 
        filling_denominator = integrate.quad(self.create_Delta_reg,
                            self.eps_min, self.eps_max,
                            limit=100)[0]
        # Filling contribution to the denomitor coming from the sp-states
        filling_denominator += integrate.quad(self.create_Delta0_reg,
                                self.eps_min, self.eps_max,
                                limit=100)[0]
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
                                Lambda_prime = self.create_Lambda_prime_arb(pole)
                                assert Lambda_prime.real <= arb(0.0)
                                localised_occupancy += acb('1.0') / (acb('1.0') - Lambda_prime)

            # # Add in the integral for the states within the Delta function
            lower_integration_bound = min(0.0, float((self.eps_d - self.wd).real) )
            upper_integration_bound = min(0.0, float((self.eps_d + self.wd).real) )
            self.na = acb.integral(lambda x, _: self.create_dos(x),
                                                lower_integration_bound,
                                                upper_integration_bound)
            self.na += localised_occupancy

        else:
            self._convert_to_acb()
            # Numerically integrate the dos to find the occupancy
            self.na = acb.integral(lambda x, _: self.create_dos(x), 
                                                self.eps_min, 
                                                arb('0.0'), 
                                                rel_tol=np.power(2, -self.precision/2))
        if self.verbose:
            print(f'Single particle occupancy: {self.na}')

    def create_energy_integrand(self, eps):
        """Create the energy integrand of the system."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_reg
        Delta0 = self.create_Delta0_reg
        Lambda = self.create_Lambda_reg

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
    
    def calculate_energy(self):
        """Calculate the energy from the Newns-Anderson model."""

        # We do not need multi-precision for this calculation
        self._convert_to_float()
        self.find_poles_green_function()

        poles_to_consider = [pole for pole in self.poles if pole is not None]
        delta_E_ = integrate.quad(self.create_energy_integrand, 
                            self.eps_min, 0,
                            points = tuple(poles_to_consider),
                            limit=100)[0]

        self.DeltaE = delta_E_ * 2 / np.pi 
        self.DeltaE -= 2 * self.eps_a

        # Check if DeltaE is positive and within the NUMERICAL_NOISE_THRESHOLD
        if self.DeltaE > 0 and self.DeltaE < self.NUMERICAL_NOISE_THRESHOLD:
            self.DeltaE = 0

        if self.verbose:
            print(f'Energy of the system: {self.DeltaE} eV')