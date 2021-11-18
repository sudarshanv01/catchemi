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
    Vsd: list
    filling: list
    width: list
    eps_a: float
    eps_min: float = -15
    eps_max: float = 15

    def __post_init__(self):
        """Extra variables that are needed for the model."""
        assert self.eps_min < self.eps_max, "eps_min must be smaller than eps_max"
        self.eps = np.linspace(self.eps_min, self.eps_max, 1000)
        # convert all lists to numpy arrays
        self.Vsd = np.array(self.Vsd)
        self.filling = np.array(self.filling)
        self.width = np.array(self.width)

    def fit_parameters(self, parameters, eps_ds):#alpha, beta, Delta0):
        """Fit the parameters alpha, beta, Delta0."""
        if len(parameters) != 3:
            Delta0 = 3.
            alpha = parameters[0]
            beta = parameters[1]
        else:
            alpha, beta, Delta0 = parameters
        # Always use the absolute value 
        alpha = abs(alpha)
        beta = abs(beta)
        Delta0 = abs(Delta0)
        # All the parameters here will have positive values
        # Vak assumed to be proportional to Vsd
        Vak = np.sqrt(beta) * self.Vsd

        # Store the hybridisation energy for all metals to compare later
        spd_hybridisation_energy = np.zeros(len(eps_ds))

        # We will need the occupancy of the single particle state
        na = np.zeros(len(eps_ds))

        # Loop over all the metals
        for i, eps_d in enumerate(eps_ds):
            hybridisation = NewnsAndersonNumerical(
                Vak = Vak[i],
                eps_a = self.eps_a,
                eps_d = eps_d,
                width = self.width[i],
                eps = self.eps,
                Delta0 = Delta0,
            )

            # The first component of the hybridisation energy
            # is the hybdridisation coming from the sp and d bands
            hybridisation.calculate_energy()
            spd_hybridisation_energy[i] = hybridisation.get_energy()
            # Get and store the occupancy because it will be needed
            # for the orthogonalisation energy term
            hybridisation.calculate_occupancy()
            na[i] = hybridisation.get_occupancy()

        # Ensure that the hybridisation energy is negative always
        assert all(spd_hybridisation_energy <= 0), "Hybridisation energy is negative"
        # Store the spd hybridisation energy
        self.spd_hybridisation_energy = spd_hybridisation_energy

        # orthonogonalisation energy
        ortho_energy = 2 * ( na +  self.filling ) * alpha * np.sqrt(beta) * self.Vsd**2
        ortho_energy = np.array(ortho_energy)

        # Ensure that the orthonogonalisation energy is positive always
        assert all(ortho_energy >= 0), "Orthogonalisation energy is positive now it is (%1.2f)"%(ortho_energy)

        # Add the orthogonalisation energy to the hybridisation energy
        hybridisation_energy = spd_hybridisation_energy + ortho_energy

        # Store the hybridisation energy for all metals
        self.hybridisation_energy = hybridisation_energy

        # Store the occupancies as well
        self.na = na

        # Store the orthogonalisation energy for all metals
        self.ortho_energy = ortho_energy
        
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
    Delta0: float
    precision: int = 50

    def __post_init__(self):
        """Perform numerical calculations of the Newns-Anderson model 
        to get the chemisorption energy."""
        self.eps_min = np.min(self.eps)
        self.eps_max = np.max(self.eps) 
        self.wd = self.width / 2
        self.eps = np.array(self.eps)
        print(f'Solving the Newns-Anderson model for eps_a = {self.eps_a} eV and eps_d = {self.eps_d}')

        # Choose the precision to which the quantities are calculated
        ctx.dps = self.precision
        
    def _convert_to_acb(self):
        """Convert the important quantities to arb so 
        that they can be manipulated freely."""
        self.eps_min = acb(self.eps_min)
        self.eps_max = acb(self.eps_max)
        self.wd = acb(self.wd)
        self.eps_d = acb(self.eps_d)
        self.eps_a = acb(self.eps_a)
        self.Vak = acb(self.Vak)
        self.Delta0 = acb(self.Delta0)
    
    def _convert_to_float(self):
        self.eps_min = float(self.eps_min.real)
        self.eps_max = float(self.eps_max.real)
        self.wd = float(self.wd.real)
        self.eps_d = float(self.eps_d.real)
        self.eps_a = float(self.eps_a.real)
        self.Vak = float(self.Vak.real)
        self.Delta0 = float(self.Delta0.real)

    def get_energy(self):
        """Get the hybridisation energy."""
        return self.DeltaE

    def get_occupancy(self):
        """Get the occupancy of the single particle state."""
        return float(self.na.real)
    
    def get_dos_on_grid(self):
        """Get the density of states."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_reg
        Lambda = self.create_Lambda_reg
        self._convert_to_float()
        dos = np.zeros(len(self.eps))
        for i, eps_ in enumerate(self.eps):
            numerator = Delta(eps_) + self.Delta0
            denominator = ( eps_function(eps_) - Lambda(eps_) )**2 + ( Delta(eps_) + self.Delta0 ) **2 
            dos[i] = numerator / denominator / np.pi
        return dos

    def get_Delta_on_grid(self):
        """Get Delta on supplied grid."""
        Delta = self.create_Delta_reg
        self._convert_to_float()
        Delta_val = np.array([Delta(e) for e in self.eps])
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

    def create_reference_eps(self, eps):
        """Create the reference energy for finding Delta and Lambda."""
        return ( eps - self.eps_d ) / self.wd 

    def create_Delta_arb(self, eps):
        """Create a function for Delta based on arb."""
        eps_ref = self.create_reference_eps(eps) 
        # If the absolute value of the reference is 
        # lower than 1 (in the units of 2width) then
        # the Delta will be non-zero 
        if acb.abs_lower(eps_ref) < arb(1): 
            Delta = acb(1.)  -  acb.pow(eps_ref, 2)
            Delta = acb.pow(Delta, 0.5)
            # Multiply by the prefactor
            Delta *= acb.pi()**2 * self.Vak
            # Normalise the area
            Delta /= self.wd
            Delta *= acb(2)
        else:
            # If eps is out of bounds there will
            # no Delta contribution
            Delta = acb(0.0)
        return Delta

    def create_Delta_reg(self, eps):
        """Create a function for Delta for regular manipulations."""
        eps_ref = self.create_reference_eps(eps) 
        # If the absolute value of the reference is 
        # lower than 1 (in the units of 2width) then
        # the Delta will be non-zero 
        if np.abs(eps_ref) < 1: 
            Delta = 1.  -  eps_ref**2
            Delta = Delta**0.5 
            # Multiply by the prefactor
            Delta *= np.pi**2 * self.Vak
            # Normalise the area
            Delta /= self.wd
            Delta *= 2
        else:
            # If eps is out of bounds there will
            # no Delta contribution
            Delta = 0.0 
        return Delta

    def create_Lambda_arb(self, eps):
        """Create the hilbert transform of Delta with arb."""
        eps_ref = self.create_reference_eps(eps)

        if eps_ref.real < arb(-1): 
            # Below the lower edge of the d-band
            Lambda = eps_ref + acb.pow( eps_ref**2 - acb(1), 0.5 )
            Lambda *= acb.pi()**2 * self.Vak
        elif eps_ref.real > arb(1):
            # Above the upper edge of the d-band
            Lambda = eps_ref - acb.pow( eps_ref**2 - acb(1), 0.5 )
            Lambda *= acb.pi()**2 * self.Vak
        elif acb.abs_lower(eps_ref) <= arb(1): 
            # Inside the d-band
            Lambda = eps_ref
            Lambda *= acb.pi()**2 * self.Vak
        else:
            raise ValueError(f'eps_ = {eps} cannot be considered in Lambda')
        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to Vak^2
        Lambda /= self.wd
        Lambda *= acb(2)
        return Lambda

    def create_Lambda_reg(self, eps):
        """Create the hilbert transform of Delta for regular manipulations."""
        eps_ref = self.create_reference_eps(eps)

        if eps_ref < -1: 
            # Below the lower edge of the d-band
            Lambda = eps_ref + ( eps_ref**2 - 1 )**0.5
            Lambda *= np.pi**2 * self.Vak
        elif eps_ref > 1:
            # Above the upper edge of the d-band
            Lambda = eps_ref - ( eps_ref**2 - 1 )**0.5
            Lambda *= np.pi**2 * self.Vak
        elif eps_ref <= 1: 
            # Inside the d-band
            Lambda = eps_ref
            Lambda *= np.pi**2 * self.Vak
        else:
            raise ValueError(f'eps_ = {eps} cannot be considered in Lambda')
        # Same normalisation for Lambda as for Delta
        # These are prefactors of Delta that have been multiplied
        # with Delta to ensure that the area is set to Vak^2
        Lambda /= self.wd
        Lambda *= 2
        return Lambda

    def create_Lambda_prime_arb(self, eps):
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
            Lambda_prime = 1
        Lambda_prime *= self.Vak
        Lambda_prime /= self.wd**2
        Lambda_prime *= acb.pi()**2
        Lambda_prime *= acb(2)
        return Lambda_prime

    def create_adsorbate_line(self, eps):
        """Create the line that the adsorbate passes through."""
        return eps - self.eps_a

    def find_poles_green_function(self, eps):
        """Find the poles of the green function. In the case that Delta=0
        these points will not be the poles, but are important to pass 
        on to the integrator anyhow."""
        self.poles = []
        # In case Delta0 is not-zero, there will be no poles
        if self.Delta0 > 0:
            return
        # If Delta0 is zero, there will be two possible
        # poles, one on the left and one on the right
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
        try:
            pole_lower = optimize.brentq(lambda x: eps_function(x) - Lambda(x), 
                                         self.eps_min,
                                         self.eps_d - self.wd,)
            self.poles.append(pole_lower)
        except ValueError:
            pass
        try:
            pole_higher = optimize.brentq(lambda x: eps_function(x) - Lambda(x),
                                          self.eps_d + self.wd,
                                          self.eps_max )
            self.poles.append(pole_higher)
        except ValueError:
            pass
        print(f'Poles of the green function:{self.poles}')

    def create_dos(self, eps):
        """Create the density of states."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_arb
        Lambda = self.create_Lambda_arb
        # Create the density of states
        numerator = Delta(eps) + self.Delta0
        denominator = ( eps_function(eps) - Lambda(eps) )**2 + ( Delta(eps) + self.Delta0 ) **2 
        return numerator / denominator / acb.pi()

    def calculate_occupancy(self):
        """Calculate the density of states from the Newns-Anderson model."""
        # If a dos is required, then switch to arb
        if self.Delta0 == 0:
            # Determine the points of the singularity
            self.find_poles_green_function(self.eps_a)
            self._convert_to_acb()
            localised_occupancy = acb(0.0)
            for pole in self.poles:
                if pole.real < arb(0.0):
                    Lambda_prime = self.create_Lambda_prime_arb(pole)
                    assert Lambda_prime.real < arb(0.0)
                    localised_occupancy += acb(1) / (acb(1) - Lambda_prime)
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
            self.na = acb.integral(lambda x, _: self.create_dos(x), self.eps_min, arb(0))

        print(f'Single particle occupancy: {self.na}')

    def create_energy_integrand(self, eps):
        """Create the energy integrand of the system."""
        eps_function = self.create_adsorbate_line
        Delta = self.create_Delta_reg
        Lambda = self.create_Lambda_reg

        # Determine the energy of the system
        numerator = Delta(eps) + self.Delta0
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
    
    def create_sp_integrand(self, eps):
        """Create the energy integrand for the sp states."""
        eps_function = self.create_adsorbate_line

        # Determine the energy of the system
        numerator = self.Delta0
        denominator = eps_function(eps) # Lambda for a constant is 0.

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
        self.find_poles_green_function(self.eps)
        delta_E_ = integrate.quad(self.create_energy_integrand, 
                            self.eps_min, 0,
                            points = (self.poles),
                            limit=100)[0]
        # delta_E0_ = integrate.quad(self.create_sp_integrand,
        #                     self.eps_min, 0,
        #                     limit=100)[0]
        # delta_E_ += delta_E0_
        self.DeltaE = delta_E_ * 2 / np.pi 
        self.DeltaE -= 2 * self.eps_a
        print(f'Energy of the system: {self.DeltaE} eV')

@dataclass
class NewnsAndersonAnalytical:
    """ Perform the Newns-Anderson model analytically for a semi-elliplical delta.
        Inputs 
        ------
        eps_a: float
            renormalised energy of the adsorbate in units of eV wrt Fermi level
        beta_p: float
            Coupling element of the adsorbate with the metal atom in units of 2beta 
        beta: float
            Coupling element of the metal with metal in units of eV
        eps_d: float
            center of Delta in units of eV wrt Fermi level
        eps: list
            Range of energies to plot in units of eV wrt d-band center 
        fermi_energy: float
            Fermi energy in the units of eV
        U: float
            Coulomb interaction parameter in units of eV
    """ 
    beta_p: float
    eps_a: float
    eps: list
    eps_d : float
    beta: float
    fermi_energy: float
    U: float
    grid_size = 20

    def __post_init__(self):
        """Setup the quantities for a self-consistent calculation."""
        # Setup spin polarised calculation with different up and down spins
        # To determine the lowest energy spin configuration determine the variation
        # n_-sigma with a fixed n_sigma and vice-versa. The point where the two
        # curves meet is the self-consistency point.
        self.nsigma_range = np.linspace(0, 1.0, self.grid_size)
        self.nmsigma_range = np.linspace(0, 1.0, self.grid_size)
        # Unit conversion details
        # coversion factor to divide by to convert to 2beta units
        self.convert = 2 * self.beta
        self.U = self.U / self.convert
        self.eps_a = self.eps_a / self.convert
        self.eps = self.eps / self.convert 
        self.eps_d = self.eps_d / self.convert
        self.fermi_energy = self.fermi_energy / self.convert
        # The quantities that will be of interest here 
        self.Delta = np.zeros(len(self.eps))
        self.Lambda = np.zeros(len(self.eps))
        self.rho_aa = np.zeros(len(self.eps))
        # Print out details of the quantities
        input_data = {
            "beta_p": self.beta_p,
            "eps_a": self.eps_a,
            "eps_d": self.eps_d,
            "beta": self.beta,
            "fermi_energy": self.fermi_energy,
            "U": self.U,
        }
        pprint(input_data)

    
    def self_consistent_calculation(self):
        """ Calculate the self-consistency point for the given parameters."""
        if self.U == 0:
            # There is no columb interaction, so the self-consistency point is
            print('No need for self-consistent calculation, U=0')
            self.n_minus_sigma = 1.
            self.n_plus_sigma = 1.
        else:
            # Find the lowest maximum value for varying n_down
            lowest_energy = None
            index_nup_overall = None
            index_ndown_overall = None
            # Store all the energies of the grid
            self.energies_grid = np.zeros((len(self.nsigma_range), len(self.nmsigma_range)))
            for j, n_down in enumerate(self.nmsigma_range):
                # Fix n_minus sigma and determine energies for different 
                # values of n_plus sigma
                energies_down = np.zeros(len(self.nsigma_range))
                for i, n_up in enumerate(self.nsigma_range):

                    # Define the energies for up and down spins
                    self.eps_sigma_up = self.eps_a + self.U * n_down 
                    self.eps_sigma_down = self.eps_a + self.U * n_up

                    # Calculate the 1electron energies
                    # First calculate the spin up energy
                    self.eps_sigma = self.eps_sigma_up
                    self.calculate_energies()
                    energies_down_ = self.DeltaE_1sigma
                    # Now calculate the spin down energy
                    self.eps_sigma = self.eps_sigma_down
                    self.calculate_energies()
                    energies_down_ += self.DeltaE_1sigma

                    # Calculate the coulomb contribution
                    coulomb_energy = self.U * n_down * n_up
                    energies_down_ -= coulomb_energy

                    # Subtract the adsorbate energy
                    energies_down_ -= self.eps_a 

                    # Store the energies to check if it is the lowest
                    energies_down[i] = energies_down_

                # determine the maximum value of the energy
                maximum_energy = np.max(energies_down)
                index_nup = np.argmax(energies_down)
                # Store the energies for the grid
                self.energies_grid[:, j] = energies_down
                
                # Choose whether to store this energy or not
                if lowest_energy is not None:
                    lowest_energy = maximum_energy if maximum_energy <= lowest_energy else lowest_energy
                    index_nup_overall = index_nup if maximum_energy <= lowest_energy else index_nup_overall
                    index_ndown_overall = j if maximum_energy <= lowest_energy else index_ndown_overall
                else:
                    # First run, store these quantities
                    lowest_energy = maximum_energy
                    index_nup_overall = index_nup
                    index_ndown_overall = j
            
            # Determine the values that give the lowest energy
            self.n_minus_sigma = self.nmsigma_range[index_ndown_overall]
            self.n_plus_sigma = self.nsigma_range[index_nup_overall]


        # Store all the quantities for the self-consistency point
        self.eps_sigma_up = self.eps_a + self.U * self.n_minus_sigma 
        self.eps_sigma_down = self.eps_a + self.U * self.n_plus_sigma

        # Sum up the energies from both of the spins
        self.eps_sigma = self.eps_sigma_up
        self.calculate_energies()
        self.rho_aa_up = self.rho_aa
        DeltaE_ = self.DeltaE_1sigma
        self.eps_sigma = self.eps_sigma_down
        self.calculate_energies()
        self.rho_aa_down = self.rho_aa
        DeltaE_ += self.DeltaE_1sigma

        # The variable rhoaa will be the sum of the two rhos
        self.rho_aa = self.rho_aa_up + self.rho_aa_down 

        # Coulomb contribution
        DeltaE_ -= self.U * self.n_minus_sigma * self.n_plus_sigma
        # Adsorbate energy difference
        DeltaE_ -= self.eps_a
        # Fermi energy
        DeltaE_ += self.fermi_energy

        # The converged energy
        self.DeltaE = DeltaE_ 

        # Print out the final results
        print('--------------------------')
        print(f"Spin up expectation value   : {self.n_minus_sigma} e")
        print(f"Spin down expectation value : {self.n_plus_sigma} e")
        print(f"Self-consistency energy     : {self.DeltaE} (2beta)")


    def calculate_energies(self):
        """Calculate the 1e energies from the Newns-Anderson model."""

        # Energies referenced to the d-band center
        # Needed for some manipulations later
        self.eps_wrt_d = self.eps - self.eps_d
        self.eps_sigma_wrt_d = self.eps_sigma - self.eps_d

        # Construct Delta in units of 2beta
        self.width_of_band =  1 
        self.Delta = 2 * self.beta_p**2 * ( 1 - self.eps_wrt_d**2 )**0.5
        self.Delta = np.nan_to_num(self.Delta)

        # Calculate the positions of the upper and lower band edge
        self.lower_band_edge = - self.width_of_band + self.eps_d
        self.upper_band_edge = + self.width_of_band + self.eps_d
        index_lower_band_edge = np.argmin(np.abs(self.eps - self.lower_band_edge))
        index_upper_band_edge = np.argmin(np.abs(self.eps - self.upper_band_edge))
        self.Delta_at_lower_band_edge = self.Delta[index_lower_band_edge]
        self.Delta_at_upper_band_edge = self.Delta[index_upper_band_edge]

        # Construct Lambda in units of 2beta
        lower_hilbert_args = []
        upper_hilbert_args = []
        for i, eps in enumerate(self.eps_wrt_d):
            if np.abs(eps) <= self.width_of_band: 
                self.Lambda[i] = 2 * self.beta_p**2 * eps 
            elif eps > self.width_of_band:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps - (eps**2 - 1)**0.5 )
                upper_hilbert_args.append(i)
            elif eps < -self.width_of_band:
                self.Lambda[i] = 2 * self.beta_p**2 * ( eps + (eps**2 - 1)**0.5 )
                lower_hilbert_args.append(i)
            else:
                raise ValueError("The epsilon value is not valid.")

        self.Lambda_at_lower_band_edge = self.Lambda[index_lower_band_edge]
        self.Lambda_at_upper_band_edge = self.Lambda[index_upper_band_edge]

        # ---------------- Adsorbate density of states ( in the units of 2 beta)
        rho_aa_ = self.eps_wrt_d**2 * ( 1 - 4 * self.beta_p**2 )
        rho_aa_ += - 2 * self.eps_wrt_d * self.eps_sigma_wrt_d * ( 1 - 2 * self.beta_p**2 )
        rho_aa_ += 4 *self.beta_p**4 + self.eps_sigma_wrt_d**2
        self.rho_aa = 2 * self.beta_p**2 * ( 1 - self.eps_wrt_d**2 )**0.5
        self.rho_aa /= rho_aa_ 
        self.rho_aa /= np.pi
        self.rho_aa = np.nan_to_num(self.rho_aa)

        # ---------------- Check all the possible root combinations ----------------
        # Check if there is a virtual root
        if 4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 < 1:
            self.has_complex_root = True
        else:
            self.has_complex_root = False 
        
        if not self.has_complex_root:
            if self.beta_p != 0.5:
                root_positive = ( 1 - 2*self.beta_p**2 ) *  self.eps_sigma_wrt_d
                root_positive += 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5
                root_positive /= ( 1 - 4 * self.beta_p**2 )
                root_negative = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma_wrt_d
                root_negative -= 2*self.beta_p**2 * (4*self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**0.5
                root_negative /= ( 1 - 4 * self.beta_p**2 )
            else:
                root_positive = 1 + 4 * self.eps_sigma_wrt_d**2
                root_positive /= ( 4 * self.eps_sigma_wrt_d)
                root_negative = root_positive

        elif self.has_complex_root:
            root_positive = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma \
                            + 2j * self.beta_p**2 * ( 1 - 4 * self.beta_p**2 - self.eps_sigma**2 )**0.5 
            root_positive /= ( 1 - 4 * self.beta_p**2)
            root_negative = ( 1 - 2*self.beta_p**2 ) * self.eps_sigma \
                            - 2j * self.beta_p**2 * ( 1 - 4 * self.beta_p**2 - self.eps_sigma**2 )**0.5 
            root_negative /= ( 1 - 4 * self.beta_p**2)

            # We do not care about the imaginary root for now
            root_positive = np.real(root_positive)
            root_negative = np.real(root_negative)

        # Store the root referenced to the energy reference scale that was chosen 
        self.root_positive = root_positive + self.eps_d #- self.fermi_energy
        self.root_negative = root_negative + self.eps_d #- self.fermi_energy
        
        # Determine if there is an occupied localised state
        if not self.has_complex_root:
            if self.root_positive < self.lower_band_edge and self.eps_sigma_wrt_d < 2 * self.beta_p**2 - 1:
                # Check if the root is below the Fermi level
                if self.root_positive < self.fermi_energy:
                    # the energy for this point is to be included
                    # print('Positive root is below the Fermi level.')
                    self.has_localised_occupied_state_positive = True
                else:
                    self.has_localised_occupied_state_positive = False
                # in both cases it is appropriate to store it as eps_l_sigma because it is localised state
                self.eps_l_sigma_pos = self.root_positive
            else:
                self.eps_l_sigma_pos = None
                self.has_localised_occupied_state_positive = False
            
            # Check if there is a localised occupied state for the negative root
            if self.root_negative > self.upper_band_edge and self.eps_sigma_wrt_d > 1 - 2 * self.beta_p**2:
                # Check if the root is below the Fermi level
                if self.root_negative < self.fermi_energy:
                    # the energy for this point is to be included
                    # print('Negative root is below the Fermi level.')
                    self.has_localised_occupied_state_negative = True
                else:
                    self.has_localised_occupied_state_negative = False
                # in both cases it is appropriate to store it as eps_l_sigma because it is localised state
                self.eps_l_sigma_neg = self.root_negative
            else:
                self.eps_l_sigma_neg = None
                self.has_localised_occupied_state_negative = False

            # Expectancy value of the occupied localised state
            if self.has_localised_occupied_state_positive:
                # Compute the expectancy value
                if self.beta_p != 0.5:
                    self.na_sigma_pos = (1 - 2 * self.beta_p**2)
                    self.na_sigma_pos += 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**-0.5 
                    self.na_sigma_pos /= (1 - 4 * self.beta_p**2)
                else:
                    self.na_sigma_pos = 4 * self.eps_sigma_wrt_d**2 - 1
                    self.na_sigma_pos /= (4 * self.eps_sigma_wrt_d**2)
            else:
                self.na_sigma_pos = 0.0
            
            if self.has_localised_occupied_state_negative:
                # Compute the expectancy value
                if self.beta_p != 0.5:
                    self.na_sigma_neg = (1 - 2 * self.beta_p**2)
                    self.na_sigma_neg -= 2 * self.beta_p**2 * self.eps_sigma_wrt_d * (4 * self.beta_p**2 + self.eps_sigma_wrt_d**2 - 1)**-0.5 
                    self.na_sigma_neg /= (1 - 4 * self.beta_p**2)
                else:
                    self.na_sigma_neg = 4 * self.eps_sigma_wrt_d**2 - 1
                    self.na_sigma_neg /= (4 * self.eps_sigma_wrt_d**2)
            else:
                self.na_sigma_neg = 0.0
        else:
            # This is a complex root
            assert self.has_complex_root
            self.has_localised_occupied_state_positive = False
            self.has_localised_occupied_state_negative = False
            self.na_sigma_neg = 0
            self.na_sigma_pos = 0

        # ---------- Calculate the energy ----------
        # Determine the upper bounds for the contour integration
        if self.upper_band_edge > self.fermi_energy:
            upper_bound = self.fermi_energy
        else:
            upper_bound = self.upper_band_edge
        occupied_states = [i for i in range(len(self.eps)) if self.lower_band_edge < self.eps[i] < upper_bound]

        # Determine the integrand 
        energy_occ = self.eps_wrt_d[occupied_states]
        numerator = - 2 * self.beta_p**2 * (1 - energy_occ**2)**0.5
        numerator = np.nan_to_num(numerator)
        denominator = energy_occ * (2*self.beta_p**2 - 1) + self.eps_sigma_wrt_d

        # This number will always be between [-pi, 0]
        arctan_integrand = np.arctan2(numerator, denominator)
        assert all(arctan_integrand < 0)
        assert all(arctan_integrand > -np.pi)

        if self.has_localised_occupied_state_positive and self.has_localised_occupied_state_negative:
            # Both positive and negative root are localised and occupied
            arctan_integrand += np.pi
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy += self.eps_l_sigma_pos 
            self.energy -= self.eps_l_sigma_neg 
        elif self.has_localised_occupied_state_positive:
            # Has only positive root and it is a localised occupied state 
            arctan_integrand += np.pi
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy += self.eps_l_sigma_pos
            self.energy -= self.fermi_energy
        elif self.has_localised_occupied_state_negative:
            # Has only negative root and it is a localised occupied state
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component
            self.energy -= self.eps_l_sigma_neg
            self.energy += self.upper_band_edge
        else:
            # Has no localised occupied states
            self.arctan_component =  np.trapz( arctan_integrand, energy_occ )
            self.arctan_component /= np.pi
            self.energy = self.arctan_component

        # The one electron energy is just the difference of eigenvalues 
        self.DeltaE_1sigma = self.energy 
        # assert self.na_sigma_pos + self.na_sigma_neg <= 1.0
        # assert self.na_sigma_pos >= 0
        # assert self.na_sigma_neg >= 0
        # self.DeltaE_1sigma -= ( self.na_sigma_pos + self.na_sigma_neg ) * self.eps_sigma
        # self.DeltaE_1sigma -= self.eps_sigma
