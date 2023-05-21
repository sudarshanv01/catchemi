"""Perform a simple calculation to get the chemisorption energy."""
from distutils.debug import DEBUG
import os
import logging
import json
from typing import Dict, List, Tuple

from collections import defaultdict

import numpy as np
import numpy.typing as nptyp
import scipy
from scipy import optimize
from scipy import signal

import matplotlib.pyplot as plt


class SimpleChemisorption:
    """Base class to perform a simple calculation
    to get the chemisorption energy."""

    debug_dir = "debug"
    # Occassionally, during the optimisation process
    # we will have a situation where the E_hyb is positive
    # This positive number is numerically caused. Allow
    # a little bit of noise (but E_hyb) will be set to 0
    # if the value is greater than 0 and below INTEGRAL_NOISE.
    INTEGRAL_NOISE = 0.3

    def __init__(
        self,
        dft_dos: List[float],
        dft_energy_range: List[float],
        Vak: float,
        Sak: float,
        Delta0: float,
        eps_a: float,
        no_of_bonds: int,
        DEBUG: bool = False,
    ):
        self.dft_dos = dft_dos
        self.eps = dft_energy_range
        self.Vak = Vak
        self.Sak = Sak
        self.Delta0 = Delta0
        self.eps_a = eps_a
        self.no_of_bonds = no_of_bonds
        self.DEBUG = DEBUG

        self._validate_inputs()
        # Get the quantities that will be used again and
        # again in the calculation.
        self.get_Delta()
        self.get_Lambda()

    def _validate_inputs(self):
        """Throw up assertion errors if the inputs
        are not in accordance with what is expected."""
        # Ensure that the integral of the density of states is 1.
        # That is, the d-projected density of states is normalised
        # to 1.
        integral_d_dos = np.trapz(self.dft_dos, self.eps)
        assert np.allclose(integral_d_dos, 1.0)
        assert self.Sak <= 0.0, "Sak must be negative or zero."
        assert self.Vak >= 0.0, "Vak must be positive"

        # Make everything a numpy array
        self.dft_dos = np.array(self.dft_dos)
        self.eps = np.array(self.eps)

        # Delta0 must be a float and greater than 0
        assert isinstance(self.Delta0, float), "Delta0 must be a float"
        assert self.Delta0 >= 0.0, "Delta0 must be positive"

        # Create debug directory if it does not exist
        if not os.path.exists(self.debug_dir):
            os.mkdir(self.debug_dir)

    def get_Delta(self) -> nptyp.ArrayLike:
        """Get Vak by multiplying the d-density of
        states by the following relations:
        Delta = pi * Vak**2 * rho_d
        where rho_d is the DFT (normalised) density
        of states.
        """
        self.Delta = np.pi * self.Vak**2 * self.dft_dos
        return self.Delta

    def get_Lambda(self) -> nptyp.ArrayLike:
        """Get the Hilbert transform of the Delta array."""
        self.Lambda = np.imag(scipy.signal.hilbert(self.Delta + self.Delta0))
        return self.Lambda

    def get_chemisorption_energy(self) -> float:
        """The chemisorption energy is the sum of the
        hybridisation energy and the orthogonalisation energy."""

        self.E_chemi = self.E_hyb + self.E_ortho
        logging.debug(
            f"Chemisorption energy: {self.E_chemi:0.2f} eV, Hybridisation energy: {self.E_hyb:0.2f} eV, Orthogonalisation energy: {self.E_ortho:0.2f} eV; n_a = {self.n_a:0.2f} e"
        )
        return self.E_chemi

    def get_hybridisation_energy(self) -> float:
        """Get the hybridisation energy on the basis
        of the DFT density of states and the Newns-Anderson
        model energy."""

        # Refresh Delta and Lambda
        self.get_Delta()
        self.get_Lambda()

        ind_below_fermi = np.where(self.eps <= 0)[0]

        # Create the integral and just numerically integrate it.
        integrand_numer = self.Delta[ind_below_fermi] + self.Delta0
        integrand_denom = (
            self.eps[ind_below_fermi] - self.eps_a - self.Lambda[ind_below_fermi]
        )

        arctan_integrand = np.arctan2(integrand_numer, integrand_denom)
        arctan_integrand -= np.pi

        # Make sure that the integral is within limits, otherwise
        # something went wrong with the arctan.
        assert np.all(arctan_integrand <= 0), "Arctan integrand must be negative"
        assert np.all(
            arctan_integrand >= -np.pi
        ), "Arctan integrand must be greater than -pi"

        E_hyb = np.trapz(arctan_integrand, self.eps[ind_below_fermi])

        E_hyb *= 2
        E_hyb /= np.pi

        E_hyb -= 2 * self.eps_a

        # Multiply by the number of bonds
        E_hyb *= self.no_of_bonds

        self.E_hyb = E_hyb

        logging.debug(
            f"Hybridisation energy: {self.E_hyb:0.2f} eV for eps_a: {self.eps_a:0.2f} eV"
        )
        try:
            assert self.E_hyb <= 0.0, "Hybridisation energy must be negative"
        except AssertionError as e:
            if self.E_hyb < self.INTEGRAL_NOISE:
                logging.warning(
                    f"Numerically, E_hyb is very slightly greater than 0 ({self.E_hyb:0.2f} eV). Setting it to 0."
                )
                self.E_hyb = 0.0
            else:
                # Store the calculation.
                if self.DEBUG:
                    # Plot the function of the energy_integrand
                    fig, ax = plt.subplots(
                        1, 1, figsize=(6, 4), constrained_layout=True
                    )
                    ax.plot(
                        self.eps[ind_below_fermi],
                        arctan_integrand,
                        color="k",
                        label="integral",
                    )
                    ax.fill_between(
                        self.eps[ind_below_fermi],
                        0,
                        arctan_integrand,
                        color="k",
                        alpha=0.2,
                    )
                    # Plot the components of the integral as well
                    ax2 = ax.twinx()
                    ax2.plot(
                        self.eps[ind_below_fermi],
                        self.Delta[ind_below_fermi],
                        color="C0",
                        label="$\Delta$",
                    )
                    ax2.plot(
                        self.eps[ind_below_fermi],
                        self.Delta0 * np.ones(len(self.eps[ind_below_fermi])),
                        color="C1",
                        label="$\Delta_0$",
                    )
                    ax2.plot(
                        self.eps[ind_below_fermi],
                        self.Lambda[ind_below_fermi],
                        color="C2",
                        label="$\Lambda$",
                    )
                    ax2.plot(
                        self.eps[ind_below_fermi],
                        self.eps[ind_below_fermi] - self.eps_a,
                        color="C3",
                        label="$\epsilon - \epsilon_a$",
                    )
                    ax.set_xlabel("Energy (eV)")
                    ax.set_ylabel("arctan(Integrand)")
                    ax2.set_ylabel("Parameters")
                    ax2.set_ylim([np.min(self.Lambda), np.max(self.Delta)])
                    logging.warning(
                        f"Plotting the components of the integral to {self.debug_dir}/integrand.png"
                    )
                    fig.savefig(os.path.join(self.debug_dir, "integrand.png"), dpi=300)
                    plt.close(fig)

                # Cannot proceed, error with Hybridisation energy.
                raise e

        return self.E_hyb

    def get_adsorbate_dos(self) -> nptyp.ArrayLike:
        """Get the adsorbate projected density of states."""
        numerator = self.Delta + self.Delta0
        denominator = (self.eps - self.eps_a - self.Lambda) ** 2
        denominator += (self.Delta + self.Delta0) ** 2
        self.rho_a = numerator / denominator / np.pi
        return self.rho_a

    def get_occupancy(self) -> float:
        """Integrate up rho_a upto the Fermi level."""
        # Integrate up rho_a upto the Fermi level.
        index_ = np.where(self.eps <= 0)[0]
        self.n_a = np.trapz(self.rho_a[index_], self.eps[index_])
        return self.n_a

    def get_filling(self) -> float:
        """Integrate up Delta to get the filling of
        the d-density of states of the metal."""
        denominator = np.trapz(self.Delta, self.eps)
        # Get the index of self.eps that are lower than 0
        index_ = np.where(self.eps < 0)[0]
        numerator = np.trapz(self.Delta[index_], self.eps[index_])
        self.filling = numerator / denominator
        assert self.filling >= 0.0, "Filling must be positive"
        return self.filling

    def get_orthogonalisation_energy(self) -> float:
        """Get the orthogonalisation energy on the basis of the
        smeared out two-state equation."""
        self.get_Delta()
        self.get_Lambda()
        self.get_adsorbate_dos()
        self.get_occupancy()
        self.get_filling()
        E_ortho = -2 * (self.n_a + self.filling) * self.Vak * self.Sak
        # Mutliply by the number of bonds
        E_ortho *= self.no_of_bonds
        self.E_ortho = E_ortho
        assert self.E_ortho >= 0.0, "Orthogonalisation energy must be positive"
        return self.E_ortho


class SemiEllipseHypothetical(SimpleChemisorption):
    """Given an arbitrary set of eps_d and w_d values, create
    a meshgrid of the chemisorption, hybridisation and orthogonalisation energies.
    Multiple eps_a is supported and added together."""

    def __init__(
        self,
        json_params: str,
        eps_d_list: nptyp.ArrayLike,
        w_d_list: nptyp.ArrayLike,
        eps: nptyp.ArrayLike,
        Vsd: float,
    ):
        self.json_params = json_params
        self.eps_d_list = eps_d_list
        self.w_d_list = w_d_list
        self.eps = eps
        self.Vsd = Vsd

        # Determine the grid size on the basis of the input parameters.
        self.GRID_SIZE = len(self.eps_d_list)
        assert self.GRID_SIZE == len(
            self.w_d_list
        ), "The number of eps_d and w_d must be equal"

        self._read_parameters()

    def _read_parameters(self) -> None:
        """Read the input parameters from the input file."""
        with open(self.json_params) as handle:
            data = json.load(handle)
        self.eps_a_list = data["eps_a"]
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.gamma = data["gamma"]
        self.factor_mult_list = [1] + data["epsilon"]
        self.Delta0 = data["delta0"]
        self.no_of_bonds_list = data["no_of_bonds_list"]

    def _create_semi_ellipse(self, eps_d: float, w_d: float) -> None:
        """Create a semi-ellipse for the d-density of states."""
        energy_ref = (self.eps - eps_d) / w_d
        delta = np.zeros(len(self.eps))
        for i, eps_ in enumerate(energy_ref):
            if np.abs(eps_) < 1:
                delta[i] = (1 - eps_**2) ** 0.5
        # We will need a very tight grid in order to make
        # the normalisation perfect. So in this case, we will
        # proceed by setting the integral of the semi-ellipse to 1.
        delta /= np.trapz(delta, self.eps)
        # This parameter is out `dft_dos`, even though it is constructed
        self.dft_dos = delta

    def generate_meshgrid(
        self,
    ) -> Tuple[nptyp.ArrayLike, nptyp.ArrayLike, nptyp.ArrayLike]:
        """Generate meshgrid of chemisorption, hybridisation
        and orthogonalisation energies."""

        self.hyb_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.chem_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.ortho_energy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.occupancy_meshgrid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        for index_epsd, eps_d in enumerate(self.eps_d_list):
            for index_wd, w_d in enumerate(self.w_d_list):

                # Define the d-density of states.
                self._create_semi_ellipse(eps_d, w_d)

                # If there are more than one eps_a values, add them up.
                for index_epsa, eps_a in enumerate(self.eps_a_list):

                    # Generate the adsorbate specific parameters
                    self.eps_a = eps_a
                    self.no_of_bonds = self.no_of_bonds_list[index_epsa]
                    self.Vak = (
                        self.factor_mult_list[index_epsa]
                        * np.sqrt(self.beta)
                        * self.Vsd
                    )
                    self.Sak = (
                        self.factor_mult_list[index_epsa] * -1 * self.alpha * self.Vak
                    )

                    self._validate_inputs()

                    # Get the chemisorption and hybridisation energies.
                    self.hyb_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_hybridisation_energy()
                    self.ortho_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_orthogonalisation_energy()
                    self.chem_energy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_chemisorption_energy()

                    # Store also the occupancies
                    self.occupancy_meshgrid[
                        index_epsd, index_wd
                    ] += self.get_occupancy()

        # Add gamma to the chemisorption energy.
        self.chem_energy_meshgrid += self.gamma
        self.hyb_energy_meshgrid += self.gamma

        return (
            self.hyb_energy_meshgrid,
            self.ortho_energy_meshgrid,
            self.chem_energy_meshgrid,
            self.occupancy_meshgrid,
        )


class AdsorbateChemisorption(SimpleChemisorption):
    """Perform the adsorbate chemisorption analysis
    based on an arbitrary set of parameters. Expected
    input is data related to Vak and the density of states
    from a DFT calculation.

    This class largely wraps around the SimpleChemisorption class
    to ensure that the lists are dealt with correctly.
    """

    def __init__(
        self,
        dos_data: Dict,
        Vak_data: Dict,
        Sak_data: Dict,
        Delta0: float,
        eps_a_data: List,
        no_of_bonds_list: List[int],
        DEBUG: bool = False,
    ):

        self.Delta0 = Delta0
        self.DEBUG = DEBUG

        # Treat each adsorbate separately.
        model_outputs = defaultdict(dict)

        for identifier, dos_dict in dos_data.items():
            # Each parameter here is a unique surface.
            self.dft_dos = dos_dict["dft_dos"]
            self.eps = dos_dict["eps"]
            Vak_list = Vak_data[identifier]
            Sak_list = Sak_data[identifier]

            for i, eps_a in enumerate(eps_a_data):
                # Iterate over the adsorbate
                self.eps_a = eps_a

                # Store the float values of Vak and Sak
                self.Vak = Vak_list[i]
                self.Sak = Sak_list[i]
                self.no_of_bonds = no_of_bonds_list[i]

                self._validate_inputs()

                # Compute the chemisorption energy.
                e_hyb = self.get_hybridisation_energy()
                e_ortho = self.get_orthogonalisation_energy()
                e_chemi = self.get_chemisorption_energy()

                # Store the results.
                model_outputs[identifier][eps_a] = {
                    "hyb": e_hyb,
                    "ortho": e_ortho,
                    "chemi": e_chemi,
                }
        self.model_outputs = model_outputs


class FittingParameters:
    """Given a json file with the dos and energy stored, and
    some information about epsilon_a values, perform the fitting
    procedure to determine alpha, beta and a _single_ constant.

    Args:
        json_filename (str): The name of the json file containing the dos and energy.
        eps_a_data (list): The list of epsilon_a values to use.
        DEBUG (bool): Whether to print debug information.
        Delta0 (float): The value of Delta0 to use, fixed constant.
        return_extended_output (bool): Whether to return the extended output.
    """

    def __init__(
        self,
        json_filename: str,
        eps_a_data: List,
        Delta0: float,
        DEBUG: bool = False,
        return_extended_output: bool = False,
        no_of_bonds_list: List[int] = None,
    ):
        self.json_filename = json_filename
        self.eps_a_data = eps_a_data
        self.Delta0 = Delta0
        self.DEBUG = DEBUG
        self.return_extended_output = return_extended_output
        self.no_of_bonds_list = no_of_bonds_list

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate inputs to make sure everything has
        the right dimensions and type."""
        assert isinstance(self.eps_a_data, list), "eps_a_data must be a list"
        assert isinstance(self.Delta0, float), "Delta0 must be a float"

        # Convert the eps_a_data to a numpy array.
        self.eps_a_data = np.array(self.eps_a_data)

    def load_data(self):
        """Load the data from the json file."""
        with open(self.json_filename, "r") as f:
            data = json.load(f)
        self.data = data

    def objective_function(
        self,
        x: Tuple,
    ) -> float:
        """Objective function to be minimised.

        Args:
            x (Tuple): The parameters to be used in the minimisation.
                       Include alpha, beta and gamma (fixed constant)
                       and a list of epsilon values, epsilon is used
                       to determine the scaling between each of the adsorbates.
        """

        # Single set of parameters
        alpha, beta, gamma, *epsilon = x

        # make sure both alpha and beta are always positive
        alpha = np.abs(alpha)
        beta = np.abs(beta)
        epsilon = np.abs(epsilon)

        # Create a list called factor_mult_list which contains
        # the factor_mult_epsa for each eps_a value from epsilon
        factor_mult_list = [1] + [epsilon[i] for i in range(len(epsilon))]
        factor_mult_list = np.array(factor_mult_list)

        # Prepare inputs to the Model class.
        inputs = defaultdict(lambda: defaultdict(dict))

        # Parse common inputs
        inputs["Delta0"] = self.Delta0
        inputs["eps_a_data"] = self.eps_a_data
        inputs["DEBUG"] = self.DEBUG
        inputs["no_of_bonds_list"] = self.no_of_bonds_list

        for _id in self.data:
            # Parse the relevant quantities from the
            # supplied dictionary.
            pdos = self.data[_id]["pdos"]
            energy_grid = self.data[_id]["energy_grid"]
            Vsd = self.data[_id]["Vsd"]
            # Make Vsd an array
            Vsd = np.array(Vsd)

            # Store the inputs.
            inputs["dos_data"][_id]["dft_dos"] = pdos
            inputs["dos_data"][_id]["eps"] = energy_grid
            inputs["Vak_data"][_id] = factor_mult_list * np.sqrt(beta) * Vsd
            inputs["Sak_data"][_id] = -factor_mult_list * alpha * Vsd

        # Get the outputs
        model_energies_class = AdsorbateChemisorption(**inputs)
        model_outputs = model_energies_class.model_outputs

        # Compute the RMSE value for the difference between
        # the model and DFT data.
        mean_absolute_error = 0.0
        for _id in model_outputs:
            # What we will compare against
            ads_energy_DFT = self.data[_id]["ads_energy"]
            # Construct the model energy
            model_energy = 0.0
            # Add separately for each eps_a
            for eps_a in model_outputs[_id]:
                model_energy += model_outputs[_id][eps_a]["chemi"]

            # Add a constant parameter gamma
            model_energy += gamma

            # Compute the RMSE
            sq_error = np.abs(model_energy - ads_energy_DFT)
            mean_absolute_error += sq_error

        # Return the RMSE
        mean_absolute_error = mean_absolute_error / len(model_outputs)

        logging.info(
            f"Parameters: {x} leads to mean absolute error: {mean_absolute_error}eV"
        )

        if self.return_extended_output:
            # Useful if we want to analyse the fitting parameters.

            predicted_energy = []
            actual_energy = []
            id_order = []

            # Store the string of species as well
            species_string = []

            for _id in model_outputs:
                # What we will compare against
                ads_energy_DFT = self.data[_id]["ads_energy"]
                actual_energy.append(ads_energy_DFT)

                # Construct the model energy
                model_energy = 0.0

                # Add separately for each eps_a
                for eps_a in model_outputs[_id]:
                    model_energy += model_outputs[_id][eps_a]["chemi"]

                # Add a constant parameter gamma
                model_energy += gamma
                predicted_energy.append(model_energy)

                # Store the _id as well
                id_order.append(_id)

                # Store the species string
                spec_string = "".join(self.data[_id]["species"])
                species_string.append(spec_string)

            logging.info("Predicted and actual energies parsed.")

            # return predicted_energy, actual_energy, id_order
            output_data = {}
            output_data["predicted_energy"] = predicted_energy
            output_data["actual_energy"] = actual_energy
            output_data["id_order"] = id_order
            output_data["species_string"] = species_string

            return mean_absolute_error, output_data

        else:
            return mean_absolute_error
