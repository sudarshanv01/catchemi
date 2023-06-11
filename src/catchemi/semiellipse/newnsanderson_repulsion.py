"""Perform fitting for the parameters in the model."""

import numpy as np
from catchemi import NewnsAndersonLinearRepulsion, NewnsAndersonGrimleyRepulsion


class FitParametersNewnsAnderson:
    """Class for fitting the Newns-Anderson model to the
    DFT energies.

    Vsd: list
        The coupling element of the Newns-Anderson model.
    width: list
           The width of the d-states of the model. Note that
           it is not the total width but that between the
           d-band centre and the edge of the semi-ellipse
    eps_a: float
           Renormalised energy of the adsorbate.
    eps_sp_max: float
           Maximum energy of the sp-states.
    eps_sp_min: float
           Minimum energy of the sp-states.
    Delta0_mag: float
           Augmentation of the sp states into the d-states.
    eps: list
           The range of energy values on which the fitting is done.
    store_hyb_energies: bool
           If True, the hybridisation energies and
           orthogonalisation energies are stored.
    no_of_bonds: list
           The number of bonds for each material.
    type_repulsion: str
            Choose between 'linear', 'linear_mod' and 'grimley'
            where linear is just the two-state repulsion, linear_mod
            is the two-state repulsion with the modification of the
            large-S contributions and grimley is the Grimley repulsion.

    Outputs:

    chemi_energy: The chemisorption energy for each material.
    """

    def __init__(self, **kwargs):
        """Here we convert everything into numpy arrays."""
        self.Vsd = kwargs.get("Vsd", None)
        self.width = kwargs.get("width", None)
        self.eps_a = kwargs.get("eps_a", None)
        self.eps_sp_max = kwargs.get("eps_sp_max", 15)
        self.eps_sp_min = kwargs.get("eps_sp_min", -15)
        self.Delta0_mag = kwargs.get("Delta0_mag", 0.0)
        self.precision = kwargs.get("precision", 50)
        self.verbose = kwargs.get("verbose", False)
        self.eps = kwargs.get("eps", np.linspace(-30, 10))
        self.store_hyb_energies = kwargs.get("store_hyb_energies", False)
        self.no_of_bonds = kwargs.get("no_of_bonds", np.ones(len(self.Vsd)))
        self.spin = kwargs.get("spin", 2)
        self.type_repulsion = kwargs.get("type_repulsion", "linear")

        self.validate_inputs()

    def validate_inputs(self):
        """Check if everything is the same length and
        has the right value."""
        assert self.Vsd != None, "Vsd is not defined."
        assert self.width != None, "width is not defined."
        assert self.eps_a != None, "eps_a is not defined."
        assert self.type_repulsion in [
            "linear",
            "linear_mod",
            "grimley",
        ], "type_repulsion must be 'linear' or 'grimley'."
        if isinstance(self.eps_a, list):
            print("Multiple eps_a have been passed.")
        if isinstance(self.eps_a, float) or isinstance(self.eps_a, int):
            self.eps_a = [self.eps_a]

    def _validate_guesses(self, alpha, beta, constant_offset):
        """Check the validity of the length of the input."""
        # Each input quantity must be exactly
        # as long as the number of single particle states
        # used in the computation.

        if isinstance(alpha, float):
            alpha = [alpha]
        if isinstance(beta, float):
            beta = [beta]
        if isinstance(constant_offset, float):
            constant_offset = [constant_offset]

        assert len(alpha) == len(self.eps_a)
        assert len(beta) == len(self.eps_a)
        assert len(constant_offset) == len(self.eps_a)

        return alpha, beta, constant_offset

    def fit_parameters(self, args, eps_ds) -> np.ndarray:
        """Fit parameters of alpha, beta and constant offset
        of the NewnsAndersonModel including repulsive interations
        to DFT energies."""

        # alpha, beta, constant_offset = args
        alpha = args[0 : len(self.eps_a)]
        beta = args[len(self.eps_a) : 2 * len(self.eps_a)]
        constant_offset = args[2 * len(self.eps_a) :]
        # Validate the inputs
        alpha, beta, constant_offset = self._validate_guesses(
            alpha, beta, constant_offset
        )

        # Make sure that all the quantities are positive
        # Constant offset can be any sign
        alpha = [abs(a) for a in alpha]
        beta = [abs(b) for b in beta]

        # Determine the chemisorption energy for the
        # materials for which we have eps_d values
        chemi_energy = []
        # Hybridisation energies if needed
        hybridisation_energies = []
        # Orthogonalisation energies if needed
        orthogonalisation_energies = []
        # Store the occupancy
        occupancies = []
        # Store the filling
        filling_factor = []

        for i, eps_d in enumerate(eps_ds):
            Vsd = self.Vsd[i]
            width = self.width[i]

            # Choose the function to use for the repulsive
            # contributions based on the type of repulsion used
            if self.type_repulsion in ["linear", "linear_mod"]:
                fitting_class = NewnsAndersonLinearRepulsion
            elif self.type_repulsion == "grimley":
                fitting_class = NewnsAndersonGrimleyRepulsion

            # Iterate over each single particle state to get
            # a different value of the energies.
            hyb_energy_i = []
            ortho_energy_i = []
            occ_i = []
            filling_i = []
            chemi_energy_i = []

            for eps_a, alpha_i, beta_i, constant_offset_i in zip(
                self.eps_a, alpha, beta, constant_offset
            ):
                chemisorption = fitting_class(
                    Vsd=Vsd,
                    eps_a=eps_a,
                    eps_d=eps_d,
                    width=width,
                    eps=self.eps,
                    Delta0_mag=self.Delta0_mag,
                    eps_sp_max=self.eps_sp_max,
                    eps_sp_min=self.eps_sp_min,
                    precision=self.precision,
                    verbose=self.verbose,
                    alpha=alpha_i,
                    beta=beta_i,
                    constant_offset=constant_offset_i,
                    spin=self.spin,
                )

                if self.type_repulsion == "linear_mod":
                    # Make sure that the largeS contribution is
                    # used when the type of repulsion is linear_mod
                    chemisorption.add_largeS_contribution = True

                # Store the chemisorption energy
                e_chem = chemisorption.get_chemisorption_energy()
                chemi_energy_i.append(e_chem)
                # Store the hybridisation energies
                e_hyb = chemisorption.get_hybridisation_energy()
                hyb_energy_i.append(e_hyb)
                # Store the orthogonalisation energies
                e_ortho = chemisorption.get_orthogonalisation_energy()
                ortho_energy_i.append(e_ortho)
                # Store the occupancy
                occ_i.append(chemisorption.get_occupancy())
                # Store the filling
                filling_i.append(chemisorption.get_dband_filling())

            # Store the energies
            if len(self.eps_a) > 1:
                # Sum over the energies of the single particle states
                chemi_energy.append(np.sum(chemi_energy_i))
            else:
                chemi_energy += chemi_energy_i

            if self.store_hyb_energies:
                if len(self.eps_a) > 1:
                    hybridisation_energies.append(np.sum(hyb_energy_i))
                    orthogonalisation_energies.append(np.sum(ortho_energy_i))
                    # Occupancies can be reported separately
                    occupancies.append(occ_i)
                    # Filling factor can be reported separately
                    filling_factor.append(filling_i)
                else:
                    hybridisation_energies.extend(hyb_energy_i)
                    orthogonalisation_energies.extend(ortho_energy_i)
                    occupancies.extend(occ_i)
                    filling_factor.extend(filling_i)

        # Multiply the chemisorption energies with the
        # number of bonds
        chemi_energy = np.multiply(chemi_energy, self.no_of_bonds)

        if self.store_hyb_energies:
            # Same treatment of bonds as the chemisorption energy
            self.hyb_energy = np.multiply(hybridisation_energies, self.no_of_bonds)
            self.ortho_energy = np.multiply(
                orthogonalisation_energies, self.no_of_bonds
            )
            # The occupancy is store as per bond
            self.occupancy = np.array(occupancies)
            # Store the fractional filling factor
            self.filling_factor = np.array(filling_factor)

        # Write out the parameters
        if self.verbose:
            print("alpha:", alpha)
            print("beta:", beta)
            print("constant_offset:", constant_offset)
            print("")

        return chemi_energy
