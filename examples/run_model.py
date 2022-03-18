"""Example of routine to fit the model."""
import json
import yaml
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from scipy import odr
import matplotlib.pyplot as plt
from ase import units
from catchemi import NewnsAndersonLinearRepulsion, FitParametersNewnsAnderson

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def create_coupling_elements(s_metal, s_Cu, 
    anderson_band_width, anderson_band_width_Cu, 
    r=None, r_Cu=None, normalise_bond_length=False,
    normalise_by_Cu=True):
    """Create the coupling elements based on the Vsd
    and r values. The Vsd values are identical to those
    used in Ruban et al. The assume that the bond lengths
    between the metal and adsorbate are the same. Everything
    is referenced to Cu, as in the paper by Ruban et al."""
    Vsdsq = s_metal**5 * anderson_band_width
    Vsdsq_Cu = s_Cu**5 * anderson_band_width_Cu 
    if normalise_by_Cu:
        Vsdsq /= Vsdsq_Cu
    if normalise_bond_length:
        assert r is not None
        if normalise_by_Cu: 
            assert r_Cu is not None
            Vsdsq *= r_Cu**8 / r**8
        else:
            Vsdsq /= r**8
    return Vsdsq

if __name__ == '__main__':
    """Determine the fitting parameters for a particular adsorbate."""

    # Choose a sequence of adsorbates
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1] # eV
    EPS_VALUES = np.linspace(-30, 10, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f"inputs/pdos_moments.json")) 
    data_from_energy_calculation = json.load(open(f"inputs/adsorption_energies.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    Vsdsq_data = data_from_LMTO['Vsdsq']
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    site = 'hollow' # choose between hollow and ontop


    # Plot the fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(6.75, 3), constrained_layout=True)
    for i in range(len(ax)):
        ax[i].set_xlabel('DFT energy (eV)')
        ax[i].set_ylabel('Chemisorption energy (eV)')
        ax[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []

        for metal in data_from_energy_calculation[adsorbate]:

            # get the parameters from DFT calculations
            width = data_from_dos_calculation[metal]['width']
            parameters['width'].append(width)
            d_band_centre = data_from_dos_calculation[metal]['d_band_centre']
            parameters['d_band_centre'].append(d_band_centre)

            # get the parameters from the energy calculations
            adsorption_energy = data_from_energy_calculation[adsorbate][metal]
            if isinstance(adsorption_energy, list):
                dft_energies.append(np.min(adsorption_energy))
            else:
                dft_energies.append(adsorption_energy)
            
            # Get the bond length from the LMTO calculations
            bond_length = data_from_LMTO['s'][metal]*units.Bohr
            bond_length_Cu = data_from_LMTO['s']['Cu']*units.Bohr 

            Vsdsq = create_coupling_elements(s_metal=s_data[metal],
                s_Cu=s_data['Cu'],
                anderson_band_width=anderson_band_width_data[metal],
                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                r=bond_length,
                r_Cu=bond_length_Cu,
                normalise_bond_length=True,
                normalise_by_Cu=True)
            # Report the square root
            Vsd = np.sqrt(Vsdsq)
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

            # Get the number of bonds based on the 
            # DFT calculation
            parameters['no_of_bonds'].append(no_of_bonds[site][metal])

        # Prepare the class for fitting routine 
        kwargs_fit = dict(
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = CONSTANT_DELTA0,
            Vsd = parameters['Vsd'],
            width = parameters['width'],
            eps_a = eps_a,
            verbose = True,
            no_of_bonds = parameters['no_of_bonds'],
        )
        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)

        initial_guess = [0.01, np.pi*0.6, 0.1]
        
        print('Initial guess: ', initial_guess)

        # Finding the fitting parameters
        data = odr.RealData(parameters['d_band_centre'], dft_energies)
        fitting_model = odr.Model(fitting_function.fit_parameters)
        fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        fitting_odr.set_job(fit_type=2)
        output = fitting_odr.run()

        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(output.beta, parameters['d_band_centre'])

        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.6, np.max(dft_energies)+0.6, 2)
        ax[i].plot(x, x, '--', color='tab:grey', linewidth=1)
        # Fix the axes to the same scale 
        ax[i].set_xlim(np.min(x), np.max(x))
        ax[i].set_ylim(np.min(x), np.max(x))

        texts = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = 'red'
            elif metal in SECOND_ROW:
                colour = 'orange'
            elif metal in THIRD_ROW:
                colour = 'green'
            ax[i].plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            texts.append(ax[i].text(dft_energies[j], optimised_hyb[j], metal, color=colour, ))

        ax[i].set_aspect('equal')

        # Write out the fitted parameters as a json file
        json.dump({
            'alpha': abs(output.beta[0]),
            'beta': abs(output.beta[1]),
            'delta0': CONSTANT_DELTA0, 
            'constant_offset': output.beta[2],
            'eps_a': eps_a,
        }, open(f'{adsorbate}_parameters.json', 'w'))

    fig.savefig(f'fitting_example.png', dpi=300)