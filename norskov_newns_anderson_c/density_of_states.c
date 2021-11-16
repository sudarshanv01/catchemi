#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arb.h>
#include <acb.h>
#include<acb_calc.h>
#include "quantities_newns_anderson.h"

// Compute the density of states from the Newns-Anderson model.
void get_density_of_states(acb_t dos, double Vak, double eps, double eps_d, double wd, double eps_a, double Delta0, int prec)
{
    // Define the variables to find the dos.
    acb_t Delta;
    acb_t Lambda;
    acb_t energy_diff;
    acb_t numerator_dos;
    acb_t denominator_1dos;
    acb_t denominator_2dos;
    acb_t denominator_dos;
    acb_t pi;

    // Initialise those variables
    acb_init(Delta);
    acb_init(Lambda);
    acb_init(energy_diff);
    acb_init(numerator_dos);
    acb_init(denominator_1dos);
    acb_init(denominator_2dos);
    acb_init(denominator_dos);
    acb_init(pi);

    // Set constant value of pi
    acb_const_pi(pi, prec);
    
    // Calculate Delta
    get_Delta_semiellipse(Delta, Vak, eps, eps_d, wd, prec);

    // Calculate Lambda
    get_Lambda_semiellipse(Lambda, Vak, eps, eps_d, wd, prec);

    // Calculate the energy difference
    get_energy_difference(energy_diff, eps, eps_a, prec);

    // Calculate the density of states
    acb_set(numerator_dos, Delta);
    acb_add_si(numerator_dos, numerator_dos, Delta0, prec);
    acb_div(numerator_dos, numerator_dos, pi, prec);
    acb_sub(denominator_1dos, energy_diff, Lambda, prec);
    acb_sqr(denominator_1dos, denominator_1dos, prec);
    acb_set(denominator_2dos, Delta);
    acb_add_si(denominator_2dos, denominator_2dos, Delta0, prec);
    acb_sqr(denominator_2dos, denominator_2dos, prec);
    acb_add(denominator_dos, denominator_1dos, denominator_2dos, prec);
    acb_div(dos, numerator_dos, denominator_dos, prec);

}

void get_occupancy(acb_t occ, double Vak, double eps, double eps_d, double wd, double eps_a, 
                    double Delta0, double eps_min, int prec)
{
    acb_calc_func_t func;

    // Integrate the density of states from eps_min to 0
    // acb_calc_integral(occ, get_density_of_states, Vak, eps, eps_d, wd, eps_a, Delta0, eps_min, prec);
}