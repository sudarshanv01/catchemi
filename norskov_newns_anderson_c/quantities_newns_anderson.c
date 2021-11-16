#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arb.h"
#include "acb.h"

// Define the Delta function in acb
void get_Delta_semiellipse(acb_t Delta, double Vak_f, double eps_f, double eps_d_f, double wd_f, int prec)
{
    acb_t Vak, eps, eps_d, wd, norm_eps;
    arb_t norm_eps_real;
    arb_t lower_energy, upper_energy;
    double lower_energy_f = -1.0;
    double upper_energy_f = 1.0;

    // Initialise the acb_t variables
    acb_init(Vak);
    acb_init(eps);
    acb_init(eps_d);
    acb_init(wd);
    acb_init(norm_eps);
    // Initialise the arb variables
    arb_init(lower_energy);
    arb_init(upper_energy);
    arb_init(norm_eps_real);

    // Get the acb form of the input variables
    acb_set_d(Vak, Vak_f);
    acb_set_d(eps, eps_f);
    acb_set_d(eps_d, eps_d_f);
    acb_set_d(wd, wd_f);
    arb_set_d(lower_energy, lower_energy_f);
    arb_set_d(upper_energy, upper_energy_f);

    // Store the norm_eps variable
    acb_sub(norm_eps, eps, eps_d, prec);
    acb_div(norm_eps, norm_eps, wd, prec);
    // Store also the real part of norm_eps
    acb_get_real(norm_eps_real, norm_eps);

    // Initialise the Delta variable
    // If the norm_eps is greater than 1, set Delta to zero
    // If the norm_eps is less than -1, set Delta to zero
    // Otherwise, calculate Delta
    if (arb_gt(norm_eps_real, upper_energy))
    {
        acb_set_ui(Delta, 0);
    }
    else if (arb_lt(norm_eps_real, lower_energy))
    {
        acb_set_ui(Delta, 0);
    }
    else
    {
        acb_set(Delta, norm_eps);
        acb_sqr(Delta, Delta, prec);
        acb_neg(Delta, Delta);
        acb_add_ui(Delta, Delta, 1, prec);
        acb_sqrt(Delta, Delta, prec);
        acb_mul(Delta, Delta, Vak, prec);
        acb_mul(Delta, Delta, Vak, prec);
        acb_div(Delta, Delta, wd, prec);
    }
}

void get_Lambda_semiellipse(acb_t Lambda_d, double Vak_f, double eps_f, double eps_d_f, double wd_f, int prec)
{
    acb_t Vak, eps, eps_d, wd, norm_eps;
    arb_t lower_energy, upper_energy;
    arb_t norm_eps_real;
    double lower_energy_f = -1.0;
    double upper_energy_f = 1.0;

    // Initialise the acb_t variables
    acb_init(Vak);
    acb_init(eps);
    acb_init(eps_d);
    acb_init(wd);
    acb_init(norm_eps);
    // Initialise the arb variables
    arb_init(lower_energy);
    arb_init(upper_energy);
    arb_init(norm_eps_real);

    // Get the acb form of the input variables
    acb_set_d(Vak, Vak_f);
    acb_set_d(eps, eps_f);
    acb_set_d(eps_d, eps_d_f);
    acb_set_d(wd, wd_f);
    arb_set_d(lower_energy, lower_energy_f);
    arb_set_d(upper_energy, upper_energy_f);

    // Store the norm_eps variable
    acb_sub(norm_eps, eps, eps_d, prec);
    acb_div(norm_eps, norm_eps, wd, prec);
    // Store also the real part of norm_eps
    acb_get_real(norm_eps_real, norm_eps);

    // Initialise the Lambda_d variable
    if ( arb_gt(norm_eps_real, upper_energy) ) 
    {
        printf("norm_eps > upper_energy\n");
        acb_set(Lambda_d, norm_eps);
        acb_sqr(Lambda_d, Lambda_d, prec);
        acb_sub_ui(Lambda_d, Lambda_d, 1, prec);
        acb_sqrt(Lambda_d, Lambda_d, prec);
        acb_neg(Lambda_d, Lambda_d);
        acb_add(Lambda_d, Lambda_d, norm_eps, prec);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_div(Lambda_d, Lambda_d, wd, prec);
    }
    else if ( arb_gt(lower_energy, norm_eps_real) )
    {
        printf("norm_eps < lower_energy\n");
        acb_set(Lambda_d, norm_eps);
        acb_sqr(Lambda_d, Lambda_d, prec);
        acb_sub_ui(Lambda_d, Lambda_d, 1, prec);
        acb_sqrt(Lambda_d, Lambda_d, prec);
        acb_add(Lambda_d, Lambda_d, norm_eps, prec);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_div(Lambda_d, Lambda_d, wd, prec);
    }
    else
    {
        printf("norm_eps >= lower_energy and norm_eps <= upper_energy\n");
        acb_set(Lambda_d, norm_eps);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_mul(Lambda_d, Lambda_d, Vak, prec);
        acb_div(Lambda_d, Lambda_d, wd, prec);
    }
}

void get_energy_difference(acb_t energy_difference, double eps_f, double eps_a_f, int prec)
{
    acb_t eps, eps_a;

    // Initialise the acb_t variables
    acb_init(eps);
    acb_init(eps_a);

    // Get the acb form of the input variables
    acb_set_d(eps, eps_f);
    acb_set_d(eps_a, eps_a_f);

    // Calculate the energy difference
    acb_sub(energy_difference, eps, eps_a, prec);
}